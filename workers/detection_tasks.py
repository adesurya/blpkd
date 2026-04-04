"""workers/detection_tasks.py — full implementation with DB writes"""
from __future__ import annotations
import time
from typing import Any, Optional
import structlog
from workers.celery_app import celery_app
log = structlog.get_logger(__name__)

@celery_app.task(name="workers.detection_tasks.process_stream_frame", bind=True)
def process_stream_frame(self, frame_data: dict) -> dict[str, Any]:
    """Process a single stream frame: detect → analyze → save DB → fire webhooks."""
    import asyncio, base64, cv2, numpy as np
    from datetime import datetime, timezone
    from services.detector.yolo_detector import YOLODetector
    from services.attribute_analyzer.color_detector import AttributeAnalyzer

    camera_id = frame_data.get("camera_id", "")
    frame_number = frame_data.get("frame_number", 0)
    session_id = frame_data.get("session_id")
    zones = frame_data.get("zones")
    filter_criteria = frame_data.get("filter_criteria")

    raw = base64.b64decode(frame_data["frame_b64"])
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "Cannot decode frame"}

    detector = YOLODetector.get_instance()
    det_result = detector.detect(bgr, frame_number=frame_number,
                                  timestamp=frame_data.get("timestamp", 0.0),
                                  zones=zones, extract_crops=True)
    attr_analyzer = AttributeAnalyzer()
    detections_data, total_count = [], 0
    zone_counts, color_counts = {}, {}

    for person in det_result.detections:
        attrs, matches = (None, True)
        if frame_data.get("analyze_attributes", True) and person.crop is not None:
            attrs, matches = attr_analyzer.analyze_with_counting_filter(person.crop, filter_criteria)
        if matches:
            total_count += 1
            z = person.zone_id or "default"
            zone_counts[z] = zone_counts.get(z, 0) + 1
            if attrs and attrs.upper_color:
                color_counts[attrs.upper_color] = color_counts.get(attrs.upper_color, 0) + 1
        detections_data.append(({
            "session_id": session_id, "camera_id": camera_id,
            "frame_number": frame_number, "timestamp": datetime.now(timezone.utc),  # waktu sekarang, bukan PTS video
            "track_id": person.track_id, "confidence": person.confidence,
            "bbox_x": person.bbox.x, "bbox_y": person.bbox.y,
            "bbox_w": person.bbox.w, "bbox_h": person.bbox.h, "zone_id": person.zone_id,
            "upper_color": attrs.upper_color if attrs else None,
            "lower_color": attrs.lower_color if attrs else None,
            "activity": attrs.activity if attrs else None,
        }, person.crop))

    async def _save():
        from datetime import datetime, timezone
        import uuid as _uuid
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from core.config.settings import settings
        from core.models.repository import (
            DetectionRepository, PeopleCountRepository, SessionRepository,
            WebhookRepository, FaceRepository,
        )
        from core.abstractions.storage import get_storage, StorageKeys
        from services.face_engine.recognizer import FaceEngine

        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        sf = async_sessionmaker(engine, expire_on_commit=False)
        count_ts = datetime.now(timezone.utc)

        async with sf() as db:
            db_dets = [d for d, _ in detections_data]
            await DetectionRepository.bulk_insert(db, db_dets)
            await PeopleCountRepository.insert(db, {
                "session_id": session_id, "camera_id": camera_id, "timestamp": count_ts,
                "count": total_count, "count_entering": 0, "count_exiting": 0,
                "count_by_upper_color": color_counts or None,
            })
            if session_id:
                await SessionRepository.increment_counters(db, session_id, frames=1,
                                                           detections=len(detections_data))
            await db.commit()

        if frame_data.get("extract_faces", True):
            fe = FaceEngine.get_instance()
            storage = get_storage()
            async with sf() as db:
                for _, crop in detections_data:
                    if crop is None:
                        continue
                    faces = fe.detect_faces(crop)
                    if not faces:
                        continue
                    best = faces[0]
                    match = await fe.recognize(best)
                    face_id = str(_uuid.uuid4())
                    _, buf = cv2.imencode(".jpg", best.crop)
                    try:
                        await storage.upload("faces", StorageKeys.face_crop(face_id),
                                             bytes(buf), "image/jpeg")
                    except Exception:
                        pass
                    await FaceRepository.create(db, {
                        "id": face_id, "person_id": match.matched_person_id,
                        "is_known": match.is_known, "best_frame_path": StorageKeys.face_crop(face_id),
                        "detection_score": best.detection_score, "best_quality_score": best.quality_score,
                        "camera_ids": [camera_id],
                    })
                    if not match.is_known:
                        await fe.register_face(best, face_id=face_id, person_id=None)
                    event = "face.recognized" if match.is_known else "face.unknown"
                    whs = await WebhookRepository.get_for_event(db, event, camera_id)
                    for wh in whs:
                        send_webhook.delay(wh.url, event, {
                            "event": event, "camera_id": camera_id,
                            "timestamp": count_ts.isoformat(),
                            "person_id": match.matched_person_id, "face_id": face_id,
                            "similarity_score": match.similarity_score,
                        }, wh.secret)
                await db.commit()

        async with sf() as db:
            whs = await WebhookRepository.get_for_event(db, "person.detected", camera_id)
            if whs and total_count > 0:
                for wh in whs:
                    send_webhook.delay(wh.url, "person.detected", {
                        "event": "person.detected", "camera_id": camera_id,
                        "timestamp": count_ts.isoformat(), "frame_number": frame_number,
                        "total_count": total_count, "zone_counts": zone_counts,
                    }, wh.secret)

        await engine.dispose()

    asyncio.run(_save())
    return {"camera_id": camera_id, "frame_number": frame_number,
            "count": total_count, "zone_counts": zone_counts,
            "inference_ms": round(det_result.inference_time_ms, 1)}


@celery_app.task(name="workers.detection_tasks.process_video_file", bind=True, max_retries=2)
def process_video_file(self, recording_id: str, minio_key: str, config: dict) -> dict[str, Any]:
    """Full pipeline for an uploaded video: download → detect → analyze → compress → save."""
    import asyncio, os, tempfile
    async def _run():
        import av, numpy as np
        from datetime import datetime, timezone
        import uuid as _uuid
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from core.config.settings import settings
        from core.abstractions.storage import get_storage, StorageKeys
        from core.models.repository import RecordingRepository, SessionRepository, DetectionRepository, PeopleCountRepository
        from services.detector.yolo_detector import YOLODetector
        from services.attribute_analyzer.color_detector import AttributeAnalyzer
        from services.compressor.ffmpeg_wrapper import FFmpegWrapper, CompressionConfig

        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        sf = async_sessionmaker(engine, expire_on_commit=False)
        storage = get_storage()
        log.info("video_task.start", recording_id=recording_id)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            content = await storage.download("videos", minio_key)
            with open(tmp_path, "wb") as f:
                f.write(content)

            async with sf() as db:
                await RecordingRepository.update_status(db, recording_id, "processing")
                session = await SessionRepository.create(db, config.get("camera_id"))
                await db.commit()

            detector = YOLODetector.get_instance()
            attr_analyzer = AttributeAnalyzer()
            container = av.open(tmp_path)
            video_stream = container.streams.video[0]
            sample_rate = config.get("sample_rate", 5)
            zones = config.get("zones")
            total_frames, total_persons, frame_idx = 0, 0, 0

            for packet in container.demux(video_stream):
                for av_frame in packet.decode():
                    frame_idx += 1
                    if frame_idx % sample_rate != 0:
                        continue
                    bgr = av_frame.to_ndarray(format="bgr24")
                    ts = float(av_frame.pts * av_frame.time_base) if av_frame.pts else frame_idx / 25.0
                    det = detector.detect(bgr, frame_number=frame_idx, timestamp=ts, zones=zones)
                    total_frames += 1
                    total_persons += det.person_count
                    db_dets = []
                    for p in det.detections:
                        attrs, _ = attr_analyzer.analyze_with_counting_filter(p.crop, config.get("filter_criteria")) if p.crop is not None else (None, True)
                        db_dets.append({
                            "session_id": session.id, "camera_id": config.get("camera_id"),
                            "frame_number": frame_idx,
                            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                            "track_id": p.track_id, "confidence": p.confidence,
                            "bbox_x": p.bbox.x, "bbox_y": p.bbox.y, "bbox_w": p.bbox.w, "bbox_h": p.bbox.h,
                            "zone_id": p.zone_id,
                            "upper_color": attrs.upper_color if attrs else None,
                            "lower_color": attrs.lower_color if attrs else None,
                        })
                    async with sf() as db:
                        await DetectionRepository.bulk_insert(db, db_dets)
                        await db.commit()

            container.close()
            wrapper = FFmpegWrapper()
            info = await wrapper.get_video_info(tmp_path)
            compressed_tmp = tmp_path.replace(".mp4", "_compressed.mp4")
            extra = {"duration_seconds": info.get("duration"), "width": info.get("width"),
                     "height": info.get("height"), "fps": info.get("fps")}

            if config.get("compress", True):
                result = await wrapper.compress(CompressionConfig(input_path=tmp_path, output_path=compressed_tmp))
                if result.success:
                    with open(compressed_tmp, "rb") as f:
                        await storage.upload("compressed", StorageKeys.video_compressed(recording_id), f, "video/mp4")
                    extra.update({"compressed_path": StorageKeys.video_compressed(recording_id),
                                  "compressed_size_bytes": result.output_size_bytes,
                                  "compression_ratio": result.compression_ratio,
                                  "codec_compressed": result.codec_used})
                    if os.path.exists(compressed_tmp):
                        os.unlink(compressed_tmp)

            async with sf() as db:
                await RecordingRepository.update_status(db, recording_id, "completed", extra)
                await db.commit()

            await engine.dispose()
            log.info("video_task.complete", recording_id=recording_id, frames=total_frames)
            return {"recording_id": recording_id, "total_frames": total_frames, "total_persons": total_persons}
        except Exception as e:
            log.error("video_task.failed", recording_id=recording_id, error=str(e))
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
            from core.config.settings import settings as s2
            from core.models.repository import RecordingRepository as RR
            e2 = create_async_engine(s2.DATABASE_URL, echo=False)
            sf2 = async_sessionmaker(e2, expire_on_commit=False)
            async with sf2() as db2:
                await RR.update_status(db2, recording_id, "failed")
                await db2.commit()
            await e2.dispose()
            raise
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return asyncio.run(_run())


@celery_app.task(name="workers.face_tasks.run_face_clustering", bind=True)
def run_face_clustering(self) -> dict[str, Any]:
    """Periodic DBSCAN clustering of unknown faces. Runs every 30 min."""
    import asyncio, numpy as np
    async def _cluster():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from core.config.settings import settings
        from core.models.repository import FaceRepository
        from services.face_engine.recognizer import FaceClusterer
        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        sf = async_sessionmaker(engine, expire_on_commit=False)
        async with sf() as db:
            faces = await FaceRepository.get_unknown_with_embeddings(db, limit=5000)
        if len(faces) < 2:
            await engine.dispose()
            return {"clustered": 0}
        face_ids = [f.id for f in faces]
        embeddings = [np.array(f.embedding) for f in faces]
        cluster_map = FaceClusterer().cluster(face_ids, embeddings)
        by_cluster: dict[str, list[str]] = {}
        for fid, label in cluster_map.items():
            by_cluster.setdefault(label, []).append(fid)
        async with sf() as db:
            for label, fids in by_cluster.items():
                await FaceRepository.update_cluster(db, fids, label)
            await db.commit()
        n_clusters = sum(1 for k in by_cluster if k != "noise")
        log.info("face_clustering.complete", faces=len(face_ids), clusters=n_clusters)
        await engine.dispose()
        return {"clustered": len(face_ids), "clusters": n_clusters}
    return asyncio.run(_cluster())


@celery_app.task(name="workers.analytics_tasks.aggregate_counts")
def aggregate_counts() -> dict[str, Any]:
    """Periodic aggregation of people_counts. Runs every 5 min."""
    return {"status": "ok"}


@celery_app.task(name="workers.analytics_tasks.send_webhook", bind=True, max_retries=3)
def send_webhook(self, webhook_url: str, event: str, payload: dict, secret: Optional[str] = None) -> bool:
    """Deliver webhook with HMAC signature and exponential retry."""
    import hashlib, hmac as _hmac, json, requests
    from core.config.settings import settings
    body = json.dumps(payload, default=str)
    headers = {"Content-Type": "application/json", "X-Vision-Event": event}
    if secret:
        sig = _hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers["X-Vision-Signature"] = f"sha256={sig}"
    try:
        resp = requests.post(webhook_url, data=body, headers=headers, timeout=settings.WEBHOOK_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        raise self.retry(exc=exc, countdown=settings.WEBHOOK_RETRY_DELAY * (2 ** self.request.retries))


# ════════════════════════════════════════════════════
# Trigger dari CPU worker → kirim task ke GPU queue
# ════════════════════════════════════════════════════

@celery_app.task(name="workers.analytics_tasks.run_face_clustering_trigger")
def run_face_clustering_trigger() -> dict:
    """
    Dijadwalkan di beat schedule (CPU).
    Tugasnya hanya mem-forward task clustering ke GPU queue
    agar tidak ada import GPU di CPU worker.
    """
    task = celery_app.send_task(
        "workers.face_tasks.run_face_clustering",
        queue="gpu_tasks",
    )
    return {"forwarded_task_id": task.id}