"""
workers/detection_tasks.py

Pipeline deteksi lengkap:
  1. Terima frame (base64) dari stream reader via Celery
  2. YOLOv8 detect orang + crop bounding box
  3. Attribute analysis (warna baju)
  4. Face detection + recognition (opsional)
  5. Simpan detections + people_counts ke PostgreSQL
  6. Upload frame snapshot ke MinIO jika ada orang (BARU)
  7. Upload face crops ke MinIO (opsional)
  8. Fire webhooks (person.detected, face.recognized, face.unknown)
"""
from __future__ import annotations

import time
from typing import Any, Optional

import structlog

from workers.celery_app import celery_app

log = structlog.get_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# TASK 1: process_stream_frame
# Dipanggil setiap frame dari live stream RTSP/m3u8
# ════════════════════════════════════════════════════════════════════

@celery_app.task(name="workers.detection_tasks.process_stream_frame", bind=True)
def process_stream_frame(self, frame_data: dict) -> dict[str, Any]:
    """
    Process satu frame dari live stream:
    detect -> analyze -> save DB -> snapshot -> face -> webhook.

    Parameters
    ----------
    frame_data : dict
        frame_b64         : str   - JPEG frame dalam base64
        camera_id         : str   - UUID kamera
        session_id        : str   - UUID sesi streaming
        frame_number      : int   - nomor frame dari stream
        timestamp         : float - PTS video (dalam detik)
        zones             : dict  - polygon zona counting (opsional)
        filter_criteria   : dict  - filter warna/aktivitas (opsional)
        extract_faces     : bool  - aktifkan face recognition
        analyze_attributes: bool  - aktifkan analisis warna baju
        save_frame_snapshot: bool - simpan gambar frame ke MinIO (BARU)
        detection_confidence: float - threshold YOLOv8
    """
    import asyncio
    import base64

    import cv2
    import numpy as np
    from datetime import datetime, timezone

    from services.detector.yolo_detector import YOLODetector
    from services.attribute_analyzer.color_detector import AttributeAnalyzer

    # Ambil parameter dari frame_data
    camera_id     = frame_data.get("camera_id", "")
    frame_number  = frame_data.get("frame_number", 0)
    session_id    = frame_data.get("session_id")
    zones         = frame_data.get("zones")
    filter_criteria = frame_data.get("filter_criteria")
    extract_faces = frame_data.get("extract_faces", False)
    analyze_attrs = frame_data.get("analyze_attributes", False)
    save_snapshot = frame_data.get("save_frame_snapshot", True)  # default aktif

    # Decode frame dari base64 ke numpy BGR
    raw = base64.b64decode(frame_data["frame_b64"])
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "Cannot decode frame"}

    # YOLOv8 Detection
    detector   = YOLODetector.get_instance()
    det_result = detector.detect(
        bgr,
        frame_number=frame_number,
        timestamp=frame_data.get("timestamp", 0.0),
        zones=zones,
        extract_crops=True,
    )

    # Attribute Analysis & counting
    attr_analyzer   = AttributeAnalyzer()
    detections_data = []   # list of (db_dict, crop_ndarray)
    total_count     = 0
    zone_counts: dict[str, int] = {}
    color_counts: dict[str, int] = {}

    for person in det_result.detections:
        attrs   = None
        matches = True

        if analyze_attrs and person.crop is not None:
            attrs, matches = attr_analyzer.analyze_with_counting_filter(
                person.crop, filter_criteria
            )

        if matches:
            total_count += 1
            z = person.zone_id or "default"
            zone_counts[z] = zone_counts.get(z, 0) + 1
            if attrs and attrs.upper_color:
                color_counts[attrs.upper_color] = color_counts.get(attrs.upper_color, 0) + 1

        detections_data.append((
            {
                "session_id":   session_id,
                "camera_id":    camera_id,
                "frame_number": frame_number,
                "timestamp":    datetime.now(timezone.utc),  # waktu nyata, bukan PTS video
                "track_id":     person.track_id,
                "confidence":   person.confidence,
                "bbox_x":       person.bbox.x,
                "bbox_y":       person.bbox.y,
                "bbox_w":       person.bbox.w,
                "bbox_h":       person.bbox.h,
                "zone_id":      person.zone_id,
                "upper_color":  attrs.upper_color if attrs else None,
                "lower_color":  attrs.lower_color if attrs else None,
                "activity":     attrs.activity if attrs else None,
            },
            person.crop,
        ))

    # Async: save ke DB + upload snapshot + face + webhook
    async def _save():
        import uuid as _uuid
        from datetime import datetime, timezone

        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from core.config.settings import settings
        from core.models.repository import (
            DetectionRepository,
            FaceRepository,
            PeopleCountRepository,
            SessionRepository,
            WebhookRepository,
        )
        from core.abstractions.storage import get_storage, StorageKeys
        from services.face_engine.recognizer import FaceEngine

        engine  = create_async_engine(settings.DATABASE_URL, echo=False)
        Session = async_sessionmaker(engine, expire_on_commit=False)
        storage = get_storage()
        count_ts = datetime.now(timezone.utc)

        # 1. Simpan detections + people_count ke DB
        async with Session() as db:
            db_dets = [d for d, _ in detections_data]
            await DetectionRepository.bulk_insert(db, db_dets)
            await PeopleCountRepository.insert(db, {
                "session_id":           session_id,
                "camera_id":            camera_id,
                "timestamp":            count_ts,
                "count":                total_count,
                "count_entering":       0,
                "count_exiting":        0,
                "count_by_upper_color": color_counts or None,
            })
            if session_id:
                await SessionRepository.increment_counters(
                    db, session_id,
                    frames=1,
                    detections=len(detections_data),
                )
            await db.commit()

        log.debug(
            "detector.frame_processed",
            frame=frame_number,
            count=total_count,
            inference_ms=round(det_result.inference_time_ms, 1),
        )

        # ═══════════════════════════════════════════════════════════
        # 2. FRAME SNAPSHOT — simpan gambar frame ke MinIO
        #    Hanya upload jika ada orang terdeteksi (hemat storage)
        #    Bucket: frames/
        #    Key   : frames/{camera_id}/{session_id}/frame_{:06d}.jpg
        # ═══════════════════════════════════════════════════════════
        if save_snapshot and total_count > 0:
            try:
                # Encode frame ke JPEG (kualitas 85)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                _, snap_buf   = cv2.imencode(".jpg", bgr, encode_params)
                snap_bytes    = bytes(snap_buf)

                snap_key = (
                    f"frames/{camera_id}/{session_id}"
                    f"/frame_{frame_number:06d}.jpg"
                )

                await storage.upload(
                    bucket="frames",
                    key=snap_key,
                    data=snap_bytes,
                    content_type="image/jpeg",
                    metadata={
                        "camera_id":    camera_id,
                        "session_id":   session_id or "",
                        "frame_number": str(frame_number),
                        "person_count": str(total_count),
                        "timestamp":    count_ts.isoformat(),
                        "zone_counts":  str(zone_counts),
                    },
                )

                log.info(
                    "frame.snapshot_saved",
                    frame=frame_number,
                    count=total_count,
                    key=snap_key,
                    size_kb=round(len(snap_bytes) / 1024, 1),
                )

            except Exception as snap_err:
                # Jangan crash task hanya karena snapshot gagal
                log.warning(
                    "frame.snapshot_failed",
                    frame=frame_number,
                    error=str(snap_err),
                )

        # 3. Face Detection & Recognition
        if extract_faces:
            fe = FaceEngine.get_instance()
            async with Session() as db:
                for _, crop in detections_data:
                    if crop is None:
                        continue

                    faces = fe.detect_faces(crop)
                    if not faces:
                        continue

                    best = faces[0]

                    # Filter kualitas dan ukuran minimum wajah
                    min_quality = frame_data.get("face_quality_threshold", 0.0)
                    if best.quality_score < min_quality:
                        continue

                    min_size = frame_data.get("face_min_size", 20)
                    h_face, w_face = best.crop.shape[:2] if best.crop is not None else (0, 0)
                    if min(h_face, w_face) < min_size:
                        continue

                    match   = await fe.recognize(best)
                    face_id = str(_uuid.uuid4())

                    # Upload face crop ke MinIO
                    _, face_buf = cv2.imencode(".jpg", best.crop)
                    face_key    = StorageKeys.face_crop(face_id)
                    try:
                        await storage.upload(
                            "faces", face_key,
                            bytes(face_buf), "image/jpeg",
                        )
                    except Exception:
                        pass

                    # Simpan face record ke DB
                    await FaceRepository.create(db, {
                        "id":                 face_id,
                        "person_id":          match.matched_person_id,
                        "is_known":           match.is_known,
                        "best_frame_path":    face_key,
                        "detection_score":    best.detection_score,
                        "best_quality_score": best.quality_score,
                        "camera_ids":         [camera_id],
                    })

                    # Register wajah baru ke vector store jika belum dikenal
                    if not match.is_known:
                        await fe.register_face(best, face_id=face_id, person_id=None)

                    # Fire webhook face event
                    event = "face.recognized" if match.is_known else "face.unknown"
                    whs   = await WebhookRepository.get_for_event(db, event, camera_id)
                    for wh in whs:
                        send_webhook.delay(
                            wh.url, event,
                            {
                                "event":            event,
                                "camera_id":        camera_id,
                                "timestamp":        count_ts.isoformat(),
                                "person_id":        match.matched_person_id,
                                "face_id":          face_id,
                                "similarity_score": match.similarity_score,
                            },
                            wh.secret,
                        )

                await db.commit()

        # 4. Webhook person.detected
        if total_count > 0:
            async with Session() as db:
                whs = await WebhookRepository.get_for_event(db, "person.detected", camera_id)
                for wh in whs:
                    send_webhook.delay(
                        wh.url, "person.detected",
                        {
                            "event":        "person.detected",
                            "camera_id":    camera_id,
                            "timestamp":    count_ts.isoformat(),
                            "frame_number": frame_number,
                            "total_count":  total_count,
                            "zone_counts":  zone_counts,
                        },
                        wh.secret,
                    )

        await engine.dispose()

    asyncio.run(_save())

    return {
        "camera_id":    camera_id,
        "frame_number": frame_number,
        "count":        total_count,
        "zone_counts":  zone_counts,
        "inference_ms": round(det_result.inference_time_ms, 1),
    }


# ════════════════════════════════════════════════════════════════════
# TASK 2: process_video_file
# Dipanggil saat user upload video recording
# ════════════════════════════════════════════════════════════════════

@celery_app.task(name="workers.detection_tasks.process_video_file", bind=True, max_retries=2)
def process_video_file(self, recording_id: str, minio_key: str, config: dict) -> dict[str, Any]:
    """Full pipeline untuk video yang di-upload: download -> detect -> analyze -> compress -> save DB."""
    import asyncio
    import os
    import tempfile

    async def _run():
        import uuid as _uuid
        import av
        import cv2
        import numpy as np
        from datetime import datetime, timezone

        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from core.config.settings import settings
        from core.abstractions.storage import get_storage, StorageKeys
        from core.models.repository import (
            DetectionRepository,
            PeopleCountRepository,
            RecordingRepository,
            SessionRepository,
        )
        from services.detector.yolo_detector import YOLODetector
        from services.attribute_analyzer.color_detector import AttributeAnalyzer
        from services.compressor.ffmpeg_wrapper import FFmpegWrapper, CompressionConfig

        engine  = create_async_engine(settings.DATABASE_URL, echo=False)
        Session = async_sessionmaker(engine, expire_on_commit=False)
        storage = get_storage()
        log.info("video_task.start", recording_id=recording_id)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Download video dari MinIO
            content = await storage.download("videos", minio_key)
            with open(tmp_path, "wb") as f:
                f.write(content)

            async with Session() as db:
                await RecordingRepository.update_status(db, recording_id, "processing")
                session = await SessionRepository.create(db, config.get("camera_id"))
                await db.commit()

            detector      = YOLODetector.get_instance()
            attr_analyzer = AttributeAnalyzer()
            container     = av.open(tmp_path)
            video_stream  = container.streams.video[0]

            sample_rate   = config.get("sample_rate", 5)
            zones         = config.get("zones")
            save_snapshot = config.get("save_frame_snapshot", False)  # default off untuk video

            total_frames  = 0
            total_persons = 0
            frame_idx     = 0

            for packet in container.demux(video_stream):
                for av_frame in packet.decode():
                    frame_idx += 1
                    if frame_idx % sample_rate != 0:
                        continue

                    bgr = av_frame.to_ndarray(format="bgr24")
                    ts  = (
                        float(av_frame.pts * av_frame.time_base)
                        if av_frame.pts else frame_idx / 25.0
                    )

                    det           = detector.detect(bgr, frame_number=frame_idx, timestamp=ts, zones=zones)
                    total_frames  += 1
                    total_persons += det.person_count

                    db_dets = []
                    for p in det.detections:
                        attrs = None
                        if p.crop is not None:
                            attrs, _ = attr_analyzer.analyze_with_counting_filter(
                                p.crop, config.get("filter_criteria")
                            )
                        db_dets.append({
                            "session_id":   session.id,
                            "camera_id":    config.get("camera_id"),
                            "frame_number": frame_idx,
                            "timestamp":    datetime.fromtimestamp(ts, tz=timezone.utc),
                            "track_id":     p.track_id,
                            "confidence":   p.confidence,
                            "bbox_x":       p.bbox.x, "bbox_y": p.bbox.y,
                            "bbox_w":       p.bbox.w, "bbox_h": p.bbox.h,
                            "zone_id":      p.zone_id,
                            "upper_color":  attrs.upper_color if attrs else None,
                            "lower_color":  attrs.lower_color if attrs else None,
                        })

                    async with Session() as db:
                        await DetectionRepository.bulk_insert(db, db_dets)
                        await db.commit()

                    # Frame snapshot untuk video (opsional)
                    if save_snapshot and det.person_count > 0:
                        try:
                            _, snap_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            snap_key = (
                                f"frames/{config.get('camera_id', 'upload')}"
                                f"/{recording_id}/frame_{frame_idx:06d}.jpg"
                            )
                            await storage.upload(
                                "frames", snap_key, bytes(snap_buf), "image/jpeg",
                                metadata={"person_count": str(det.person_count)},
                            )
                        except Exception:
                            pass

            container.close()

            # Kompresi video
            wrapper        = FFmpegWrapper()
            info           = await wrapper.get_video_info(tmp_path)
            compressed_tmp = tmp_path.replace(".mp4", "_compressed.mp4")
            extra = {
                "duration_seconds": info.get("duration"),
                "width":  info.get("width"),
                "height": info.get("height"),
                "fps":    info.get("fps"),
            }

            if config.get("compress", True):
                result = await wrapper.compress(
                    CompressionConfig(input_path=tmp_path, output_path=compressed_tmp)
                )
                if result.success:
                    with open(compressed_tmp, "rb") as f:
                        await storage.upload(
                            "compressed",
                            StorageKeys.video_compressed(recording_id),
                            f, "video/mp4",
                        )
                    extra.update({
                        "compressed_path":       StorageKeys.video_compressed(recording_id),
                        "compressed_size_bytes": result.output_size_bytes,
                        "compression_ratio":     result.compression_ratio,
                        "codec_compressed":      result.codec_used,
                    })
                    if os.path.exists(compressed_tmp):
                        os.unlink(compressed_tmp)

            async with Session() as db:
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
            e2  = create_async_engine(s2.DATABASE_URL, echo=False)
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


# ════════════════════════════════════════════════════════════════════
# TASK 3: run_face_clustering
# ════════════════════════════════════════════════════════════════════

@celery_app.task(name="workers.face_tasks.run_face_clustering", bind=True)
def run_face_clustering(self) -> dict[str, Any]:
    """Periodic DBSCAN clustering of unknown faces. Runs every 30 min via beat_schedule."""
    import asyncio
    import numpy as np

    async def _cluster():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from core.config.settings import settings
        from core.models.repository import FaceRepository
        from services.face_engine.recognizer import FaceClusterer

        engine  = create_async_engine(settings.DATABASE_URL, echo=False)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        async with Session() as db:
            faces = await FaceRepository.get_unknown_with_embeddings(db, limit=5000)

        if len(faces) < 2:
            await engine.dispose()
            return {"clustered": 0, "reason": "not enough faces"}

        face_ids   = [f.id for f in faces]
        embeddings = [np.array(f.embedding) for f in faces]
        cluster_map = FaceClusterer().cluster(face_ids, embeddings)

        by_cluster: dict[str, list[str]] = {}
        for fid, label in cluster_map.items():
            by_cluster.setdefault(label, []).append(fid)

        async with Session() as db:
            for label, fids in by_cluster.items():
                await FaceRepository.update_cluster(db, fids, label)
            await db.commit()

        n_clusters = sum(1 for k in by_cluster if k != "noise")
        log.info("face_clustering.complete", faces=len(face_ids), clusters=n_clusters)
        await engine.dispose()
        return {"clustered": len(face_ids), "clusters": n_clusters}

    return asyncio.run(_cluster())


# ════════════════════════════════════════════════════════════════════
# TASK 4: aggregate_counts
# ════════════════════════════════════════════════════════════════════

@celery_app.task(name="workers.analytics_tasks.aggregate_counts")
def aggregate_counts() -> dict[str, Any]:
    """Periodic aggregation of people_counts. Runs every 5 min via beat_schedule."""
    return {"status": "ok"}


# ════════════════════════════════════════════════════════════════════
# TASK 5: send_webhook
# ════════════════════════════════════════════════════════════════════

@celery_app.task(name="workers.analytics_tasks.send_webhook", bind=True, max_retries=3)
def send_webhook(
    self,
    webhook_url: str,
    event: str,
    payload: dict,
    secret: Optional[str] = None,
) -> bool:
    """Kirim webhook dengan HMAC-SHA256 signature dan exponential retry."""
    import hashlib
    import hmac as _hmac
    import json
    import requests
    from core.config.settings import settings

    body    = json.dumps(payload, default=str)
    headers = {"Content-Type": "application/json", "X-Vision-Event": event}
    if secret:
        sig = _hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers["X-Vision-Signature"] = f"sha256={sig}"

    try:
        resp = requests.post(webhook_url, data=body, headers=headers, timeout=settings.WEBHOOK_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        raise self.retry(
            exc=exc,
            countdown=settings.WEBHOOK_RETRY_DELAY * (2 ** self.request.retries),
        )


# ════════════════════════════════════════════════════════════════════
# TASK 6: run_face_clustering_trigger
# ════════════════════════════════════════════════════════════════════

@celery_app.task(name="workers.analytics_tasks.run_face_clustering_trigger")
def run_face_clustering_trigger() -> dict:
    """
    Dijadwalkan beat di CPU worker.
    Forward task clustering ke GPU queue agar tidak ada import GPU di CPU worker.
    """
    task = celery_app.send_task(
        "workers.face_tasks.run_face_clustering",
        queue="gpu_tasks",
    )
    return {"forwarded_task_id": task.id}