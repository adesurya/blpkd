Fitur Vision Platform
1. Video Ingestion & Stream Management

RTSP stream — IP camera langsung via protokol RTSP
HLS/m3u8 stream — live stream atau VoD
Video file upload — MP4, AVI, MOV, MKV, WEBM
Frame sampling — configurable (1–30 fps), hemat resource
Auto-reconnect — stream putus otomatis reconnect dengan exponential backoff
Multi-camera — kelola banyak kamera sekaligus

2. People Detection & Tracking

YOLOv8 object detection — deteksi orang di setiap frame
ByteTrack/BotSORT multi-object tracking — setiap orang dapat ID unik yang persisten
Confidence threshold — configurable via API (0.1–1.0)
Zone-based counting — definisikan area polygon, hitung per zona
Count threshold alert — kirim notifikasi jika jumlah orang melebihi batas

3. Face Detection & Recognition

InsightFace buffalo_l — deteksi dan recognition wajah
Face enrollment — daftarkan orang dikenal dengan foto
Face search — cari wajah dari foto via API
Similarity threshold — configurable (lenient hingga strict)
Face quality scoring — pilih capture terbaik berdasarkan sharpness + size
Face size filter — abaikan wajah terlalu kecil (min pixel configurable)

4. Face Clustering (Unknown Faces)

DBSCAN clustering — kelompokkan wajah tidak dikenal yang mirip
Auto-clustering — berjalan otomatis setiap 30 menit
Manual trigger — bisa dipanggil kapan saja via API
Cluster assignment — assign cluster ke identitas dikenal setelah review

5. Attribute Analysis

Warna baju atas — deteksi dominan warna kemeja/jaket (12 warna)
Warna baju bawah — deteksi warna celana/rok
Aktivitas — standing, walking, running, sitting (butuh custom model)
Filter counting — hitung hanya orang dengan atribut tertentu

6. Video Compression

FFmpeg NVENC — kompresi GPU H.264/H.265
CPU fallback — libx264 jika tidak ada GPU
CRF quality control — configurable quality vs size tradeoff
Compression ratio tracking — catat berapa persen penghematan per video
Presigned download URL — download original atau compressed via link sementara

7. Storage

MinIO S3-compatible — simpan video, frame, face crop, compressed video
4 bucket terpisah — videos, frames, faces, compressed
Face crop storage — simpan gambar wajah terbaik per identitas
Optional person crop — simpan full-body crop (configurable)
Anonymization mode — blur wajah unknown sebelum disimpan

8. Analytics & Reporting

Live people count — jumlah orang real-time per kamera per zona
Time-series data — grafik count per interval waktu (1/5/15/30/60 menit)
Heatmap — visualisasi posisi orang terbanyak di frame
Color distribution — statistik warna baju yang paling sering muncul
Recognition rate — persentase wajah yang berhasil dikenali
Analytics summary — peak count, average, zone breakdown per periode

9. Webhook & Notifikasi

Event-driven — notifikasi real-time ke endpoint Anda
6 jenis event — person.detected, face.recognized, face.unknown, count.threshold, stream.started, video.processing_complete
HMAC signature — verifikasi keamanan payload
Retry otomatis — exponential backoff jika endpoint down
Per-camera filter — webhook hanya untuk kamera tertentu

10. API & Konfigurasi

REST API lengkap — semua fitur bisa dikontrol via curl/HTTP
JWT authentication — role-based (admin, operator, viewer)
DetectionConfig — 15+ parameter AI bisa diubah per-stream tanpa restart
Live config update — PATCH /streams/{id}/config ubah behavior tanpa stop stream
Swagger UI — dokumentasi interaktif di /api/v1/docs

11. Infrastructure

Docker Compose — deploy satu perintah
Celery workers — pemrosesan async, GPU dan CPU worker terpisah
Redis queue — message queue antar service
PostgreSQL + pgvector — database + vector similarity search
Prometheus + Grafana — monitoring CPU, memory, request rate
Structured logging — JSON log untuk production



to-do
=====
Yang Belum Ada (Roadmap)
FiturKompleksitasKeteranganLicense plate recognitionMediumButuh custom YOLO modelAge/gender estimationLowSudah ada di InsightFace, tinggal aktifkanEmotion detectionMediumInsightFace + custom classifierFight/violence detectionHighButuh video classification modelCrowd density estimationMediumCustom model atau counting dari heatmapRe-ID lintas kameraHighTorchreid, butuh koordinasi multi-cameraObject detection non-orangLowYOLOv8 sudah support, tinggal tambah classDashboard UIMediumFrontend React/Next.jsHelm chart (Kubernetes)MediumUntuk scale enterprise