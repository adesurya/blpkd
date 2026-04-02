# CCTV & Video Source Configuration Guide

## Supported Source Types

| Type | Protocol | Example URL |
|------|----------|-------------|
| IP Camera (RTSP) | RTSP over TCP | `rtsp://user:pass@192.168.1.100:554/stream1` |
| HLS Live Stream | HTTP/HTTPS m3u8 | `https://cdn.example.com/live/cam01.m3u8` |
| HLS VoD | HTTP/HTTPS m3u8 | `https://cdn.example.com/vod/recording.m3u8` |
| Uploaded File | Local path | Upload via `POST /videos/upload` |
| Webcam (dev) | Index | `0` (first webcam) |

---

## 1. RTSP IP Camera Setup

### Common Camera RTSP URL Formats

```bash
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101   # Main stream
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/102   # Sub stream (lower res)

# Dahua
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://root:password@192.168.1.100/axis-media/media.amp

# Generic ONVIF
rtsp://user:pass@192.168.1.100:554/stream1
rtsp://user:pass@192.168.1.100:554/h264Preview_01_main

# Reolink
rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
rtsp://admin:password@192.168.1.100:554/h264Preview_01_sub

# Uniview
rtsp://admin:password@192.168.1.100:554/media/video1
```

### Verify Stream Before Registering

```bash
# Test with FFprobe
ffprobe -v quiet -print_format json -show_streams \
  "rtsp://admin:password@192.168.1.100:554/stream1"

# Test with FFplay (visual check)
ffplay -rtsp_transport tcp "rtsp://admin:password@192.168.1.100:554/stream1"

# Test with VLC
vlc "rtsp://admin:password@192.168.1.100:554/stream1"
```

### Camera Registration via API

```bash
curl -X POST http://localhost:8000/api/v1/cameras \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Entrance Camera 01",
    "location": "Main Gate",
    "source_type": "rtsp",
    "source_url": "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101",
    "frame_sample_rate": 5,
    "zones": {
      "entry_area": [[0.0,0.2],[0.6,0.2],[0.6,0.9],[0.0,0.9]]
    }
  }'
```

---

## 2. m3u8 HLS Stream Setup

### Live Stream (e.g., from Frigate NVR or media server)

```bash
# If you have Frigate NVR running, access HLS streams:
# http://frigate-host:5000/vod/camera_name/index.m3u8

curl -X POST http://localhost:8000/api/v1/cameras \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Mall CCTV Live",
    "source_type": "m3u8",
    "source_url": "http://192.168.1.200:8888/stream/cam01.m3u8",
    "frame_sample_rate": 10
  }'
```

### Generate m3u8 from RTSP using FFmpeg (if camera only provides RTSP)

```bash
# Convert RTSP → HLS and serve locally
ffmpeg -i "rtsp://admin:pass@192.168.1.100:554/stream1" \
  -c:v copy \
  -c:a copy \
  -f hls \
  -hls_time 2 \
  -hls_list_size 5 \
  -hls_flags delete_segments \
  /var/www/html/streams/cam01.m3u8
```

---

## 3. Zone Configuration (People Counting Areas)

Zones are polygon coordinates normalized to 0.0–1.0 of frame dimensions.

```
Frame (width x height):
(0,0)─────────────────(1,0)
  │                     │
  │    Zone A           │
  │  [0.0,0.2]─[0.5,0.2]│
  │      │         │    │
  │  [0.0,0.8]─[0.5,0.8]│
  │                     │
(0,1)─────────────────(1,1)
```

### Example: Entrance/Exit split

```json
{
  "entrance": [[0.0,0.0],[0.5,0.0],[0.5,1.0],[0.0,1.0]],
  "exit":     [[0.5,0.0],[1.0,0.0],[1.0,1.0],[0.5,1.0]]
}
```

### Example: Retail store counting line

```json
{
  "counting_line": [[0.0,0.5],[1.0,0.5],[1.0,0.6],[0.0,0.6]]
}
```

### Example: Multiple zones with names

```json
{
  "cashier_area": [[0.0,0.6],[0.3,0.6],[0.3,1.0],[0.0,1.0]],
  "browsing_area": [[0.0,0.0],[1.0,0.0],[1.0,0.6],[0.0,0.6]],
  "exit_door": [[0.85,0.0],[1.0,0.0],[1.0,0.4],[0.85,0.4]]
}
```

---

## 4. Frame Sample Rate Tuning

| Use Case | Recommended Rate | Notes |
|----------|-----------------|-------|
| Entrance counting (slow traffic) | 3–5 | Process 1 in 5 frames |
| Busy area counting | 2–3 | More frequent sampling |
| Face recognition focus | 1–2 | Need high frame rate for good captures |
| Archive video analysis | 5–10 | Can process faster than realtime |
| Surveillance overview | 10–15 | Low compute, lower accuracy |

Formula: `actual_fps_processed = camera_fps / frame_sample_rate`
At 25fps camera, sample_rate=5 → processing 5fps

---

## 5. Recommended Camera Settings for Best AI Results

Configure your IP camera's OSD/settings:

```
Resolution:   1080p (1920×1080) minimum — 2MP preferred
Frame Rate:   15–25 fps
Bitrate:      2–4 Mbps (VBR recommended)
Codec:        H.264 or H.265
Day/Night:    Auto (with IR for low-light)
WDR:          Enable (helps with backlit entrances)
Sharpness:    Medium (over-sharpening adds noise)
Compression:  Medium (avoid too much compression)
```

### Camera Placement for Face Recognition

- Mount angle: 10–25° downward tilt (not too steep)
- Face height in frame: minimum 80–100px height
- Distance: 2–5 meters from face detection zone
- Lighting: avoid direct backlight (window behind subjects)
- Avoid IR overexposure at night (causes washed-out faces)

---

## 6. Multi-Camera Configuration Example

```bash
# Register multiple cameras
for cam in 1 2 3 4; do
  curl -X POST http://localhost:8000/api/v1/cameras \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"name\": \"Camera $cam\",
      \"source_type\": \"rtsp\",
      \"source_url\": \"rtsp://admin:pass@192.168.1.10${cam}:554/stream1\",
      \"frame_sample_rate\": 5
    }"
done

# Start all streams
CAMERAS=$(curl -s http://localhost:8000/api/v1/cameras -H "Authorization: Bearer $TOKEN")
echo $CAMERAS | jq -r '.[].id' | while read cam_id; do
  curl -X POST "http://localhost:8000/api/v1/streams/$cam_id/start" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"camera_id":"'"$cam_id"'","extract_faces":true,"analyze_attributes":true}'
  echo "Started stream: $cam_id"
done
```

---

## 7. Testing with Sample Video

```bash
# Download a test video
wget https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4 -O test.mp4

# Upload for processing
curl -X POST http://localhost:8000/api/v1/videos/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.mp4" \
  -F "extract_faces=true" \
  -F "analyze_attributes=true" \
  -F "compress=true" \
  -F "sample_rate=5"
```
