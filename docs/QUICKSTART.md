# Vision Platform — Quick Start & Implementation Guide

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| RAM | 16 GB | 32 GB |
| CPU | 8 cores | 16 cores |
| GPU | NVIDIA GTX 1060 (CPU fallback works) | NVIDIA RTX 3060+ |
| Storage | 100 GB SSD | 500 GB+ NVMe |
| Docker | 24.0+ | latest |
| Docker Compose | 2.20+ | latest |
| CUDA | 11.8+ | 12.1+ |

---

## Step 1: Clone & Configure

```bash
git clone https://github.com/your-org/vision-platform.git
cd vision-platform

# Copy and configure environment
cp .env.example .env
nano .env

# REQUIRED changes in .env:
# SECRET_KEY=<generate 32+ char random string>
# POSTGRES_PASSWORD=<strong password>
# MINIO_SECRET_KEY=<strong password>

# Generate a strong secret key:
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Step 2: Start Services

### CPU-only (no GPU, for testing)

```bash
# Update .env for CPU mode:
# YOLO_DEVICE=cpu
# INSIGHTFACE_DEVICE=cpu
# COMPRESSION_CODEC=libx264
# GPU_DECODE=false

docker compose up -d
```

### With NVIDIA GPU (recommended)

```bash
# Install NVIDIA Container Toolkit (if not installed)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Start with GPU support
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## Step 3: Initialize Database

```bash
# Wait for postgres to be healthy
docker compose ps postgres

# Run migrations (pgvector tables, indexes, initial admin user)
docker compose exec api python scripts/setup/init_db.py

# Verify
docker compose exec postgres psql -U visionuser -d visiondb -c "\dt"
```

---

## Step 4: Download AI Models

```bash
# Create models directory
mkdir -p models

# Models auto-download on first run, but pre-downloading is faster:

# YOLOv8 nano (MVP - fastest)
docker compose exec worker-gpu python3 -c \
  "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# InsightFace buffalo_l (best accuracy)
docker compose exec worker-gpu python3 -c \
  "from insightface.app import FaceAnalysis; \
   app = FaceAnalysis(name='buffalo_l', root='/app/models'); \
   app.prepare(ctx_id=0, det_size=(640,640))"
```

---

## Step 5: Verify Everything is Running

```bash
# Check all services
docker compose ps

# Expected output:
# NAME                    STATUS
# vision-api              Up (healthy)
# vision-worker-cpu       Up
# vision-worker-gpu       Up
# vision-celery-beat      Up
# vision-flower           Up
# vision-postgres         Up (healthy)
# vision-redis            Up (healthy)
# vision-minio            Up (healthy)
# vision-nginx            Up
# vision-prometheus       Up
# vision-grafana          Up

# Test API health
curl http://localhost:8000/health
# {"status":"healthy","version":"0.1.0","env":"development"}

# Detailed health check
curl http://localhost:8000/health/detailed

# Get auth token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme123"

# Auto-generated API docs
open http://localhost:8000/api/v1/docs
```

---

## Step 6: Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | JWT token |
| API Docs | http://localhost:8000/api/v1/docs | — |
| Flower (Celery UI) | http://localhost:5555 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| Grafana | http://localhost:3001 | admin / admin123 |
| Prometheus | http://localhost:9090 | — |

---

## Step 7: First Workflow — Upload & Analyze Video

```bash
# 1. Get auth token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme123" | jq -r '.access_token')

# 2. Upload test video
RESULT=$(curl -s -X POST http://localhost:8000/api/v1/videos/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/video.mp4" \
  -F "extract_faces=true" \
  -F "analyze_attributes=true" \
  -F "compress=true")

RECORDING_ID=$(echo $RESULT | jq -r '.recording_id')
TASK_ID=$(echo $RESULT | jq -r '.task_id')
echo "Recording: $RECORDING_ID, Task: $TASK_ID"

# 3. Poll status
watch -n 2 "curl -s http://localhost:8000/api/v1/videos/$RECORDING_ID/status \
  -H 'Authorization: Bearer $TOKEN' | jq '.status, .progress'"

# 4. Get results when done
curl http://localhost:8000/api/v1/videos/$RECORDING_ID/result \
  -H "Authorization: Bearer $TOKEN" | jq
```

---

## Step 8: First Workflow — Live Stream with Face Recognition

```bash
# 1. Enroll a known person
PERSON=$(curl -s -X POST http://localhost:8000/api/v1/persons \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"John Doe","employee_id":"EMP001"}')
PERSON_ID=$(echo $PERSON | jq -r '.id')

# 2. Enroll face photos (3 photos minimum)
curl -X POST "http://localhost:8000/api/v1/persons/$PERSON_ID/enroll" \
  -H "Authorization: Bearer $TOKEN" \
  -F "photos[]=@john_front.jpg" \
  -F "photos[]=@john_side.jpg" \
  -F "photos[]=@john_angle.jpg"

# 3. Register camera
CAM=$(curl -s -X POST http://localhost:8000/api/v1/cameras \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Camera",
    "source_type": "rtsp",
    "source_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
    "frame_sample_rate": 5,
    "zones": {"main": [[0,0],[1,0],[1,1],[0,1]]}
  }')
CAM_ID=$(echo $CAM | jq -r '.id')

# 4. Start stream analysis
curl -X POST "http://localhost:8000/api/v1/streams/$CAM_ID/start" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"'"$CAM_ID"'","extract_faces":true,"analyze_attributes":true}'

# 5. Check live count
curl "http://localhost:8000/api/v1/analytics/count/live" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Monitoring & Logs

```bash
# View API logs (structured JSON)
docker compose logs -f api | jq

# View GPU worker logs
docker compose logs -f worker-gpu

# Monitor Celery tasks
open http://localhost:5555   # Flower UI

# View Grafana dashboard
open http://localhost:3001   # Grafana

# Check GPU usage
watch -n 1 nvidia-smi

# Check resource usage
docker stats
```

---

## Custom Model Training Workflow

```bash
# 1. Collect person crops from existing detections
#    (The system automatically saves person crops to MinIO/frames bucket)

# 2. Download and organize crops by label
mkdir -p /data/clothing/{train,val}/{red_shirt,blue_shirt,white_shirt,walking,running,standing}
# Move your labeled crops into the appropriate folders

# 3. Run training
docker compose exec worker-gpu python scripts/training/train_clothing_classifier.py \
  --data /data/clothing \
  --epochs 100 \
  --name clothing_v1 \
  --deploy   # auto-copy to /app/models/custom_attribute.pt

# 4. Restart GPU worker to load new model
docker compose restart worker-gpu
```

---

## Scaling to Growth Phase

```bash
# 1. Update QUEUE_BACKEND in .env
echo "QUEUE_BACKEND=rabbitmq" >> .env

# 2. Add RabbitMQ to docker-compose (create docker-compose.scale.yml)
# 3. Spin up more GPU workers
docker compose up -d --scale worker-gpu=3

# 4. Switch vector DB (when faces > 1M)
echo "VECTOR_BACKEND=milvus" >> .env
# No code changes needed — abstraction layer handles it
```
