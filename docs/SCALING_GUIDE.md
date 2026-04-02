# Scaling Guide: MVP → Growth → Enterprise

## Design Philosophy

The platform uses **abstraction layers** at every integration point.
This means scaling is done by:
1. Changing env vars
2. Spinning up more containers
3. **Zero code changes to business logic**

---

## Phase 1 → Phase 2: MVP to Growth

### Trigger Points (when to scale up)

| Signal | Action |
|--------|--------|
| GPU worker queue depth > 100 consistently | Add more `worker-gpu` instances |
| Postgres query time > 500ms | Add read replicas, tune indexes |
| MinIO throughput saturated | Expand to 3-node cluster |
| Redis memory > 80% | Increase maxmemory or switch to RabbitMQ |
| >20 cameras active simultaneously | Add dedicated stream servers |

### Scale GPU Workers (horizontal, no config change)

```bash
# Scale to 3 GPU workers (requires 3 GPUs)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --scale worker-gpu=3

# Or with specific GPU assignments per worker:
# Add to docker-compose.gpu.yml:
# environment:
#   - CUDA_VISIBLE_DEVICES=0   # worker 1 uses GPU 0
#   - CUDA_VISIBLE_DEVICES=1   # worker 2 uses GPU 1
```

### Switch Queue: Redis → RabbitMQ

```bash
# 1. Add to .env
QUEUE_BACKEND=rabbitmq
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

# 2. Add RabbitMQ to docker-compose
# (create docker-compose.rabbitmq.yml)

# 3. Restart — zero code change
docker compose restart
```

`docker-compose.rabbitmq.yml`:
```yaml
services:
  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: visionuser
      RABBITMQ_DEFAULT_PASS: visionpass

  worker-gpu:
    environment:
      - CELERY_BROKER_URL=amqp://visionuser:visionpass@rabbitmq:5672/

volumes:
  rabbitmq_data:
```

### MinIO: Single → 3-Node Cluster

```yaml
# docker-compose.minio-cluster.yml
services:
  minio1:
    image: minio/minio
    command: server http://minio{1...3}/data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    volumes:
      - minio1_data:/data
    hostname: minio1

  minio2:
    image: minio/minio
    command: server http://minio{1...3}/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    volumes:
      - minio2_data:/data
    hostname: minio2

  minio3:
    image: minio/minio
    command: server http://minio{1...3}/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    volumes:
      - minio3_data:/data
    hostname: minio3
```

---

## Phase 2 → Phase 3: Growth to Enterprise

### Switch to Kubernetes

```bash
# Generate Helm values from current compose config
kompose convert -f docker-compose.yml -o k8s/

# Or use the included Helm chart (create charts/ directory)
helm install vision-platform ./charts/vision-platform \
  --set gpu.enabled=true \
  --set replicas.apiServer=3 \
  --set replicas.gpuWorker=5 \
  --set storage.class=fast-ssd
```

### Switch Queue: RabbitMQ → Apache Kafka

```bash
# .env change
QUEUE_BACKEND=kafka
KAFKA_BOOTSTRAP_SERVERS=kafka-broker-1:9092,kafka-broker-2:9092,kafka-broker-3:9092

# Zero code change — KafkaBackend in queue.py handles it
docker compose restart
```

### Switch Vector DB: pgvector → Milvus

When face database exceeds 1 million embeddings:

```bash
# .env change
VECTOR_BACKEND=milvus
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530

# Deploy Milvus (docker or k8s)
docker compose -f docker-compose.milvus.yml up -d

# Migrate existing embeddings
docker compose exec api python scripts/setup/migrate_pgvector_to_milvus.py
```

`docker-compose.milvus.yml`:
```yaml
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
    volumes:
      - etcd_data:/etcd

  minio-milvus:
    image: minio/minio:latest
    command: minio server /minio_data --console-address ":9001"

  milvus-standalone:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio-milvus:9000
    depends_on:
      - etcd
      - minio-milvus
    ports:
      - "19530:19530"
```

### NVIDIA Triton Inference Server (High-Throughput GPU)

Replace direct model calls with Triton for:
- Dynamic batching (process multiple frames in one GPU call)
- Multi-model serving from single GPU
- Model versioning and A/B testing

```bash
# Export YOLOv8 to Triton format
docker compose exec worker-gpu python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0)  # TensorRT engine
"

# Deploy Triton
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  -v ./models/triton:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

Then update `YOLO_MODEL` in settings to point to Triton endpoint.

---

## Cost Estimation by Phase

### MVP (1–5 cameras, on-premise)

| Item | Cost |
|------|------|
| Server (16-core, 32GB RAM) | Rp 15–25 juta (one-time) |
| NVIDIA RTX 3060 12GB | Rp 4–7 juta (one-time) |
| 4TB SSD | Rp 2–4 juta (one-time) |
| Software licenses | Rp 0 |
| **Total MVP** | **Rp 21–36 juta** |

### Growth (10–50 cameras, on-premise)

| Item | Cost |
|------|------|
| 3x servers (GPU each) | Rp 60–120 juta |
| Network switch | Rp 3–10 juta |
| NAS storage (20TB) | Rp 15–30 juta |
| UPS | Rp 5–10 juta |
| **Total Growth** | **Rp 83–170 juta** |

### Cloud Alternative (AWS)

| Phase | Instance | Monthly Cost |
|-------|---------|-------------|
| MVP | g4dn.xlarge (T4 GPU) | ~USD 400/mo |
| Growth | g4dn.4xlarge × 3 | ~USD 3,500/mo |
| Enterprise | p3.8xlarge cluster | ~USD 15,000+/mo |

---

## Performance Benchmarks (Reference)

Measured on NVIDIA RTX 3060 12GB:

| Operation | Throughput | Latency |
|-----------|-----------|---------|
| YOLOv8n detection | ~80 FPS @ 1080p | ~12ms |
| YOLOv8s detection | ~55 FPS @ 1080p | ~18ms |
| InsightFace detection | ~40 FPS @ 1080p | ~25ms |
| Face recognition (pgvector, 10K faces) | ~200 searches/s | ~5ms |
| FFmpeg H.264 NVENC | ~4x realtime speed | — |
| FFmpeg H.265 NVENC | ~3x realtime speed | — |

Frame sampling of 1-in-5 effectively handles:
- YOLOv8n: up to **16 simultaneous 1080p streams**
- YOLOv8s: up to **11 simultaneous 1080p streams**
