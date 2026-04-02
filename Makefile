# Makefile — Vision Platform Development Commands

.PHONY: help up up-gpu down logs ps init-db test lint clean

help:
	@echo "Vision Platform — Available Commands"
	@echo "======================================"
	@echo "  make up         Start all services (CPU mode)"
	@echo "  make up-gpu     Start all services with NVIDIA GPU"
	@echo "  make down       Stop all services"
	@echo "  make logs       Tail all service logs"
	@echo "  make ps         Show service status"
	@echo "  make init-db    Initialize database & storage (run once)"
	@echo "  make test       Run all tests"
	@echo "  make test-unit  Run unit tests only (no Docker needed)"
	@echo "  make lint       Run ruff + mypy"
	@echo "  make shell-api  Open shell in API container"
	@echo "  make shell-gpu  Open shell in GPU worker container"
	@echo "  make models     Pre-download AI models"
	@echo "  make clean      Remove all containers and volumes (DESTRUCTIVE)"

up:
	docker compose up -d
	@echo "✅ Platform started. API: http://localhost:8000/api/v1/docs"

up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
	@echo "✅ Platform started with GPU support."

down:
	docker compose down

logs:
	docker compose logs -f --tail=50

ps:
	docker compose ps

init-db:
	@echo "Waiting for postgres..."
	@sleep 5
	docker compose exec api python scripts/setup/init_db.py
	@echo "✅ Database initialized"

test:
	docker compose exec api pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

lint:
	ruff check . --fix
	mypy services/ core/ workers/ --ignore-missing-imports

models:
	docker compose exec worker-gpu python3 -c \
		"from ultralytics import YOLO; YOLO('yolov8n.pt'); print('YOLOv8n ready')"
	docker compose exec worker-gpu python3 -c \
		"from insightface.app import FaceAnalysis; \
		 app = FaceAnalysis(name='buffalo_l', root='/app/models'); \
		 app.prepare(ctx_id=-1); print('InsightFace ready')"
	@echo "✅ Models downloaded"

shell-api:
	docker compose exec api bash

shell-gpu:
	docker compose exec worker-gpu bash

token:
	@curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin&password=changeme123" | python3 -m json.tool

clean:
	@echo "⚠️  This will delete ALL containers and volumes. Press Ctrl+C to cancel."
	@sleep 3
	docker compose down -v --remove-orphans
	@echo "✅ Cleaned up"
