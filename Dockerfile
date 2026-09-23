FROM --platform=$BUILDPLATFORM node:20-bookworm-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM docker.m.daocloud.io/library/python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libsndfile1 fonts-noto-cjk ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-final.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && pip check
COPY backend/app /app/backend/app
COPY config /app/config
COPY models/manifest.json /app/models/manifest.json
COPY --from=frontend /build/dist /app/frontend/dist
COPY docker/requirements-final.lock /app/requirements.lock
WORKDIR /app/backend
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=20s --start-period=90s --retries=3 \
    CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=15)); assert d['status']=='ok' and d['worker_spawned']"
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
