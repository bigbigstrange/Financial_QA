FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-linux.lock ./
RUN pip install --no-cache-dir --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux.lock
COPY . .
COPY --from=frontend /frontend/dist ./frontend/dist
RUN useradd --uid 10001 --create-home finance && \
    mkdir -p output vector_cache data_runtime && chown -R finance:finance /app
USER finance
CMD ["python", "-m", "app.main"]
