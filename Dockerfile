# 金融问答批量生成镜像（CPU；一键运行 main.py 输出 qa_results.json）
FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p /app/data /app/output /app/test /app/vector_cache && \
    chmod +x /app/scripts/*.sh

ENV MMD_PATH=/app/output/knowledge_base.json
ENV CACHE_DIR=/app/vector_cache

CMD ["python", "main.py"]
