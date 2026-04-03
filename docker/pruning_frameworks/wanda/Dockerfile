ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/framework
COPY docker/pruning_frameworks/run_prune.py /workspace/framework/run_prune.py

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.2.2 transformers==4.31.0 accelerate==0.27.2

ENTRYPOINT ["python3", "/workspace/framework/run_prune.py"]
