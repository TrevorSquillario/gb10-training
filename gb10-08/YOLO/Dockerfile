FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg libgl1 libsm6 libxext6 libxrender1 \
    build-essential git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create venv (keeps python deps isolated inside container)
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Upgrade pip and packaging tools
RUN python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Install CUDA-enabled PyTorch for CUDA 13.0 (cu130). Adjust if you need
# a different CUDA/PyTorch combination.
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio

# Then install the remaining Python packages (ultralytics expects torch)
RUN python3 -m pip install --no-cache-dir ultralytics flask opencv-python-headless

# Example (commented) - uncomment and adjust to the correct torch wheel for CUDA13
# RUN python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly

WORKDIR /app
COPY track_rtsp.py /app/

EXPOSE 5000

CMD ["python3", "track_rtsp.py"]
