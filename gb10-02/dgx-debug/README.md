Lab GB10-07:
Include original tutorial step 6
cd models/checkpoints/
wget https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
cd ../../

This allows the default workflow to run

---

README – GB10 Docker Images for Dell Pro Max with GB10
======================================================

This repository contains two Docker images designed for Dell Pro Max with GB10 systems running NVIDIA DGX OS. Both images assume CUDA-enabled drivers on the host and --gpus all when running containers.

Overview
--------

- gb10-dev    – Full AI development + troubleshooting environment
- gb10-diag   – Lightweight diagnostics and support image (no heavy ML frameworks)

Use gb10-dev when you need to develop or reproduce model-level issues (PyTorch / TensorFlow).
Use gb10-diag when you need to quickly validate system health (GPU visibility, NCCL, DCGM).

------------------------------------------------------------
Image 1: gb10-dev (Development + Troubleshooting)
------------------------------------------------------------

Purpose
-------

- Primary image for AI development, repro, and performance debugging on GB10.
- Good for reproducing customer workloads end-to-end (framework -> CUDA -> GPU).

Key Contents
------------

- Base:
  - nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04

- Tools:
  - Standard Linux utilities: curl, git, iproute2, ping, net-tools, dnsutils, htop, nvtop, etc.
  - Full CUDA toolkit and CLI tools (via base image) so nvidia-smi and nvcc are available.

- Python + AI frameworks:
  - Python 3, pip, build tooling
  - PyTorch (CUDA 12.1 wheels): torch, torchvision, torchaudio
  - TensorFlow (GPU build)

- NCCL:
  - nccl-tests built under /opt/nccl-tests for all-reduce and comms tests

Typical Use Cases
-----------------

- Reproduce customer issues in PyTorch or TensorFlow
- Validate framework compatibility with GB10 CUDA drivers
- Run performance comparisons, micro-benchmarks, and NCCL tests
- Experimental development and quick prototyping on GB10

Example Commands
----------------

Build:

  docker build -f Dockerfile.dev -t gb10-dev:latest .

Smoke test:

  docker run --rm --gpus all gb10-dev:latest nvidia-smi

Interactive session:

  docker run -it --rm --gpus all \
    --net=host --ipc=host \
    -v /var/log:/var/log:ro \
    gb10-dev:latest

Inside container:

  # NCCL sanity check
  cd $NCCL_TESTS_DIR
  ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

  # PyTorch GPU check
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  # TensorFlow GPU check
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

--------------------------------------------------------
Image 2: gb10-diag (Diagnostics-Only, No ML Frameworks)
--------------------------------------------------------

Purpose
-------

- Lean, support-oriented image for health checks and troubleshooting.
- Minimizes size and complexity by omitting big AI frameworks (PyTorch, TensorFlow).

Key Contents
------------

- Base:
  - nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04

- Tools:
  - Standard Linux utilities: curl, iproute2, ping, net-tools, dnsutils, htop, nvtop, etc.
  - CUDA user-space tools so nvidia-smi and related commands are available.

- Python:
  - Python 3 with a light pip stack 

---

GB10 Diag Dockerfile – Dell Pro Max with GB10 

 

# File: Dockerfile.diag 

# GB10 diagnostics-only image (no large ML frameworks) 

 

FROM nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04 

 

ENV DEBIAN_FRONTEND=noninteractive \ 

    TZ=UTC \ 

    LANG=C.UTF-8 \ 

    LC_ALL=C.UTF-8 

 

# 1. Base OS tools and utilities 

RUN apt-get update && apt-get install -y --no-install-recommends \ 

    wget \ 

    curl \ 

    ca-certificates \ 

    git \ 

    vim \ 

    nano \ 

    less \ 

    pciutils \ 

    iproute2 \ 

    iputils-ping \ 

    net-tools \ 

    dnsutils \ 

    htop \ 

    nvtop \ 

    python3 \ 

    python3-pip \ 

    python3-venv \ 

    pkg-config \ 

    build-essential \ 

    && rm -rf /var/lib/apt/lists/* 

 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 

 

# 2. Light Python stack for scripting tests 

RUN pip install --no-cache-dir --upgrade pip setuptools wheel 

 

# 3. NCCL tests 

RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \ 

    cd /opt/nccl-tests && \ 

    make MPI=0 

 

ENV NCCL_TESTS_DIR=/opt/nccl-tests 

 

# 4. DCGM (Data Center GPU Manager) tools 

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \ 

    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \ 

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \ 

    dpkg -i cuda-keyring_1.1-1_all.deb && \ 

    rm cuda-keyring_1.1-1_all.deb && \ 

    apt-get update && \ 

    apt-get install -y --no-install-recommends \ 

      datacenter-gpu-manager \ 

    && rm -rf /var/lib/apt/lists/* 

 

# Ensure CUDA tools and nvidia-smi are on PATH 

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} 

 

WORKDIR /workspace 

 

CMD ["/bin/bash"] 

 

Usage Notes 

----------- 

 

Build: 

  docker build -f Dockerfile.diag -t gb10-diag:latest . 

 

Quick check: 

  docker run --rm --gpus all gb10-diag:latest nvidia-smi 

 

Interactive diag session: 

  docker run -it --rm --gpus all \ 

    --net=host --ipc=host \ 

    -v /var/log:/var/log:ro \ 

    gb10-diag:latest 

 

Inside container: 

  # NCCL test 

  cd $NCCL_TESTS_DIR 

  ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1 

 

  # DCGM basic diagnostics 

  dcgmi discovery -l 

  dcgmi diag -r 1 



----

GB10 Dev Dockerfile – Dell Pro Max with GB10 

 

# File: Dockerfile.dev 

# GB10 AI development + troubleshooting image 

 

FROM nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04 

 

ENV DEBIAN_FRONTEND=noninteractive \ 

    TZ=UTC \ 

    LANG=C.UTF-8 \ 

    LC_ALL=C.UTF-8 

 

# 1. Base OS tools and utilities 

RUN apt-get update && apt-get install -y --no-install-recommends \ 

    wget \ 

    curl \ 

    ca-certificates \ 

    git \ 

    vim \ 

    nano \ 

    less \ 

    pciutils \ 

    iproute2 \ 

    iputils-ping \ 

    net-tools \ 

    dnsutils \ 

    htop \ 

    nvtop \ 

    python3 \ 

    python3-pip \ 

    python3-venv \ 

    pkg-config \ 

    build-essential \ 

    && rm -rf /var/lib/apt/lists/* 

 

# Symlink python3 -> python 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 

 

# 2. Python + AI frameworks 

RUN pip install --no-cache-dir --upgrade pip setuptools wheel 

 

# PyTorch (CUDA 12.1 wheels, compatible with 12.x) 

RUN pip install --no-cache-dir \ 

    torch==2.4.0+cu121 \ 

    torchvision==0.19.0+cu121 \ 

    torchaudio==2.4.0+cu121 \ 

    --index-url https://download.pytorch.org/whl/cu121 

 

# TensorFlow (GPU) 

RUN pip install --no-cache-dir \ 

    tensorflow==2.17.0 

 

# 3. NCCL tests (for basic comms tests on a single node or multi-node) 

RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \ 

    cd /opt/nccl-tests && \ 

    make MPI=0 

 

ENV NCCL_TESTS_DIR=/opt/nccl-tests 

 

# 4. Make sure CUDA tools and nvidia-smi are on PATH 

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} 

 

# 5. Working directory 

WORKDIR /workspace 

 

CMD ["/bin/bash"] 

 

Usage Notes 

----------- 

 

Build: 

  docker build -f Dockerfile.dev -t gb10-dev:latest . 

 

Smoke test: 

  docker run --rm --gpus all gb10-dev:latest nvidia-smi 

 

Interactive: 

  docker run -it --rm --gpus all \ 

    --net=host --ipc=host \ 

    -v /var/log:/var/log:ro \ 

    gb10-dev:latest 

 

Inside container: 

  # NCCL test 

  cd $NCCL_TESTS_DIR 

  ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1 

 

  # PyTorch GPU check 

  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))" 

 

  # TensorFlow GPU check 

  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 


  ---