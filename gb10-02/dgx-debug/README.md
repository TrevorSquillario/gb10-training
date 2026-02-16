# GB10 Docker Images for Dell Pro Max with GB10

This repository contains two Docker images designed for Dell Pro Max with GB10 systems running NVIDIA DGX OS. Both images assume CUDA-enabled drivers on the host and --gpus all when running containers.

Overview
--------

- `gb10-dev` — Full AI development + troubleshooting environment
- `gb10-diag` — Lightweight diagnostics and support image (no heavy ML frameworks)

Use `gb10-dev` when you need to develop or reproduce model-level issues (PyTorch / TensorFlow).
Use `gb10-diag` when you need to quickly validate system health (GPU visibility, NCCL, DCGM).

## Image 1 — `gb10-dev` (Development + Troubleshooting)

### Purpose

- Primary image for AI development, repro, and performance debugging on GB10.
- Good for reproducing customer workloads end-to-end (framework → CUDA → GPU).

### Key contents

#### Base

- `nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04`

#### Tools

- Standard Linux utilities: `curl`, `git`, `iproute2`, `ping`, `net-tools`, `dnsutils`, `htop`, `nvtop`, etc.
- Full CUDA toolkit and CLI tools (via base image) so `nvidia-smi` and `nvcc` are available.

#### Python + AI frameworks

- Python 3, `pip`, build tooling
- PyTorch (CUDA 12.1 wheels): `torch`, `torchvision`, `torchaudio`
- TensorFlow (GPU build)

#### NCCL

- `nccl-tests` built under `/opt/nccl-tests` for all-reduce and communications tests

### Typical use cases

- Reproduce customer issues in PyTorch or TensorFlow
- Validate framework compatibility with GB10 CUDA drivers
- Run performance comparisons, micro-benchmarks, and NCCL tests
- Experimental development and quick prototyping on GB10

### Example commands

Build:

```sh
cd dgx-debug
docker build --progress=plain -f Dockerfile.dev -t gb10-dev:latest .
```

Smoke test:

```sh
docker run --rm --gpus all gb10-dev:latest nvidia-smi
```

Interactive session:

```sh
docker run -it --rm --gpus all \
  --net=host --ipc=host \
  -v /var/log:/var/log:ro \
  gb10-dev:latest
```

Inside container:

```sh
# NCCL sanity check
cd $NCCL_TESTS_DIR
./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# Print all CUDA related info
python cudainfo.py

# PyTorch GPU check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# TensorFlow GPU check
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Image 2 — `gb10-diag` (Diagnostics-only, no ML frameworks)

### Purpose

- Lean, support-oriented image for health checks and troubleshooting.
- Minimizes size and complexity by omitting big AI frameworks (PyTorch, TensorFlow).

### Key contents

#### Base

- `nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04`

#### Tools

- Standard Linux utilities: `curl`, `iproute2`, `ping`, `net-tools`, `dnsutils`, `htop`, `nvtop`, etc.
- CUDA user-space tools so `nvidia-smi` and related commands are available.

#### Python

- Python 3 with a light `pip` stack

---

## Usage notes

Build:

```sh
cd dgx-debug
docker build -f Dockerfile.diag -t gb10-diag:latest .
```

Quick check:

```sh
docker run --rm --gpus all gb10-diag:latest nvidia-smi
```

Interactive diag session:

```sh
docker run -it --rm --gpus all \
  --net=host --ipc=host \
  -v /var/log:/var/log:ro \
  gb10-diag:latest
```

Inside container:

```sh
# NCCL test
cd $NCCL_TESTS_DIR
./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# DCGM basic diagnostics
dcgmi discovery -l
dcgmi diag -r 1
```