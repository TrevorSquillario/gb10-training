# Lesson 3: The Model Zoo & The Blackwell "Secret Sauce" (NVFP4)

**Objective:** Understand the math that makes the GB10 a "Supercomputer at your desk." Move beyond simply downloading models and start optimizing them for the Blackwell architecture using NVFP4.

## 1. The 128GB Unified Memory Challenge

On a standard PC, you are limited by GPU VRAM (e.g., 24GB on a 5090). On the GB10, the 128GB of Unified Memory allows us to run massive models, but we still want to maximize "Tokens Per Second" (TPS).

- **The Bandwidth Bottleneck:** The GB10 uses LPDDR5x RAM (~273 GB/s). While this is plenty of capacity, it is slower than the HBM3 used in data center H100s.
- **The Solution:** Quantization. By shrinking the model size, we reduce the amount of data moving through the memory bus, directly increasing speed.

## 2. NVFP4: Why Blackwell is Different

Most AI enthusiasts are used to INT4 or GGUF quantization. Blackwell introduces a new hardware-native format: **NVFP4 (4-bit Floating Point)**.

| Feature  | Standard 4-bit (INT4)              | Blackwell NVFP4                    |
|----------|------------------------------------|------------------------------------|
| Logic    | Rounds numbers to whole integers.  | Uses 4-bit floating point (E2M1).  |
| Accuracy | High "perplexity" (loss of smarts).| Near-FP8 accuracy (<1% loss).      |
| Hardware | Software-based dequantization.     | Native Tensor Core support.        |
| Scaling  | Block-wise scaling (usually 128).  | Two-level Micro-block scaling (16).|

## 3. How do I get NVFP4 models?

- Use a NIM from the NGC. Models must support the arm64 archeticture for the GB10 

```bash
docker manifest inspect nvcr.io/nim/qwen/qwen3-32b-dgx-spark:1.0 | grep architecture
```
- Nvidia has compatible models at https://huggingface.co/nvidia/models
- Convert open source models manually

## 4. Important things to remember

1. Ollama only works with GGUF models. When you download a model from within Ollama it hosts it's own GGUF versions of select models. If you see a model on HuggingFace and has `*.safetensors` you need to convert these to a quant format or use vLLM or TensorRT-LLM to host these.

## Hands-on Lab: Quantizing Your First Model

We will use the NVIDIA TensorRT Model Optimizer container to convert a standard Hugging Face model into an NVFP4-optimized engine.

### Prepare the environment

1. Login to https://huggingface.co
2. Click your profile in the top right and select Access Tokens
3. Click Create New Token
4. Name the Token (e.g. gb10)
5. Under User Permissions (username) select
```
- Read access to contents of all repos under your personal namespace
- View access requests for all gated repos under your personal namespace
- Read access to contents of all public gated repos you can access
```
6. Copy the token and add this line to your `~/.bashrc`. ***This is the only opportunity to copy this token, it will not be available later and you'll have to regenerate a new one.***
```bash
export HF_TOKEN="your_token_here"
```

Reload your bash profile by logging out or
```bash
source ~/.bashrc
```

### Run the optimizer container

We will use the TensorRT-LLM Spark Dev container, which contains the specific libraries for the GB10's SM 12.1 architecture. Then clone the NVIDIA Model Optimizer git repo and execute that against the `Qwen/Qwen3-14B` model to quantanize the model using NVFP4.

```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  -e ACCELERATE_USE_FSDP=false \
  -e CUDA_VISIBLE_DEVICES=0 \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/TensorRT-Model-Optimizer && \
    cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model 'Qwen/Qwen3-14B' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  "
```

#### Notes
- Ignore error `pynvml.NVMLError_NotSupported: Not Supported`
- `ACCELERATE_USE_FSDP=false` and `CUDA_VISIBLE_DEVICES=0` are required for certain models. Otherwise the model will be split between CPU and GPU and we don't want that. It's fine to just leave these on for all models but the quant will take a bit longer.

#### What this command does (step‑by‑step)

- `docker run --rm -it`: start an interactive container and remove it when the command exits.
- `--gpus all`: gives the container access to all GPUs on the host via the NVIDIA container toolkit.
- `--ipc=host`: shares the host IPC namespace (large shared memory buffers used by some ML frameworks).
- `--ulimit memlock=-1 --ulimit stack=67108864`: increase locked memory and stack limits so large models and threads don't fail due to OS limits.
- `-v "./output_models:/workspace/output_models"`: bind-mounts a host folder for saving the converted/quantized model outputs.
- `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"`: reuses the host Hugging Face cache so downloads are cached and not re-fetched inside the container.
- `-e HF_TOKEN=$HF_TOKEN`: passes your Hugging Face token into the container (required only for gated models).
- `-e ACCELERATE_USE_FSDP=false -e CUDA_VISIBLE_DEVICES=0`: force single-GPU, non-FSDP runs so the optimization runs on the intended device and doesn't split across CPU/GPU.
- `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`: the TensorRT-LLM developer container with required libs for quantization/optimization; this image includes tooling tuned for NVIDIA hardware.

Inside the container the one-line `bash -c` does:

- Clone a specific Model Optimizer release (`git clone -b 0.35.0`) to ensure a known-good code version.
- `pip install -e '.[dev]'`: install the optimizer and dev dependencies inside the container.
- Set `ROOT_SAVE_PATH` to point at the mounted `/workspace/output_models` so results are written back to your host.
- Run the provided `huggingface_example.sh` script with these flags:
  - `--model 'Qwen/Qwen3-14B'`: the Hugging Face model to download and convert (gated — needs `HF_TOKEN`).
  - `--quant nvfp4`: quantize to NVIDIA NVFP4 (the format used on Blackwell/GB10).
  - `--tp 1`: tensor-parallel degree (1 for single-GPU in this example).
  - `--export_fmt hf`: export the result in Hugging Face format so downstream tools can consume it.

What you'll find after completion:

- The `./output_models` directory will contain the converted/quantized model files (bins/safetensors/configs). Use the `find` snippet below to verify.

### Quick verification commands (on host):

```bash
# List generated files
ls -la ./output_models/

# Find model files
find ./output_models/ -name "*.bin" -o -name "*.safetensors" -o -name "config.json"
```

### Serve the model with OpenAI-compatible API

#### Important
```
These NVFP4 models can't be run in Ollama. vLLM has some basic support but it's very new. TensorRT-LLM is really it for now. As the industry gets behind them support will improve. 
```

Start the TensorRT-LLM OpenAI-compatible API server with the quantized model. First, set the path to your quantized model:

```bash
# Set path to quantized model directory
export MODEL_PATH="/home/trevor/git/output_models/saved_models_Qwen3-14B_nvfp4_hf/"

docker run --rm -it \
  --name trtllm-server \
  --gpus all \
  --ipc=host \
  --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$MODEL_PATH:/workspace/model" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  trtllm-serve /workspace/model \
    --backend pytorch \
    --max_batch_size 4 \
    --host 0.0.0.0 \
    --port 8000
```

### Test out your TensorRT-LLM Server

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-14B",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

---

## Resources for Lesson 4

- https://build.nvidia.com/spark/nvfp4-quantization/instructions