# Session 3: The Model Zoo & The Blackwell "Secret Sauce" (NVFP4)

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

## Hands-on Lab: Quantizing Your First Model

We will use the NVIDIA TensorRT Model Optimizer container to convert a standard Hugging Face model into an NVFP4-optimized engine.

### Step A: Prepare the environment

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

### Step B: Run the optimizer container

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

Notes:
- Ignore error `pynvml.NVMLError_NotSupported: Not Supported`
- `ACCELERATE_USE_FSDP=false` and `CUDA_VISIBLE_DEVICES=0` are required for certain models. Otherwise the model will be split between CPU and GPU and we don't want that. It's fine to just leave these on for all models but the quant will take a bit longer.

### Step C: Validate the quantized model
After the container completes, verify that the quantized model files were created successfully.

```bash
# Check output directory contents
ls -la ./output_models/

# Verify model files are present
find ./output_models/ -name "*.bin" -o -name "*.safetensors" -o -name "config.json"
```

### Step D: Serve the model with OpenAI-compatible API

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

### Step E: Test out your TensorRT-LLM Server

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

## Resources for Session 4

- https://build.nvidia.com/spark/nvfp4-quantization/instructions