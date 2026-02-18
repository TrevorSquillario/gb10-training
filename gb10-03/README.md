# Lesson 3: The Model Zoo & The Blackwell "Secret Sauce" (NVFP4)

**Objective:** Understand the math that makes the GB10 a "Supercomputer at your desk." Move beyond simply downloading models and start optimizing them for the Blackwell architecture using NVFP4.

## The 128GB Unified Memory Challenge

On a standard PC, you are limited by GPU VRAM (e.g., 24GB on a 5090). On the GB10, the 128GB of Unified Memory allows us to run massive models, but we still want to maximize "Tokens Per Second" (TPS).

- **The Bandwidth Bottleneck:** The GB10 uses LPDDR5x RAM (~273 GB/s). While this is plenty of capacity, it is slower than the HBM3 used in data center H100s.
- **The Solution:** Quantization. By shrinking the model size, we reduce the amount of data moving through the memory bus, directly increasing speed.

## NVFP4: Why Blackwell is Different

Most AI enthusiasts are used to INT4 or GGUF quantization. Blackwell introduces a new hardware-native format: **NVFP4 (4-bit Floating Point)**.

| Feature  | Standard 4-bit (INT4)              | Blackwell NVFP4                    |
|----------|------------------------------------|------------------------------------|
| Logic    | Rounds numbers to whole integers.  | Uses 4-bit floating point (E2M1).  |
| Accuracy | High "perplexity" (loss of smarts).| Near-FP8 accuracy (<1% loss).      |
| Hardware | Software-based dequantization.     | Native Tensor Core support.        |
| Scaling  | Block-wise scaling (usually 128).  | Two-level Micro-block scaling (16).|

## How do I get NVFP4 models?

- Use a NIM from the NGC. Models must support the arm64 archeticture for the GB10 

```bash
docker manifest inspect nvcr.io/nim/qwen/qwen3-32b-dgx-spark:1.0 | grep architecture
```
- Nvidia has compatible models at https://huggingface.co/nvidia/models
- Convert open source models manually

## FP4 Model Formats: MXFP4 vs NVFP4

When running quantized models on compatible hardware, you may encounter two main FP4 (4-bit floating point) formats:

### MXFP4 (Mixed eXponent Floating Point 4-bit)
- **Origin**: Developed by Meta for their Llama series models
- **Format**: 4-bit floating point with a shared exponent per group
- **Advantages**:
  - Better dynamic range due to shared exponent scheme
  - More balanced precision across values
  - Open source and well-documented
- **Use cases**: Llama 3.1 4-bit, Llama 3 4-bit, other non-meta models as well

### NVFP4 (NVIDIA Floating Point 4-bit)
- **Origin**: NVIDIA's proprietary 4-bit format
- **Format**: 4-bit floating point optimized for NVIDIA hardware
- **Advantages**:
  - Hardware-accelerated on NVIDIA GPUs with FP4 support
  - Potentially faster inference on compatible GPUs
  - Optimized for Tensor Core operations
- **Use cases**: NVIDIA-optimized models, faster inference on RTX 40xx+ and newer GPUs

### Notes
- Both formats reduce model size to ~25% of original FP16 size
- Quality differences are typically minimal for most use cases
- Choice depends on your hardware and inference speed requirements
- Ollama supports both formats - the model tag usually indicates which is used

## Model Hosting

Common hosting options you'll encounter when deploying models on the GB10:

- **Ollama** — Local, developer-friendly model runner.
  - Pros: Extremely simple to install and use for local testing; good for iterating on prompts and small-scale demos.
  - Cons: Not designed for high concurrency or maximum throughput; limited to supported model formats and runtimes.
  - Use cases: Local development, demos, single-user testing.

- **TensorRT-LLM** — NVIDIA's production-grade engine optimized for Blackwell/NVFP4.
  - Pros: Best raw latency and throughput on NVIDIA hardware (supports NVFP4/TensorRT engines); excellent for large models on GB10.
  - Cons: NVIDIA-only, more complex to build and deploy; requires GPUs, Docker and specific tooling; steeper ops cost.
  - Use cases: Low-latency, high-throughput production inference on GB10/DGX systems.

- **vLLM** — High-throughput, research-friendly model server.
  - Pros: Easy OpenAI-compatible API, good batching and memory optimizations; works across GPUs/CPUs for many HF models.
  - Cons: Not as deeply hardware-optimized for NVFP4/TensorRT as TensorRT-LLM; can be memory-hungry for very large models.
  - Use cases: Multi-tenant inference, experimentation at scale, mid-tier production serving.

- **SGLang** — A throughput-focused serving framework (lesson 10 discusses this in depth).
  - Pros: Exceptional concurrent performance (RadixAttention, KV cache sharing), Blackwell-native kernels for top TPS on GB10.
  - Cons: More specialized and complex; newer ecosystem and tooling than Ollama/vLLM.
  - Use cases: High-concurrency production serving where many users share similar context or prompts.



## Hands-on Lab: Download model from huggingface.co

First ensure our python virtual environment is started. Refer to Lesson 1 if you haven't created this venv. Then download the model.
```bash
source ~/venv/gb10-training/bin/activate
pip install -U "huggingface_hub[cli]"

mkdir -p ~/gb10/models
hf download nvidia/Qwen3-8B-NVFP4 \
  --local-dir ~/gb10/models/Qwen3-8B-NVFP4

tree ~/gb10/models/Qwen3-8B-NVFP4
```

## Hands-on Lab: Serving a model using NVIDIA TensorRT-LLM

Now we will use TensorRT-LLM to serve up the model on port `8001`. This will host an OpenAI API compatible endpoint. 

```bash
cd gb10-03/trtllm
docker compose up
```

### Test the model endpoint

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Simple LLM Benchmark

See the appendix for step-by-step instructions to set this up: [Simple LLM Benchmark](../appendix/README.md)

Start the benchmark
```bash
python ~/git/gb10-training/appendix/llm_benchmark.py
# Select option 2
```

### (Optional) Advanced Benchmark/Stress Testing
```bash
pip install aiperf

aiperf profile \
  --model nvidia/Llama-3.3-70B-Instruct-NVFP4 \
  --url http://localhost:8001 \
  --endpoint-type chat \
  --request-rate 1 \
  --request-count 1 \
  --streaming
```

## Stop trtllm container. Due to memory limitations if we're using a large model we can only run either TensorRT-LLM or vLLM. 

`Ctrl + C` in the window it's running or `docker stop ttrtllm`

## Hands-on Lab: Serving a model using vLLM

Now we will use vLLM to serve up the model on port `8002`. This will host an OpenAI API compatible endpoint. 

```bash
cd gb10-03/vllm
docker compose up
```

Test the model endpoint

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Simple LLM Benchmark

See the appendix for step-by-step instructions to set this up: [Simple LLM Benchmark](../appendix/README.md)

Start the benchmark
```bash
python ~/git/gb10-training/appendix/llm_benchmark.py
# Select option 3
```

### View the report
```bash
python ~/git/gb10-training/appendix/llm_benchmark.py --report
```

## Hands-on Lab: Generation Parameters & Advanced Prompting

This lab explores how different parameters and prompting techniques affect model behavior. Make sure you have either TensorRT-LLM (port 8001) or vLLM (port 8002) running from the previous labs.

### Part 1: Understanding Generation Parameters

#### OpenAI API request fields

- **model**: The model identifier or local model path to use. For TensorRT-LLM this is the model name; for vLLM you can pass a local path to the model directory.
- **messages**: Array of chat messages in OpenAI-compatible format. Each item should include `role` (e.g., `user`, `assistant`, `system`) and `content` (the prompt text).
- **max_tokens**: Maximum number of tokens the model will generate in the response. Lower values limit response length.
- **temperature**: Sampling temperature controlling randomness. `0.0` is deterministic; higher values (e.g., `0.7`, `1.0`) produce more varied replies.
- **top_p**: Nucleus sampling threshold (0.0-1.0). Only considers tokens whose cumulative probability is within top_p. Lower values = more focused responses.
- **top_k**: Only sample from the top K most likely tokens. Limits vocabulary diversity per token generated.
- **stream**: When `true`, the server sends partial tokens as they are generated (useful for low-latency streaming UIs). When `false`, the full response is returned in one payload.

#### Exercise 1.1: Temperature Comparison

Temperature controls randomness. Run the same prompt with different temperatures to see the effect:

**Temperature = 0.0 (Deterministic, Factual)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "Write a creative story about a robot learning to paint."}],
    "max_tokens": 150,
    "temperature": 0.0
  }' | jq -r '.choices[0].message.content'
```

**Temperature = 0.7 (Balanced)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "Write a creative story about a robot learning to paint."}],
    "max_tokens": 150,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
```

**Temperature = 1.2 (Creative, Unpredictable)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "Write a creative story about a robot learning to paint."}],
    "max_tokens": 150,
    "temperature": 1.2
  }' | jq -r '.choices[0].message.content'
```

**Observation:** Run each command 2-3 times. Temperature 0.0 produces identical output each time, while higher temperatures generate varied creative responses.

#### Exercise 1.2: Top-P (Nucleus Sampling)

Top-P controls the diversity by limiting the token pool based on cumulative probability:

**Top-P = 0.1 (Very Focused)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.1
  }' | jq -r '.choices[0].message.content'
```

**Top-P = 0.9 (More Diverse)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }' | jq -r '.choices[0].message.content'
```

**Observation:** Lower top_p values produce more conservative, predictable language. Higher values allow more vocabulary diversity.

#### Exercise 1.3: Practical Use Cases

**Use Case: Code Generation (Deterministic)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers recursively."}
    ],
    "max_tokens": 200,
    "temperature": 0.0
  }' | jq -r '.choices[0].message.content'
```

**Use Case: Creative Writing (High Temperature)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [
      {"role": "user", "content": "Write a haiku about artificial intelligence and creativity."}
    ],
    "max_tokens": 100,
    "temperature": 1.0
  }' | jq -r '.choices[0].message.content'
```

**Use Case: Factual Q&A (Low Temperature, Low Top-P)**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [
      {"role": "user", "content": "What is the capital of France and what is its population?"}
    ],
    "max_tokens": 100,
    "temperature": 0.2,
    "top_p": 0.5
  }' | jq -r '.choices[0].message.content'
```

### Part 2: Advanced Prompting Techniques

#### Exercise: System Prompts and Persona Creation

System prompts guide the model's behavior and personality. They are processed before user messages:

**Example: Technical Expert Persona**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [
      {"role": "system", "content": "You are a senior DevOps engineer with 15 years of experience. You explain concepts clearly with practical examples and best practices. You prefer Kubernetes and Docker for deployments."},
      {"role": "user", "content": "How should I deploy a Python web application?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
```

**Example: Creative Writing Coach Persona**
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [
      {"role": "system", "content": "You are an enthusiastic creative writing coach. You encourage vivid descriptions, show-don'\''t-tell techniques, and character development. Always provide specific actionable feedback."},
      {"role": "user", "content": "Review this sentence: The man was sad."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
```

## Advanced Hands-on Lab: Quantizing Your First Model

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
  -v "~/gb10/output_models:/workspace/output_models" \
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
- `-v "~/gb10/output_models:/workspace/output_models"`: bind-mounts a host folder for saving the converted/quantized model outputs.
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

- The `~/gb10/output_models` directory will contain the converted/quantized model files (bins/safetensors/configs). Use the `find` snippet below to verify.

### Quick verification commands (on host):

```bash
# List generated files
ls -la ~/gb10/output_models/

# Find model files
find ~/gb10/output_models/ -name "*.bin" -o -name "*.safetensors" -o -name "config.json"
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
- https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/trt-llm
- https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm
