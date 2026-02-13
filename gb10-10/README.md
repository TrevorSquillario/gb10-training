# Lesson 10: High-Throughput Serving (SGLang)

**Objective:** Move from "One user, one model" to Concurrent Serving. Deploy SGLang, a high-performance serving framework that is currently the throughput champion for the Blackwell architecture. Learn how to handle dozens of simultaneous requests without the latency "choke" typical of simpler engines.

## Why SGLang over Ollama for Production?

While Ollama is great for personal "vibe coding," it lacks the advanced scheduling needed for multiple users.

- **RadixAttention:** SGLang uses a "Radix Tree" to manage the KV Cache. If 10 people ask the same question about a document, SGLang only computes the prompt once and shares the result instantly with the others.
- **Data Parallelism (--dp):** On the GB10, you can run multiple "copies" of the model weights in memory. SGLang can route different users to different copies, effectively doubling your throughput for smaller models (8B–32B).
- **Blackwell Native Kernels:** SGLang has specialized Triton kernels specifically tuned for the SM 12.1 (GB10) architecture, enabling higher token generation speeds than standard PyTorch.

## Hands-on Lab: Launching the SGLang Server

We will use the official Spark-optimized image to host a high-concurrency endpoint.

### Pull the optimized image

```bash
docker pull lmsysorg/sglang:spark
```

### Launch the server

We will serve a 32B model (Qwen3-Coder) and allocate 80% of the memory to the KV cache to support long, multi-turn conversations for many users.

```bash
docker run --rm --name sglang-server --gpus all \
  --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/gb10/models:/models \
  lmsysorg/sglang:spark \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen3-8B-NVFP4 \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.8 \
  --enable-torch-compile \
  --quantization modelopt_fp4
```

`--enable-torch-compile`: This triggers a "warm-up" phase where the model is compiled into optimized Blackwell machine code.

#### Test the model endpoint

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
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
# Select option 4
```

## The Multi-User Benchmark

Once the server is up, we will simulate a "Sales Team" hitting the server all at once.

Install the benchmark tool:

```bash
# Ensure our python venv is started
source ~/venv/gb10-training/bin/activate
pip install sglang[bench]
```

Run a stress test with 10 concurrent requests:

```bash
python3 -m sglang.bench_serving \
  --host 0.0.0.0 \
  --port 30000 \
  --dataset-name sharegpt \
  --num-prompts 50 \
  --max-concurrency 10 \
  --tokenizer ~/gb10/models/Qwen3-8B-NVFP4

nvtop
```

**Analyze the Results:** Look for the "Total Token Throughput". On a GB10, a 32B model should maintain a blistering pace even with 10 parallel conversations.

## How SGLang Manages GPU Memory and High Concurrency

- The weights are "read-only". Model weights are static; they don't change whether you're serving 1 user or 1,000. The large VRAM footprint (e.g., ~89GB) is mostly:
    - **Static weights:** the NVFP4 parameters of the model.
    - **KV cache metadata:** SGLang pre-allocates a large buffer (controlled by `--mem-fraction-static`, e.g., `0.8`) to store intermediate states for active conversations.

- Continuous batching (iteration-level batching). Instead of running requests one after another, the scheduler groups many requests into a single "super-batch" for each forward pass. For example:
    - Step 1: the GPU takes one token from Request A, one from Request B, one from Request C, etc.
    - Step 2: it runs those tokens through the model weights simultaneously using Tensor Cores.
    - Step 3: it emits the next token for all requests at once.

- RadixAttention: intelligent memory sharing. SGLang's Radix Tree manages the KV cache so common prefix context can be shared instead of duplicated. If many users share the same prompt prefix ("Explain quantum physics"), SGLang processes that part once, stores it in the tree, and multiple requests point to the same node. This prefix sharing saves massive VRAM and compute compared to engines that treat every request as entirely independent.

    - Context switching on the GPU is minimal because the GPU processes the entire batch in parallel across thousands of CUDA cores rather than switching back and forth between jobs.


## Hands-on Lab: 