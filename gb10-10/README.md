# Lesson 10: High-Throughput Serving (SGLang & NIMs)

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


## What are NIMs?

Think of NIM as the "Professional Wrapper" for everything we've learned. While SGLang is a raw engine, a NIM is a complete service.

- **Standardized APIs:** Every NIM exposes an industry-standard OpenAI-compatible API. This means if you build an app using a Llama-3 NIM, you can swap it for a Mistral NIM without changing a single line of your application code.
- **Architecture-Aware Optimization:** When you launch a NIM on your GB10, it automatically detects the Blackwell architecture and selects the most optimized TensorRT-LLM or vLLM backend specifically for the SM 12.1 instruction set.
- **Enterprise Security:** NIMs are part of the NVIDIA AI Enterprise stack, meaning they come with security patches and are validated to run 24/7 in production environments.

## 2. Hands-on Lab: Deploying a Llama-3 NIM

We will deploy the Llama-3.1-8B-Instruct NIM, which is officially supported on the DGX Spark.

### Step A: Authentication (NGC)

NIMs are hosted on the NVIDIA GPU Cloud (NGC). You will need your API key from Lesson 9.

Get your NGC API Key
1. Go to [ngc.nvidia.com](ngc.nvidia.com) and sign in/create an account. 
2. Confirm your email
3. Create your Nvidia Cloud account when it prompts you. Select your name when it asks you to select a Team/Organization
4. Eventually you'll see your name in the top right. Select Setup or go directly to [https://org.ngc.nvidia.com/setup](https://org.ngc.nvidia.com/setup)
5. Click Generate API Key or go to [https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys)
```
  Key Name: gb10 (Name it whatever you want)
  Expiration: Never
  Key Permissions: NGC Catalog
```
6. Copy the key and save it somewhere. ***This will be the only time you will be able to copy this key***

Now let's start the NIM containers for FP8 and NVFP4 of the `qwen3-32b` model

First login to the `nvcr.io` container registry. Enter `$oauthtoken` as-is, this is important.

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <Your-NGC-API-Key>
```

https://www.nvidia.com/en-us/solutions/ai/agentic-ai

#### Start the NIM container for the Qwen3-32B model
*You can add the `export` line to your `~/.bashrc`
```bash
export NGC_API_KEY=<Your-NGC-API-Key>
```

Create the NIM cache directory
```bash
mkdir -p /app/cache/nim
sudo chmod -R 777 /app/cache
```

```bash
docker run -it --rm --name qwen3-nim \
  --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "/app/cache/nim:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/mistralai/mistral-7b-instruct-v0.3:1.12.0
```

```bash
# Login to the NVIDIA container registry
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Step B: Launching the NIM

This command pulls the NIM and starts the microservice. We map the port to 8000 to differentiate it from our other servers.

```bash
export MODEL_NAME="meta/llama-3.1-8b-instruct"

docker run -it --rm --name llama3-nim \
  --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

`-v ~/.cache/nim`: This ensures that once the model is downloaded, it stays on your GB10 even if you stop the container.

### Step C: Testing the production API

Open a second terminal and send a request to your new enterprise endpoint:

```bash
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role":"user", "content":"What are the three core benefits of NVIDIA NIM?"}]
  }'
```

## 3. Workflow: The "Self-Healing" Demo

In a production environment, if a service crashes, it must restart.

- **The Demo:** Show the customer how the NIM handles errors. Stop the container (`docker stop llama3-nim`) and restart it.
- **Observation:** Notice how quickly it comes back online. Because the model weights are already cached in your `~/.cache/nim` folder, the "Time to Ready" is minimal, which is a key requirement for enterprise SLAs.

---

🌟 **Lesson 11 Challenge: The "API Switcheroo"**

**Task:** Prove the portability of NIM.

1. Create a simple Python script that asks an AI to summarize a text using the URL `http://localhost:8000/v1`.
2. Deploy a second NIM (e.g., Mistral-7B) on a different port (e.g., 8001).
3. Change only the port number in your script.

**Goal:** Show that your "Customer Application" works perfectly with two different AI models without any code refactoring.

---

## Resources for Lesson 10

- Playbook: NIM on Spark Documentation https://build.nvidia.com/spark/nim-llm

