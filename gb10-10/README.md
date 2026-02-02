# Week 10: High-Throughput Serving (SGLang & vLLM)

**Objective:** Move from "One user, one model" to Concurrent Serving. Deploy SGLang, a high-performance serving framework that is currently the throughput champion for the Blackwell architecture. Learn how to handle dozens of simultaneous requests without the latency "choke" typical of simpler engines.

## 1. Why SGLang over Ollama for Production?

While Ollama is great for personal "vibe coding," it lacks the advanced scheduling needed for multiple users.

- **RadixAttention:** SGLang uses a "Radix Tree" to manage the KV Cache. If 10 people ask the same question about a document, SGLang only computes the prompt once and shares the result instantly with the others.
- **Data Parallelism (--dp):** On the GB10, you can run multiple "copies" of the model weights in memory. SGLang can route different users to different copies, effectively doubling your throughput for smaller models (8B–32B).
- **Blackwell Native Kernels:** SGLang has specialized Triton kernels specifically tuned for the SM 12.1 (GB10) architecture, enabling higher token generation speeds than standard PyTorch.

## 2. Hands-on Lab: Launching the SGLang Server

We will use the official Spark-optimized image to host a high-concurrency endpoint.

### Step A: Pull the optimized image

```bash
docker pull lmsysorg/sglang:spark
```

### Step B: Launch the server

We will serve a 32B model (Qwen3-Coder) and allocate 80% of the memory to the KV cache to support long, multi-turn conversations for many users.

```bash
docker run -d --name sglang-server --gpus all \
  --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:spark \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Coder-32B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.8 \
  --enable-torch-compile
```

`--enable-torch-compile`: This triggers a "warm-up" phase where the model is compiled into optimized Blackwell machine code.

## 3. The Multi-User Benchmark

Once the server is up, we will simulate a "Sales Team" hitting the server all at once.

Install the benchmark tool:

```bash
pip install sglang[bench]
```

Run a stress test with 50 concurrent requests:

```bash
python3 -m sglang.bench_serving --host 0.0.0.0 --port 30000 \
  --dataset-name sharegpt --num-prompts 50 --parallel 10
```

**Analyze the Results:** Look for the "Total Token Throughput". On a GB10, a 32B model should maintain a blistering pace even with 10 parallel conversations.

---

🌟 **Week 10 Challenge: The "Shared Context" Win**

**Task:** Prove the power of RadixAttention.

1. Send a very long prompt (e.g., a 10,000-word transcript) to the SGLang server and ask for a summary. Note the "First Token Latency."
2. Immediately send a different question about the same transcript.

**Observe:** The second response should start instantly (under 100ms) because SGLang "remembered" the transcript in its Radix tree.

**Compare:** Try the same two-step process in Ollama. You will notice Ollama has to "re-process" the entire transcript every single time.

---

## Resources for Week 10

- Playbook: SGLang for Inference on DGX Spark
- Technical Paper: SGLang: Efficient Execution of Structured LLM Programs