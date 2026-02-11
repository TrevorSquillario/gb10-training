# Lesson 4: Ollama & The Interactive WebUI

**Objective:** Move from the command line to a professional-grade interface. Deploy Ollama as the local inference engine and Open WebUI as the "front door" for AI applications, providing "ChatGPT-like" performance while keeping 100% of the data local to the GB10.

## The Power of Ollama on Blackwell

Ollama simplifies model management by bundling weights, configurations, and datasets into "Modelfiles." On the GB10, Ollama is highly optimized for ARM64 and Grace-Blackwell unified memory.

- **Architecture:** Ollama runs as a background service that manages the GPU lifecycle. It loads models into VRAM on demand and unloads them after inactivity to save power.
- **The Unified Advantage:** Because the GB10 has 128GB of memory, Ollama can easily "park" multiple large models (like Llama-3 70B and a Coding model) and swap between them in seconds.
- **Supported Models** Ollama works with GGUF models by default. When you download a model from within Ollama it hosts it's own GGUF versions of select models. If you see a model on HuggingFace and has `*.safetensors` you can usually find the GGUF variant or import them manually https://docs.ollama.com/import

## Hands-on Lab: Deploying the Local LLM Stack

### Run the Docker Compose file to start the Ollama service with WebUI

The `docker compose` command launches the Ollama WebUI interface on port 12000 the Ollama API on port 11434 and ensures all chat history and models are saved permanently to your GB10 storage. It looks for a `docker-compose.yaml` or `compose.yaml` file by default.

```bash
sudo mkdir -p ~/gb10/ollama
sudo mkdir -p ~/gb10/open_webui
cd ~/git/gb10-training/gb10-04
docker compose up 
# Make sure the container starts without error then Ctrl + C and run it in the background (-d)

docker compose up -d
```

#### Docker Compose (compose.yaml) Summary
- **Services:** ollama (local LLM inference) and open-webui (web frontend).
- **Ports:** Ollama API: `11434`; WebUI: host `12000` → container `8080`.
- **Volumes:** Persists data/models to `~/gb10/ollama`, `~/gb10/models`, and `~/gb10/open_webui`.
- **GPU:** `ollama` reserves NVIDIA GPUs via `deploy.resources.reservations.devices` (requires GPU-enabled Docker; `deploy` is swarm-specific).
- **Networking:** Both attach to `ai-network` (bridge); `open-webui` depends_on `ollama` and uses `OLLAMA_BASE_URL` to talk to it.
- **Auth:** `WEBUI_AUTH=True` enables a login screen for the WebUI.

### Accessing the UI

Open your laptop's browser and go to: `http://<gb10-ip>:12000`

1. **First Run:** You will be asked to create an Admin account. This is entirely local to your machine.
2. **Download a Model:** Click the "Settings" (gear icon) → "Models" → "Pull a Model."
  - Enter `llama3.3:70b` if you want to test the GB10's reasoning limits.
  - Enter `gpt-oss:20b` if you want to test with a smaller model.
3. Provide the prompt "What is AI?" and hit Enter
4. Open a new terminal tab or window and run `nvidia-smi` or `nvtop` to view the GPU and RAM utilization.
5. How many token/s is the GB10 outputting for the response? 

### Useful commands

- `docker exec -it ollama ollama list`: List models downloaded
- `sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"`: If you get a message about not enough memory when it's less than 128GB and it should work or did before.

### Downloading Models

There are 3 ways that you can add models to Ollama

#### 1. From ollama.com directly
Type the model name into the search box. If you go to https://ollama.com you can search for models easier there, then copy/paste the name into Ollama.
#### 2. From huggingface.co through ollama.com
Using the `ollama` command line. Since we're running Ollama in a container we'll have to execute the `ollama` command through the container using `docker exec`
```bash
# For models hosted in ollama.com directly
docker exec -it ollama ollama pull qwen3-coder-next

# For models on HuggingFace. When on the model page on huggingface.co look for a "Use this model" dropdown on the right, select Ollama and it will give you the proper string. Be sure to select the proper quant. 
docker exec -it ollama ollama run hf.co/Qwen/Qwen3-14B-GGUF:Q8_0
```
#### 3. (Advanced) Manually download and copy to container models directory
The last method is by manually adding the model files and creating the model definition. You don't need to run through this, it is here for future reference if you need it.

See the appendix for step-by-step instructions: [Manually Add Model to Ollama](../appendix/README.md)

### Hands-on Lab: LLM Benchmark

See the appendix for step-by-step instructions to set this up: [Simple LLM Benchmark](../appendix/README.md)

First download the model

```bash
docker exec -it ollama ollama pull qwen3:8b
```
Then start the benchmark
```bash
python ~/git/gb10-training/appendix/llm_benchmark.py
# Select option 1
```

### Hands-on Lab: Model Size Comparison

**Objective:** Compare quality, speed, and memory usage across quantization levels using Qwen3 models at different sizes.


#### Choosing Model Size

- **Larger models (e.g. 32B+):** stronger reasoning, better long-form coherence, and higher factual accuracy for complex tasks; require much more VRAM, longer latency, and are costlier to run. Use when quality and deep understanding matter (research, coding, summarization, few-shot tasks).
- **Medium models (e.g. 8B):** a balance of capability and performance — suitable for many production chatbots and single-host deployments where moderate quality with reasonable latency is desired.
- **Smaller models (e.g. 4B and below):** fast, low-latency, and memory-efficient. Great for high-throughput services, edge or resource-constrained environments, or when you combine them with retrieval (RAG) for factual accuracy.
- **Quantization trade-offs:** lower-bit quant (e.g., Q4/Q8) reduces memory and increases speed but can slightly degrade output quality; test quant levels for your task.
- **Practical tip:** start with a smaller model for iteration and scale up to larger models for evaluation on representative prompts; consider ensemble or retrieval-augmented approaches to get the best of both worlds.

#### Download Models

Pull three Qwen3 models at different sizes:

```bash
# 4B model variants
docker exec -it ollama ollama pull qwen3:4b

# 8B model variants  
docker exec -it ollama ollama pull qwen3:8b

# 32B model variants
docker exec -it ollama ollama pull qwen3:32b
```

Open a New Chat, select each model and compare the results. You can use the prompt:

```
Explain quantum computing in simple terms.
```

Compare speed and quality of responses.

## Resources for Lesson 4

- https://build.nvidia.com/spark/open-webui