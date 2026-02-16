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
mkdir -p ~/gb10/ollama
mkdir -p ~/gb10/open_webui
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
#### 3. Manually download and copy to models directory

```bash
# If don't have the hf cli downloaded
# Ensure our python venv is started
source ~/venv/gb10-training/bin/activate
pip install -U "huggingface_hub[cli]"

# Go to the Hugging Face page for a particular model then to the `Files and versions` tab. Click the file you want to download then find the `Copy download link` button. 
wget -P  ~/gb10/models/Qwen3-8B-Q8_0 https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf

# Create the Modelfile https://docs.ollama.com/modelfile
cat << EOF >  ~/gb10/models/Qwen3-8B-Q8_0/Modelfile
FROM ./Qwen3-8B-Q8_0.gguf

# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 32768
EOF

# Create the model in Ollama
docker exec -it ollama /bin/bash
# CD to the /models volume mount we specified in the compose.yaml. This corresponds to your ~/gb10/models directory on the host.
cd /models/Qwen3-8B-Q8_0
ollama create minimax-q8 -f Modelfile
```

See the appendix for a more advanced example with split .gguf files: [Manually Add Model to Ollama](../appendix/README.md)

### Choosing the Right Model

There are two questions we must ask ourselves:

1. Will the full `fp16/bf16` model fit into RAM? Yes. This true some of the image models in ComfyUI. We have enough RAM so let's get the most out of this model. For that we should choose the full `fp16/bf16` model.

2. Will the full `fp16/bf16` model fit into RAM? No. This is true of some LLMs and some the top coding models. So we should pick the largest quant that fits into the amount of RAM we have available. If you go to the HuggingFace site and login, go to the model page of a GGUF model, find the `Log In to add your hardware` link on the right above the quants. If you put in your GB10 it will tell you which ones fit. Really it just highlights the ones below 128GB in green. 

Finding an NVFP4 variant from another user on HuggingFace is another option. These are supported in Ollama and ComfyUI.

#### Choosing Model Size

- **Larger models (e.g. 32B+):** stronger reasoning, better long-form coherence, and higher factual accuracy for complex tasks; require much more VRAM, longer latency, and are costlier to run. Use when quality and deep understanding matter (research, coding, summarization, few-shot tasks).
- **Medium models (e.g. 8B):** a balance of capability and performance — suitable for many production chatbots and single-host deployments where moderate quality with reasonable latency is desired.
- **Smaller models (e.g. 4B and below):** fast, low-latency, and memory-efficient. Great for high-throughput services, edge or resource-constrained environments, or when you combine them with retrieval (RAG) for factual accuracy.
- **Quantization trade-offs:** lower-bit quant (e.g., Q4/Q8) reduces memory and increases speed but can slightly degrade output quality; test quant levels for your task.
- **Practical tip:** start with a smaller model for iteration and scale up to larger models for evaluation on representative prompts; consider ensemble or retrieval-augmented approaches to get the best of both worlds.

The main difference between gpt-oss:20b and a quantized gpt-oss:120b (a "quant") is a trade-off between native efficiency and compressed "brain power."

Think of it like comparing a high-performance sports car (20b) to a massive semi-truck that has been stripped down to fit into a garage (120b quant). Both are from OpenAI’s "Open-Source Series" (OSS) and use a Mixture-of-Experts (MoE) architecture, but they serve very different roles.

1. Intelligence & Reasoning Depth
While the 20b model is surprisingly smart for its size, the 120b model is a powerhouse. Even when quantized (compressed), the 120b variant generally retains a higher "intelligence ceiling."

gpt-oss:20b: Optimized for low-latency and local tasks. It's excellent for coding and STEM but can struggle with the deepest levels of nuance or highly complex multi-step planning compared to its big brother.

gpt-oss:120b (Quant): Even at 4-bit quantization (MXFP4), it rivals frontier models like o3-mini or o4-mini. It has a much larger "internal library" of knowledge and better "zero-shot" reasoning.

### Hands-on Lab: Ollama Model Runner

Pull three Qwen3 models at different sizes:

```bash
docker exec -it ollama ollama pull qwen3:8b
docker exec -it ollama ollama pull gpt-oss:20b
docker exec -it ollama ollama pull qwen3:32b
```

Open 3 tabs of Ollama and start a New Chat, select each model and execute a prompt. We want to run these all at the same time and we can do that because they are small models. Ollama spins up a separate model runner for each different model. 

You can use the prompt:

```
Explain quantum computing in simple terms.
```

1. Go to the Terminal and run `nvtop` take notice of multiple instances of ollama running, their RAM and CPU usage. 
2. Compare speed and quality of responses. At the bottom of the response you can hover over the i icon and make note of the response_tokens/s. This is how fast we are generating tokens, smaller models higher t/s.


## Resources for Lesson 4

- https://build.nvidia.com/spark/open-webui