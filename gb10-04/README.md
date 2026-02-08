# Session 4: Ollama & The Interactive WebUI

**Objective:** Move from the command line to a professional-grade interface. Deploy Ollama as the local inference engine and Open WebUI as the "front door" for AI applications, providing "ChatGPT-like" performance while keeping 100% of the data local to the GB10.

## The Power of Ollama on Blackwell

Ollama simplifies model management by bundling weights, configurations, and datasets into "Modelfiles." On the GB10, Ollama is highly optimized for ARM64 and Grace-Blackwell unified memory.

- **Architecture:** Ollama runs as a background service that manages the GPU lifecycle. It loads models into VRAM on demand and unloads them after inactivity to save power.
- **The Unified Advantage:** Because the GB10 has 128GB of memory, Ollama can easily "park" multiple large models (like Llama-3 70B and a Coding model) and swap between them in seconds.

## Hands-on Lab: Deploying the Local LLM Stack

### Run the Docker Compose file to start the Ollama service with WebUI

The `docker compose` command launches the Ollama WebUI interface on port 12000 the Ollama API on port 11434 and ensures all chat history and models are saved permanently to your GB10 storage. It looks for a `docker-compose.yaml` or `compose.yaml` file by default.

```bash
sudo mkdir -p ~/models
sudo mkdir -p ~/ollama
sudo mkdir -p ~/open_webui
cd ~/git/gb10-training/gb10-04
docker compose up -d
```

#### Docker Compose (compose.yaml) Summary
- **Services:** ollama (local LLM inference) and open-webui (web frontend).
- **Ports:** Ollama API: `11434`; WebUI: host `12000` → container `8080`.
- **Volumes:** Persists data/models to `~/ollama`, `~/models`, and `~/open_webui`.
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
4. Open a new terminal tab or window and run `nvidia-smi` or use the `gpustats` bash alias to view the GPU and RAM utilization.
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
#### 3. Manually download and copy to container models directory
The last method is by manually adding the model files and creating the model definition. You don't need to run through this, it is here for future reference if you need it.

*Note: 
```The Q8 (8-big) quant I selected for this model requires 243GB of RAM so it's not actually going to run on the GB10. If you want to use this model you'll need to choose the IQ4_XS version```

Start by installing the hf cli and download the model.

```bash
pip install -U "huggingface_hub[cli]"
# FYI: This model is 268GB. We'll include a specific subdirectory for the the 8-bit quant
hf download unsloth/MiniMax-M2.1-GGUF \
  --include "*Q8_0*" \
  --local-dir ~/models/MiniMax-M2.1-GGUF_Q8_0
```
For models split into parts Ollama requires that they are merged into one file

```bash
# Run a temporary container to use the llama.cpp tools
docker run --rm \
  -v ~/models:/models \
  --entrypoint /llm/llama-gguf-split \
  amperecomputingai/llama.cpp:latest \
  --merge \
  /models/MiniMax-M2.1-GGUF_Q8_0/Q8_0/MiniMax-M2.1-Q8_0-00001-of-00005.gguf \
  /models/MiniMax-M2.1-GGUF_Q8_0/MiniMax-M2.1-Q8_0-merged.gguf
```

Create the model definition
```bash
cd ~/models/MiniMax-M2.1-GGUF_Q8_0
vi Modelfile

FROM ./MiniMax-M2.1-Q8_0-merged.gguf

# This template handles standard chat AND tool calling for MiniMax
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}
{{- if .Tools }}
When you need to use a tool, you must respond in JSON format:
{"name": "function_name", "parameters": {"arg": "value"}}
Available tools:
{{ .Tools }}
{{- end }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ if .Thinking }}<think>
{{ .Thinking }}</think>
{{ end }}{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<think>"
PARAMETER stop "</think>"
PARAMETER temperature 1
PARAMETER num_ctx 32768
```

Create the model using the `ollama` command in the container 
```bash
docker exec -it ollama /bin/bash
# CD to the /models volume mount we specified in the compose.yaml. This corresponds to your ~/models directory on the host.
cd /models/MiniMax-M2.1-GGUF_Q8_0
ollama create minimax-q8 -f Modelfile

# Verify the model is working with
ollama run minimax-q8
```

---

🌟 **Challenge 1: The NVFP4 Difference**

Download the Qwen3-14B-GGUF 8-bit quant model
```bash
docker exec -it ollama ollama run hf.co/Qwen/Qwen3-14B-GGUF:Q8_0
```

Start the TensorRT-LLM Server from the previous lab but this time specify the docker network `ai-network` that the Ollama WebUI service is using so they will be able to talk to each other. 
```bash
# Set path to quantized model directory
export MODEL_PATH="/home/trevor/git/output_models/saved_models_Qwen3-14B_nvfp4_hf/"

docker run --rm -it \
  --name trtllm-server \
  --gpus all \
  --ipc=host \
  --network ai-network \
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

1. Go to the Ollama WebUI `http://<gb10-ip>:12000`
2. Click your profile icon in the top right, select Admin Panel
3. Click the Settings tab, select Connections
4. Under OpenAI API, click the + sign on the right to "Add Connection
```
URL: http://trtllm-server:8000/v1
Auth: Bearer > tensorrt_llm
```
5. Click the circle with arrows to test the connection
6. Click Save
7. Click New Chat on the left panel
8. The Local tab will contain models downloaded in Ollama directly. The External tab will show our external connections like the TensorRT-LLM hosted model we just started
9. Compare the performance of the `Qwen3-14B-GGUF:Q8_0` local model with the NVFP4 quant we created. It will appear as `model`.


---

## Resources for Session 4

- https://build.nvidia.com/spark/open-webui