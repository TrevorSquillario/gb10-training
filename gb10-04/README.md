# Session 4: Ollama & The Interactive WebUI

**Objective:** Move from the command line to a professional-grade interface. Deploy Ollama as the local inference engine and Open WebUI as the "front door" for AI applications, providing "ChatGPT-like" performance while keeping 100% of the data local to the GB10.

## 1. The Power of Ollama on Blackwell

Ollama simplifies model management by bundling weights, configurations, and datasets into "Modelfiles." On the GB10, Ollama is highly optimized for ARM64 and Grace-Blackwell unified memory.

- **Architecture:** Ollama runs as a background service that manages the GPU lifecycle. It loads models into VRAM on demand and unloads them after inactivity to save power.
- **The Unified Advantage:** Because the GB10 has 128GB of memory, Ollama can easily "park" multiple large models (like Llama-3 70B and a Coding model) and swap between them in seconds.

## 2. Hands-on Lab: Deploying the Local LLM Stack

### Step A: Run the bundled container

This command launches the Ollama WebUI interface on port 12000 the Ollama API on port 11434 and ensures all chat history and models are saved permanently to your GB10 storage at `/app`

```bash
sudo mkdir -p ~/ollama
sudo mkdir -p ~/open_webui
cd ~/git/gb10-training/gb10-04
docker compose up -d
```

### Step B: Accessing the UI

Open your laptop's browser and go to: `http://<gb10-ip>:12000`

1. **First Run:** You will be asked to create an Admin account. This is entirely local to your machine.
2. **Download a Model:** Click the "Settings" (gear icon) → "Models" → "Pull a Model."
  - Enter `llama3.3:70b` if you want to test the GB10's reasoning limits.
  - Enter `gpt-oss:20b` if you want to test with a smaller model.
3. Provide the prompt "What is AI?" and hit Enter
4. Open a new terminal tab or window and run `nvidia-smi` or use the `gpustats` bash alias to view the GPU and RAM utilization.
5. How many token/s is the GB10 outputting for the response? 

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