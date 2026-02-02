# Week 4: Ollama & The Interactive WebUI

**Objective:** Move from the command line to a professional-grade interface. Deploy Ollama as the local inference engine and Open WebUI as the "front door" for AI applications, providing "ChatGPT-like" performance while keeping 100% of the data local to the GB10.

## 1. The Power of Ollama on Blackwell

Ollama simplifies model management by bundling weights, configurations, and datasets into "Modelfiles." On the GB10, Ollama is highly optimized for ARM64 and Grace-Blackwell unified memory.

- **Architecture:** Ollama runs as a background service that manages the GPU lifecycle. It loads models into VRAM on demand and unloads them after inactivity to save power.
- **The Unified Advantage:** Because the GB10 has 128GB of memory, Ollama can easily "park" multiple large models (like Llama-3 70B and a Coding model) and swap between them in seconds.

## 2. Hands-on Lab: Deploying the AI Stack

We will use a "Bundled" approach, running Open WebUI and Ollama in a single container for maximum simplicity.

### Step A: Run the bundled container

This command launches the interface on port 11000 and ensures all chat history and models are saved permanently to your GB10 storage.

```bash
docker run -d -p 11434:11434 -p 11000:8080 \
  --gpus all \
  --name ai-stack \
  -v open-webui:/app/backend/data \
  -v ollama:/root/.ollama \
  --restart always \
  ghcr.io/open-webui/open-webui:ollama
```

- `-p 11000:8080`: Maps the Web interface to port 11000.
- `-p 11434:11434`: Exposes the Ollama API so you can connect Claude Code or VS Code to it later.
- `-v`: Creates "Volumes" so you don't lose your models when you stop the container.

### Step B: Accessing the UI

Open your laptop's browser and go to: `http://spark-xxxx.local:11000` (replacing `spark-xxxx` with your hostname).

- **First Run:** You will be asked to create an Admin account. This is entirely local to your machine.
- **Download a Model:** Click the "Settings" (gear icon) → "Models" → "Pull a Model."
  - Enter `qwen3-coder:32b` for a high-performance coding model.
  - Enter `llama3.3:70b` if you want to test the GB10's reasoning limits.

## 3. Integrating with External Tools

Now that your backend is running, you can connect other apps to your GB10's API.

- **For Claude Code:** You can now point Claude to your GB10 by setting:
  ```bash
  export ANTHROPIC_BASE_URL="http://spark-xxxx.local:11434"
  ```

- **For VS Code (Continue Extension):** In the `config.json` for the Continue extension, add a new model provider using the "Ollama" type and point the URL to your GB10's IP on port 11434.

---

🌟 **Week 4 Challenge: The "Multi-Model" Stress Test**

**Task:** Open two different browser tabs.

1. In Tab 1, start a long coding task with `qwen3-coder:32b`.
2. In Tab 2, start a creative writing task with `llama3.3:70b`.

**Watch the hardware:** Open your VS Code terminal and run `gpustats` (the alias we made in Week 1).

**Observe:** Watch how the GB10 manages the memory pressure. Notice the "Memory Used" jump as the 70B model loads, and check the "Volatile GPU-Util" to see the Blackwell cores in action.

---

## Resources for Week 4

- Playbook: Open WebUI with Ollama on Spark
- Documentation: Ollama API Reference