# Week 7: Visualizing AI with ComfyUI & FLUX.1

**Objective:** Master high-fidelity image generation using ComfyUI. While previous weeks focused on text, the GB10's 128GB of Unified Memory makes it a "visual powerhouse," capable of running massive models like FLUX.1-dev and Stable Diffusion 3.5 without the out-of-memory (OOM) errors common on consumer cards.

## 1. Why ComfyUI for Sales Engineers?

ComfyUI is a node-based interface. Instead of just a "chat box," it shows the "plumbing" of how AI works.

- **The Demo Factor:** It allows you to build custom "workflows" (e.g., an SE-specific workflow that generates a LinkedIn banner with your company logo automatically).
- **Efficiency:** Unlike other UIs, ComfyUI only executes the parts of the graph that changed. On the GB10, this means near-instant iterations once the model is loaded into memory.
- **The Memory Advantage:** FLUX.1-dev requires significant VRAM to run at full precision. The GB10 handles this natively, allowing for 2K resolution images that look like professional photography.

## 2. Hands-on Lab: Deploying ComfyUI on GB10

We will use an optimized Docker image designed specifically for the DGX Spark's Blackwell architecture.

### Step A: Create model directories

We want to keep our large model files on the host system so we don't have to re-download them if the container restarts.

```bash
mkdir -p ~/comfyui_models/checkpoints
mkdir -p ~/comfyui_models/unet
mkdir -p ~/comfyui_models/clip
```

### Step B: Launch the ComfyUI container

Run this command to start the server on port 8188.

```bash
docker run -d --name spark-comfy --gpus all \
  -p 8188:8188 \
  -v ~/comfyui_models:/workspace/ComfyUI/models \
  knamdar/spark_comfy_ui:v1
```

### Step C: Downloading the "heavyweight" (FLUX.1)

FLUX.1 is currently the gold standard for open-weights image generation.

Navigate to your checkpoints folder:

```bash
cd ~/comfyui_models/checkpoints
```

Download the FP8 (optimized) version to save space while maintaining quality:

```bash
wget https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors
```

Access the UI: Open `http://spark-xxxx.local:8188` in your browser.

## 3. Workflow: The "Professional Headshot" Demo

Once in ComfyUI, you will load a "Workflow JSON" from our course repo.

- **The Goal:** Input a prompt like: "A professional cinematic headshot of a software engineer in a modern office, 8k resolution, highly detailed skin texture."
- **The Secret:** Use the Blackwell-optimized sampling nodes. Because the GB10 supports CUDA 13.0, it can utilize faster cross-attention kernels that reduce generation time for FLUX images from minutes to under 60 seconds.

---

🌟 **Week 7 Challenge: The "Product Design" Sprint**

**Task:** Use ComfyUI to design a "NVIDIA-themed" mechanical keyboard.

1. Load the `Basic_Flux_Workflow.json`.
2. **Prompt:** "A top-down view of a mechanical keyboard, neon green LED backlighting, brushed aluminum chassis, high-tech industrial design, 4k."
3. **The Tweak:** Change the CFG Scale and Steps to see how it affects the "vibe" of the image.
4. **Benchmark:** Use the "ComfyUI Manager" to time your execution. Compare the speed of a 1024x1024 image vs a 2048x2048 image. On a GB10, the high-res jump is surprisingly small due to the massive memory bandwidth.

---

## Resources for Week 7

- Playbook: ComfyUI on DGX Spark Instructions
- Models: FLUX.1-dev on Hugging Face

> **Pro Tip:** If your ComfyUI feels "stuck" on the first run, check the logs with `docker logs -f spark-comfy`. It is likely just initializing the Blackwell-specific kernels, which only happens once!

> **Next Step:** Ready for Week 8: Vision-Language Models (VLM), where we give your GB10 "eyes" to describe images and live video feeds?

**Video Resource:** NVIDIA DGX Spark - Comfy UI Image Generation Demo. This video provides a practical walkthrough of setting up ComfyUI and Stable Diffusion on the DGX Spark, which perfectly aligns with our goal of generating high-quality local AI art.
