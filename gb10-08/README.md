# Session 8: Giving the GB10 "Eyes" (Vision-Language Models)

**Objective:** Explore Multi-modal AI by moving beyond text and static images to Vision-Language Models (VLMs). Build a live interface that can "see" and describe the world in real-time, showcasing the low-latency inference capabilities of the Blackwell architecture.

## 1. The VLM Landscape: Moondream & LLaVA

While LLMs understand words, VLMs understand spatial relationships and visual context.

- **Moondream2:** A tiny but mighty 1.6B parameter model. It's perfect for real-time video because it can process multiple frames per second on the GB10.
- **LLaVA v1.6 (Large Language-and-Vision Assistant):** A more powerful 7B or 13B model that can handle complex reasoning (e.g., "Find the safety hazard in this factory photo").

**Why GB10?** Video processing requires massive throughput. The GB10's 5th-gen Tensor Cores are optimized for the interleaved vision/text tokens used in these architectures, providing a 2x-3x speedup over previous generations.

## 2. Hands-on Lab: Real-Time Vision WebUI

We will deploy the Live VLM WebUI playbook, which uses a Streamlit interface to pipe a webcam or video file into the model.

### Step A: Launch the VLM container

This container comes pre-loaded with the drivers needed to access local video devices.

```bash
docker run -d --name spark-vlm --gpus all \
  -p 8501:8501 \
  --device /dev/video0:/dev/video0 \
  nvcr.io/nvidia/dgx-spark/live-vlm-webui:latest
```

**Note:** If you don't have a webcam plugged into the GB10, you can still use the interface to upload MP4 video files.

### Step B: Interacting with the "eyes"

Open `http://spark-xxxx.local:8501` in your browser.

1. Select **Moondream2** for the fastest response.
2. Click "Start Camera."
3. **The Prompt:** In the chat box, type: "What am I holding in my hand?" or "Describe the expression on the person's face."

## 3. Workflow: Automated Image Tagging

SEs often need to show how AI can automate "boring" tasks. We will use a Python script to perform Structured Data Extraction from images.

Use Ollama to pull the vision-capable LLaVA model:

```bash
ollama pull llava
```

Run a query that forces a JSON response (perfect for developers):

```bash
# Example of asking a VLM to categorize an image for a database
ollama run llava "Describe this image in JSON format with keys: 'object', 'color', 'estimated_value'" < path/to/photo.jpg
```

---

🌟 **Session 8 Challenge: The "Security Guard" Script**

**Task:** Create a simple "Security" demo.

1. Use the Live VLM WebUI with a video of a busy office or a street scene.
2. **The Goal:** Write a prompt that makes the AI act as an observer.
3. **Example:** "Alert me if you see a person wearing a red hat."

**Observation:** Notice the "Tokens Per Second" vs. "Frames Per Second." On a GB10, you should be able to analyze 2-5 frames per second with Moondream2, which is fast enough for basic motion monitoring.

---

## Resources for Session 8

- Playbook: Live VLM WebUI on DGX Spark
- Blog: VLM Prompt Engineering Guide

> **Pro Tip:** If the video feed feels choppy, check your browser console. Sometimes the 1GbE network can struggle with high-resolution MJPEG streams. Lowering the resolution to 720p usually fixes the "lag" without hurting the AI's accuracy.

> **Next Step:** Ready for Session 9: Retrieval Augmented Generation (RAG), where we teach the AI to answer questions using your own private company documents?

**Video Resource:** Local AI Models with Ollama + NVIDIA DGX Spark. This video demonstrates setting up an Ollama server on the DGX Spark to run large models locally, which is a key component for the VLM and LLM workflows we are building in this course.
