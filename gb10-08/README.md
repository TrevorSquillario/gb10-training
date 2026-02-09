# Lesson 8: Giving the GB10 "Eyes" (Vision-Language Models)

**Objective:** Explore Multi-modal AI by moving beyond text and static images to Vision-Language Models (VLMs). Build a live interface that can "see" and describe the world in real-time, showcasing the low-latency inference capabilities of the Blackwell architecture.

## 1. The VLM Landscape: Moondream & LLaVA

While LLMs understand words, VLMs understand spatial relationships and visual context.

- **Moondream2:** A tiny but mighty 1.6B parameter model. It's perfect for real-time video because it can process multiple frames per second on the GB10.
- **LLaVA v1.6 (Large Language-and-Vision Assistant):** A more powerful 7B or 13B model that can handle complex reasoning (e.g., "Find the safety hazard in this factory photo").

**Why GB10?** Video processing requires massive throughput. The GB10's 5th-gen Tensor Cores are optimized for the interleaved vision/text tokens used in these architectures, providing a 2x-3x speedup over previous generations.

## 2. Hands-on Lab: Real-Time Vision WebUI

We will deploy the Live VLM WebUI playbook, which uses a Streamlit interface to pipe a webcam or video file into the model.

### Launch the VLM container

This container comes pre-loaded with the drivers needed to access local video devices.

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <Your-NGC-API-Key>

docker run --name spark-vlm --gpus all \
  --network host \
  -p 8090:8090 \
  ghcr.io/nvidia-ai-iot/live-vlm-webui:0.2.1
```

**Note:** If you don't have a webcam plugged into the GB10, you can still use the interface to upload MP4 video files. You are also able to use an IP camera RTSP stream https://github.com/nvidia-ai-iot/live-vlm-webui/blob/main/docs/usage/rtsp-ip-cameras.md

### Interacting with the "eyes"

Download the VLM models
```bash
docker exec -it ollama ollama pull gemma3:4b
docker exec -it ollama ollama pull gemma3:27b
docker exec -it ollama ollama pull llama3.2-vision:11b
```

Open `http://<gb10-ip>:8090` in your browser.

1. VLM API Configuration
```bash
API Base URL: http://localhost:11434/v1
API Key: ollama
Model: llama3.2-vision:11b
```
2. Use your webcam or RTSP stream (e.g. rtsp://192.168.0.100:8554/back_cam)
3. Click "Connect to Stream and Start VLM Analysis"

---

🌟 **Lesson 8 Challenge: Live Object Tracking**

This demo uses ffmpeg to take a public traffic camera's video stream which is an m3u8 stream and convert it to an RTSP stream using mediamtx. It then uses the ultralytics YOLO (You Only Look Once) model and library to enable object tracking. Then streams the labeled output to a webpage.  

```bash
cd gb10-08/YOLO
docker compose up

# Open a browser to http://<gb10-ip>:5000
```

