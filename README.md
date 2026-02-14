# GB10 Training Program: Course Overview

This curriculum transitions Technical SEs into experts on the Blackwell architecture via the GB10, moving from foundational Linux workflows to advanced enterprise inference.

## Contents
- [Lesson 1](./gb10-01/)
- [Lesson 2](./gb10-02/)
- [Lesson 3](./gb10-03/)
- [Lesson 4](./gb10-04/)
- [Lesson 5](./gb10-05/)
- [Lesson 6](./gb10-06/)
- [Lesson 7](./gb10-07/)
- [Lesson 8](./gb10-08/)
- [Lesson 9](./gb10-09/)
- [Lesson 10](./gb10-10/)
- [Lesson 11](./gb10-11/)
- [Lesson 12](./gb10-12/)

## Phase 1: Foundations & Local Access
Focus: Getting comfortable with the Blackwell architecture and Linux-based workflows.

### [Lesson 1 — The "Unboxing" & Remote Development](./gb10-01/)
- Topics: Hardware tour (Blackwell/Grace), DGX OS (Ubuntu ARM64), SSH networking
- Lab: Configuring VS Code Remote‑SSH
- Playbook: VS Code Setup
- Deliverable: Access GB10 terminal and customize .bashrc for speed

### [Lesson 2 — Containerization for SEs](./gb10-02/)
- Topics: Why Docker?, NVIDIA Container Toolkit, managing GPU resources
- Lab: Pull and run a GPU-accelerated container; inspect nvidia-smi
- Key Concept: Environment isolation to prevent OS "clutter"

### [Lesson 3 — The Model Zoo: Sizes, Quants, and Blackwell](./gb10-03/)
- Topics: GGUF vs EXL2 vs FP4; 128GB LPDDR5x memory considerations
- Lab: Benchmark tokens-per-second (TPS) across quantization levels
- Playbook: NVFP4 Quantization

## Phase 2: Local Intelligence Workflows
Focus: Turning the GB10 into a coding and logic powerhouse.

### [Lesson 4 — Ollama & The Local API Backend](./gb10-04/)
- Topics: Installing Ollama, managing model pulls, Anthropic-compatible API
- Lab: Deploy Open WebUI for a ChatGPT-like internal experience
- Playbook: Ollama & Open WebUI

### [Lesson 5 — Vibe Coding & IDE Integration](./gb10-05/)
- Topics: Ghost-text and autocomplete; local vs cloud decisioning
- Lab: Set up Continue.dev extension in VS Code
- Playbook: Vibe Coding

### [Lesson 6 — GPU Container Orchestration with Kubernetes and SLURM](./gb10-06/)
- Topics: Kubernetes (microk8s) and SLURM
- Lab: Setup microk8s, install GPU Operator and deploy pods. Setup SLURM and execute GPU compute jobs
- Goal: Learn about container orchestration using Kuberneted and SLURM

## Phase 3: Visuals & Multi-Modal Demos
Focus: High-impact demos that "show" rather than "tell."

### [Lesson 7 — ComfyUI: Node-Based Image Gen](./gb10-07/)
- Topics: Diffusion models (FLUX.1/SDXL) on Blackwell; VRAM and resolution tradeoffs
- Lab: Install ComfyUI and run a basic text-to-image workflow
- Playbook: ComfyUI

### [Lesson 8 — Vision-Language Models (VLM)](./gb10-08/)
- Topics: Multi-modal AI; enabling GB10 to "see" images and video
- Lab: Live vision demo (webcam or file upload) where AI describes objects
- Playbook: Live VLM WebUI

### [Lesson 9 — RAG (Retrieval Augmented Generation)](./gb10-09/)
- Topics: Vector DBs and "Chat with your Docs"
- Lab: Ingest PDF directory (Sales Playbooks) and build a local search bot
- Playbook: RAG App in AI Workbench

## Phase 4: Enterprise Optimization & Advanced Topics
Focus: Performance tuning and professional customer delivery.

### [Lesson 10 — High-Throughput Serving (vLLM / SGLang)](./gb10-10/)
- Topics: Moving beyond Ollama; batching and multi-user serving
- Lab: Set up SGLang server for Blackwell performance
- Playbook: SGLang Inference Server

### [Lesson 11 — NVIDIA NIMs (Inference Microservices)](./gb10-11/)
- Topics: Transition local work to enterprise-grade production with NIMs
- Lab: Deploy a Llama-3 NIM and understand the Enterprise AI stack
- Playbook: NIM on Spark

### [Lesson 12 — Q & A](./gb10-12/)

## [Appendix](./appendix/)

- No space available on disk
- OOM (OutofMemory) Errors
- How to Reinstall the NVIDIA DGX Operating System
- Simple LLM Benchmark
- Manually Add Model to Ollama

