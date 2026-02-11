# OSEChat - GPU Monitoring Chat Application

A Chainlit-based chat application powered by Pydantic AI that monitors GPU status using nvidia-smi.

## Features

- 🤖 AI agent powered by Pydantic AI
- 💬 Interactive chat interface with Chainlit
- 🎮 Real-time GPU monitoring via nvidia-smi
- 🔧 Tool calling with visible agent "thinking" process
- 🐳 Fully Dockerized with GPU support

## Prerequisites

- Docker with NVIDIA Container Toolkit
- Ollama running locally with llama3.1 model

## Quick Start

1. **Ensure Ollama is running:**
   ```bash
   # If not installed, install Ollama first
   ollama serve
   ollama pull llama3.1
   ```

2. **Build and run the application:**
   ```bash
   docker compose up --build
   ```

3. **Access the chat:**
   Open your browser to http://localhost:8000

## Development Mode

The compose.yaml is configured for local development:
- `app.py` is mounted as a volume
- Chainlit runs with `--watch` flag
- Changes to `app.py` will auto-reload (no rebuild needed)

## Example Queries

Try asking the agent:
- "What's the GPU temperature?"
- "How much GPU memory is being used?"
- "Can you check the GPU status?"
- "Tell me about my GPU"

## Architecture

- **Framework**: Chainlit for UI, Pydantic AI for agent orchestration
- **Model**: Ollama with llama3.1 (OpenAI-compatible API)
- **Tool**: `get_nvidia_smi_stats` - Executes nvidia-smi and parses results
- **Visibility**: Agent thinking and tool calls shown via `cl.Step`

## Troubleshooting

**Ollama connection issues:**
- Ensure Ollama is running on `http://localhost:11434`
- Update `OLLAMA_BASE_URL` in compose.yaml if needed

**GPU not available:**
- Verify NVIDIA drivers: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`

**Port already in use:**
- Change the port mapping in compose.yaml: `"8001:8000"`

## FP4 Model Formats: MXFP4 vs NVFP4

When running quantized models on compatible hardware, you may encounter two main FP4 (4-bit floating point) formats:

### MXFP4 (Mixed eXponent Floating Point 4-bit)
- **Origin**: Developed by Meta for their Llama series models
- **Format**: 4-bit floating point with a shared exponent per group
- **Advantages**:
  - Better dynamic range due to shared exponent scheme
  - More balanced precision across values
  - Open source and well-documented
- **Use cases**: Llama 3.1 4-bit, Llama 3 4-bit, other Meta models

### NVFP4 (NVIDIA Floating Point 4-bit)
- **Origin**: NVIDIA's proprietary 4-bit format
- **Format**: 4-bit floating point optimized for NVIDIA hardware
- **Advantages**:
  - Hardware-accelerated on NVIDIA GPUs with FP4 support
  - Potentially faster inference on compatible GPUs
  - Optimized for Tensor Core operations
- **Use cases**: NVIDIA-optimized models, faster inference on RTX 40xx+ and newer GPUs

### Key Differences
| Feature | MXFP4 | NVFP4 |
|---------|-------|-------|
| Origin | Meta | NVIDIA |
| Hardware | Any (CPU/GPU) | NVIDIA GPUs only |
| Speed | Standard | Hardware-accelerated |
| Compatibility | Wide (Open) | NVIDIA-optimized |
| Exponent | Shared per group | NVIDIA proprietary |

### Notes
- Both formats reduce model size to ~25% of original FP16 size
- Quality differences are typically minimal for most use cases
- Choice depends on your hardware and inference speed requirements
- Ollama supports both formats - the model tag usually indicates which is used
