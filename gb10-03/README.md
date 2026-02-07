# Session 3: The Model Zoo & The Blackwell "Secret Sauce" (NVFP4)

**Objective:** Understand the math that makes the GB10 a "Supercomputer at your desk." Move beyond simply downloading models and start optimizing them for the Blackwell architecture using NVFP4.

## 1. The 128GB Unified Memory Challenge

On a standard PC, you are limited by GPU VRAM (e.g., 24GB on a 5090). On the GB10, the 128GB of Unified Memory allows us to run massive models, but we still want to maximize "Tokens Per Second" (TPS).

- **The Bandwidth Bottleneck:** The GB10 uses LPDDR5x RAM (~273 GB/s). While this is plenty of capacity, it is slower than the HBM3 used in data center H100s.
- **The Solution:** Quantization. By shrinking the model size, we reduce the amount of data moving through the memory bus, directly increasing speed.

## 2. NVFP4: Why Blackwell is Different

Most AI enthusiasts are used to INT4 or GGUF quantization. Blackwell introduces a new hardware-native format: **NVFP4 (4-bit Floating Point)**.

| Feature  | Standard 4-bit (INT4)              | Blackwell NVFP4                    |
|----------|------------------------------------|------------------------------------|
| Logic    | Rounds numbers to whole integers.  | Uses 4-bit floating point (E2M1).  |
| Accuracy | High "perplexity" (loss of smarts).| Near-FP8 accuracy (<1% loss).      |
| Hardware | Software-based dequantization.     | Native Tensor Core support.        |
| Scaling  | Block-wise scaling (usually 128).  | Two-level Micro-block scaling (16).|

**The SE Talking Point:** "NVFP4 allows us to run a 70B model with the speed of a 4-bit model but the intelligence of an 8-bit model. It's the first time we don't have to choose between 'fast' and 'smart'."

## 3. Hands-on Lab: Quantizing Your First Model

We will use the NVIDIA TensorRT Model Optimizer container to convert a standard Hugging Face model into an NVFP4-optimized engine.

### Step A: Prepare the environment

In your VS Code terminal, create an output folder:

```bash
mkdir -p ~/gb10-training/models/nvfp4_output
```

Set your Hugging Face token (needed to pull gated models like Llama):

```bash
export HF_TOKEN="your_token_here"
```

### Step B: Run the optimizer container

We will pull the TensorRT-LLM Spark Dev container, which contains the specific libraries for the GB10's SM 12.1 architecture.

```bash
docker run --rm -it --gpus all \
  -v ~/gb10-training/models:/workspace/models \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

### Step C: Run the NVFP4 script

Inside the container, run the optimization script for a mid-sized model (e.g., Llama-3.1-8B):

```bash
python3 examples/quantization/quantize.py \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --output_dir /workspace/models/nvfp4_output \
  --qformat nvfp4 \
  --calib_size 128
```

Note: The `--calib_size` tells the AI to "calibrate" its math using 128 sample sentences to ensure it doesn't lose its mind during the shrink.

---

🌟 **Session 3 Challenge: The Efficiency Audit**

**Task:** Compare two models.

1. Download a standard GGUF (Q4_K_M) version of Llama-3-8B and run it in Ollama. Record the TPS (Tokens Per Second).
2. Run your newly created NVFP4 version of the same model using the `trtllm-python` backend.

**Observation:** You should notice a significantly faster "Time to First Token" (TTFT) on the NVFP4 version. Why? Because Blackwell's 5th-gen Tensor Cores process FP4 math natively without "unpacking" it first.

---

## Resources for Session 3

- Playbook: NVFP4 Quantization Guide
- Deep Dive: Introducing NVFP4 for Efficient Inference