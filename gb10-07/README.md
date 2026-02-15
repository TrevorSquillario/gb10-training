# Lesson 7: Visualizing AI with ComfyUI 

**Objective:** Master high-fidelity image generation using ComfyUI. While previous Sessions focused on text, the GB10's 128GB of Unified Memory makes it a "visual powerhouse," capable of running massive models like FLUX.2-dev and Stable Diffusion 3.5 without the out-of-memory (OOM) errors common on consumer cards. This allows us to build text-to-image, image-to-image, text-to-video and many other workflows using ComfyUI's node based web UI.

## Hands-on Lab: Deploying ComfyUI on GB10

We will use an optimized Docker image designed specifically for the DGX Spark's Blackwell architecture. The official guide is here https://build.nvidia.com/spark/comfy-ui/instructions. So why not just follow that? Installing applications on the host OS is generally a no-no these days. We're in the age of containers, let's embrace it. This makes the application portable, allows us to separate the dependencies from other applications, allows us to specifically define all the storage locations, and easily add environment variables. This also includes NVIDIA SageAttention which adds some performance optimizations for our hardware. In the end we'll have a service that is production-ready (for your home lab). 

### Create model directories

We want to keep our large model files and other configs on the host system so we don't have to re-download them if the container restarts.

Since we are mapping the docker volume like `~/gb10/models/comfyui:/opt/ComfyUI/models` just the last subdirectory is important and must match exactly. ComfyUI nodes will look in these specific subdirectories

```bash
mkdir -p ~/gb10/models/comfyui/checkpoints
mkdir -p ~/gb10/models/comfyui/diffusion_models
mkdir -p ~/gb10/models/comfyui/controlnet
mkdir -p ~/gb10/models/comfyui/clip
mkdir -p ~/gb10/models/comfyui/clip_vision
mkdir -p ~/gb10/models/comfyui/ipadapter
mkdir -p ~/gb10/models/comfyui/loras
mkdir -p ~/gb10/models/comfyui/model_patches
mkdir -p ~/gb10/models/comfyui/unet
mkdir -p ~/gb10/models/comfyui/upscale_models
mkdir -p ~/gb10/models/comfyui/vae
mkdir -p ~/gb10/comfyui/custom_nodes
mkdir -p ~/gb10/comfyui/output
mkdir -p ~/gb10/comfyui/input
mkdir -p ~/gb10/comfyui/wheels
```

You can also create custom model paths https://docs.comfy.org/development/core-concepts/models

### Launch the ComfyUI container

We are going to build our own container based on a `Dockerfile` that includes optimizations for the Blackwell architecture. Based on https://github.com/ecarmen16/SparkyUI

```bash
cd gb10-07/comfyui
# I leave off the -d until I see the container start successfully. Then Ctrl + C and add the -d to run the container in the background.
docker compose up
# You should eventually see "[ComfyUI-Manager] All startup tasks have been completed."

# Open a browser to http://<gb10-ip>:8188
```

#### Note: 
- Only the top level folders matter under `~/gb10/models/comfyui`. These are used by nodes to populate the models. You can create whatever subdirectories you would like. 
- When adding models the ComfyUI needs and F5 refresh to see them

Model Types Explained

- **Diffusion:** The core generative model (checkpoints go in `diffusion_models`). It iteratively denoises a latent to produce images from noise. The "Load Diffusion Model" node will show models from this directory.
- **CLIP (text encoder):** Converts text prompts into embeddings that guide the diffusion model during generation (store CLIP models in `clip`). The "DualCLIPLoader" node will show models from this directory.
- **VAE:** The encoder/decoder that maps between image and latent spaces; used to encode images into latents and decode latents back into high-quality images (store VAE files in `vae`). The "Load VAE" node will show models from this directory.

## Workflows

In ComfyUI workflows can be complex and difficult to understand so it's best to use existing workflows to understand what nodes are needed, what they do and how others have done it. Some great resources are available at:

*Warning: These sites contain NSFW material and are blocked on the corporate network
- https://civitai.com
- https://openart.ai/workflows/home
- https://aistudynow.com/category/comfyui-workflows

## Workflow: text-to-image 

Workflows can be loaded by File > Open or dragging dropping them onto the canvas. They can be saved as `json` files or `png` files. We'll start with a simple text-to-image workflow that uses generative AI to create an image from a prompt using the SDXL model.

1. From the Workflows panel on the left pane select `txt2img_sdxl_refiner`
2. Click Run
3. Use `nvtop` to monitor GPU usage and RAM
4. Workflow will take a few minutes to run. 

#### Knobs and Levers

***Steps (The "Iterations")***

Think of Steps as the amount of time and effort the AI spends "chiseling" an image out of a block of random static (noise).

How it works: Each step is a pass where the AI tries to remove a little bit of noise and replace it with detail that matches your prompt.

- Low Steps (1–8): Usually results in blurry, "soupy" images with no texture. Only used for "Lightning" or "Schnell" distilled models.

- Medium Steps (20–35): The "Goldilocks Zone" for most models. Details are sharp, and the image is coherent.

- High Steps (50+): Diminishing returns. Beyond 50 steps, you are often just making tiny, unnoticeable changes while burning electricity on your GB10.

***CFG: Classifier Free Guidance (The "Strictness")***

CFG is a multiplier that determines how hard the AI should try to follow your text prompt versus how much it should rely on its "artistic intuition."

- Low CFG (1.0 – 3.0): The AI is "creative" and loose. Colors are often more natural/realistic, but it might ignore parts of your prompt.

- Standard CFG (4.0 – 8.0): This is the classic range for Stable Diffusion. It tries hard to give you exactly what you asked for.

- High CFG (10.0+): The AI becomes "obsessive." Colors get oversaturated, edges get sharp/neon, and the image can "fry" or look over-baked.

***Seed (Randomness / Reproducibility)***

- **Seed:** An integer used to initialize the pseudo-random number generator the sampler uses. The same model, prompt, workflow, and settings combined with the same seed will produce the same image output (reproducible). Changing the seed produces different variations. Leaving the seed blank or using a sentinel like `-1` causes a new random seed to be chosen each run.

***Denoise (Denoising Strength)***

- **Denoise (Denoising Strength):** Controls how strongly the model removes noise and follows the conditioning (prompt or source image). In image-to-image (img2img) workflows, a denoise value of `0.0` keeps the original image unchanged, while `1.0` lets the model fully regenerate the image from noise. Lower values preserve structure and make only subtle edits; higher values produce larger, more creative changes. For text-to-image, the effect is tied into the sampling process (steps and scheduler); increasing denoise-like parameters generally increases variability and may require more steps for stable results.

## Workflow: qwen-image-edit

This is the state-of-the-art open source image editing model. The results are impressive. Upload your Base Image, add your prompt and click Run (`Ctrl + Enter`)! The purple nodes are on bypass mode and won't be used. This will let you switch upscalers if you wish. 

```bash
wget -P  ~/gb10/models/comfyui/unet https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-BF16.gguf
wget -P  ~/gb10/models/comfyui/loras https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
wget -P  ~/gb10/models/comfyui/checkpoints https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0Q_fp16.safetensors
wget -P  ~/gb10/models/comfyui/vae https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
wget -P  ~/gb10/models/comfyui/clip https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

# SeedVR2 Upscaler
wget -P  ~/gb10/models/comfyui/SEEDVR2 https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors
wget -P  ~/gb10/models/comfyui/SEEDVR2 https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_7b_fp16.safetensors

# Ultimate SD Upscaler
wget -P ~/gb10/models/comfyui/upscale_models https://huggingface.co/Tenofas/ComfyUI/resolve/d79945fb5c16e8aef8a1eb3ba1788d72152c6d96/upscale_models/4x_NMKD-Siax_200k.pth
```

## Workflow: FLUX.2

There are also some workflows that use the FLUX.2 model. This is the state-of-the-art open source image model. They take a long time to run.

### Downloading FLUX.2

ComfyUI is fine with the `*.safetensors` models. No need to worry about GGUF here. 

Download the FLUX.2-dev model:

```bash
mkdir -p ~/gb10/models/comfyui/diffusion_models/FLUX.2-dev
mkdir -p ~/gb10/models/comfyui/vae/FLUX.2-dev

wget -P  ~/gb10/models/comfyui/vae https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
wget -P  ~/gb10/models/comfyui/checkpoints https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors
wget -P  ~/gb10/models/comfyui/checkpoints https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors

# If you want to try the FLUX.2 image generation model. Requires submitting your into on Hugging Face. Other models like Llama require this as well. 
hf download black-forest-labs/FLUX.2-dev \
  --include "*flux2-dev.safetensors" \
  --local-dir ~/gb10/models/comfyui/diffusion_models/FLUX.2-dev

# These are nested in the hf model repo so we download them to /tmp to keep things clean
hf download Comfy-Org/flux2-dev \
  --include "*mistral_3_small_flux2_bf16.safetensors" \
  --local-dir /tmp
mv /tmp/split_files/text_encoders/*.safetensors ~/gb10/models/comfyui/clip/

hf download Comfy-Org/flux2-dev \
  --include "*flux2-vae.safetensors" \
  --local-dir /tmp
mv /tmp/split_files/vae/*.safetensors ~/gb10/models/comfyui/vae/
```

## Workflow: Qwen-TTS_VoiceClone

This workflow will allow you to upload an audio file of someone speaking and transcribe the voice to text using the Qwen3-ASR (Automated Speech Recognition).  Then it clones the voice and generates speech from the `reference_text` based in the cloned voice. You don't need a very long sample 10-15 seconds.

## Workflow: F5_TTS_Voice_Emulator_Record

This workflow will allow you to record your voice, clone the voice then generate speech from text based on that.

You'll need to add this flag to Chrome to enable mic access for non-HTTPS sites
chrome://flags/#unsafely-treat-insecure-origin-as-secure
http://<gb10-ip>:8188

```bash
cd ~/gb10/comfyui/custom_nodes
git clone https://github.com/niknah/ComfyUI-F5-TTS
cd ComfyUI-F5-TTS
git submodule update --init --recursive
```

## Resources

- Self-hosted web based Photoshop https://www.photopea.com




