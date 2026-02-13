# Lesson 7: Visualizing AI with ComfyUI & FLUX

**Objective:** Master high-fidelity image generation using ComfyUI. While previous Sessions focused on text, the GB10's 128GB of Unified Memory makes it a "visual powerhouse," capable of running massive models like FLUX.2-dev and Stable Diffusion 3.5 without the out-of-memory (OOM) errors common on consumer cards. This allows us to build text-to-image, image-to-image, text-to-video and many other workflows using ComfyUI's node based web UI.

## Hands-on Lab: Deploying ComfyUI on GB10

We will use an optimized Docker image designed specifically for the DGX Spark's Blackwell architecture. The official guide is here https://build.nvidia.com/spark/comfy-ui/instructions. So why not just follow that? Installing applications on the host OS is generally a no-no these days. We're in the age of containers, let's embrace it. This makes the application portable, allows us to separate the dependencies from other applications, allows us to specifically define all the storage locations, and easily add environment variables. This also includes NVIDIA SageAttention which adds some performance optimizations for our hardware. In the end we'll have a service that is production-ready (for your home lab). 

### Create model directories

We want to keep our large model files and other configs on the host system so we don't have to re-download them if the container restarts.

```bash
mkdir -p ~/gb10/models/comfyui/checkpoints
mkdir -p ~/gb10/models/comfyui/diffusion_models
mkdir -p ~/gb10/models/comfyui/unet
mkdir -p ~/gb10/models/comfyui/clip
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


### Downloading the "heavyweight" (FLUX.2)

ComfyUI is fine with the `*.safetensors` models. No need to worry about GGUF here. 

Download the FLUX.2-dev model:

```bash
mkdir -p ~/gb10/models/comfyui/diffusion_models/FLUX.2-dev
mkdir -p ~/gb10/models/comfyui/vae/FLUX.2-dev

# If don't have the hf cli downloaded
# Ensure our python venv is started
source ~/venv/gb10-training/bin/activate
pip install -U "huggingface_hub[cli]"

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

## Workflow: text-to-image 

Workflows can be loaded by File > Open or dragging dropping them onto the canvas. They can be saved as `json` files or `png` files. We'll start with a simple text-to-image workflow that uses generative AI to create an image from a prompt using the SDXL model.

1. From the Workflows panel on the left pane select `txt2img_sdxl_refiner`
2. Click Run
3. Use `nvtop` to monitor GPU usage and RAM
4. Workflow will take a few minutes to run. 

#### Knobs and Levers

On the `KSampler (Advanced)` node you have `cfg`
On the `total steps` and `steps spent on base` node you have `steps`

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


## Workflow: F5_TTS_Voice_Emulator_Record

This workflow will allow you to record your voice, clone the voice then generate speech from text based on that.

You'll need to add this flag to Chrome
chrome://flags/#unsafely-treat-insecure-origin-as-secure
http://<gb10-ip>:8188

```bash
cd ~/gb10/comfyui/custom_nodes
git clone https://github.com/niknah/ComfyUI-F5-TTS
cd ComfyUI-F5-TTS
git submodule update --init --recursive
```
## Workflow: F5_TTS_Voice_Emulator

This workflow will allow you to upload an audio file of someone speaking and transcribe the voice to text using the Whisper model. The transcribed text will be in the `Voice to Text` node. It will then clone the voice and use the text in the `Audio Sample Text (Must match audio exactly)` node to generate speech based on the clone. The reason for 2 text nodes is sometimes the transcribed text will be off slightly. It must match exactly for this to work properly

## Workflow: FLUX.2

There are also some workflows that use the FLUX.2 model. This is the state-of-the-art open source image model. They take a long time to run.


## Resources for Lesson 7



