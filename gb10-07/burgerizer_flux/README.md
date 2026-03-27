# FLUX.1 LoRA Training

*Not sure this is actually working as expected. 

We are training a LoRA for the FLUX.1-dev model.

LoRA (Low-Rank Adaptation) is an efficient technique for fine-tuning large AI models (like Large Language Models or Stable Diffusion) without needing to retrain the entire model. Instead of updating all billions of parameters, which is computationally expensive, LoRA freezes the original model and injects small, trainable "adapter" matrices into specific layers (like attention layers).

Use ffmpeg to capture iframe every 30 seconds and crop for Flux.1-dev size (1024x1024)
```bash
ffmpeg -i video.mkv -vf "fps=1/30,scale=1024:1024:force_original_aspect_ratio=increase,crop=1024:1024" "~/gb10/images/bb_images/S10E02/S10E02_%04d.png"
```

Use default VLM `qwen2.5vl:32b` with the default Ollama host `http://localhost:11434` to auto caption images. This script is written to help the VLM recognize Bob's Burger characters and create accurate image captions. It will write the caption in a file next to the image with a .txt extension.

```bash
cd ~/git/gb10-training/gb10-07/burgerizer
python caption_burger.py --dir ~/gb10/images/bb_images/S10E02
```

Training

Training took 14-26 hours on 43 images. In the training output you want to watch for the `Val loss:` that is calculated after each checkpoint is saved. This should slowly drop as the model trains. Since we don't know how many epochs it will take to get the "best" checkpoint we save one every 200 steps, use the parameter `--save_every` to adjust this. 

| Val loss | Interpretation | Notes |
|---|---|---|
| 0.50+ | Underfit | Way too early. Keep going. |
| 0.25–0.35 | The Target Zone | This is where the magic usually happens for FLUX. |
| 0.15–0.20 | Risk of Overfit | The model may start losing flexibility or "burning" colors. |
| Below 0.10 | Deep Fried | Usually signifies the LoRA is too "stiff" and won't respond to prompts well. |

After struggling to get `ai-toolkit` working (it just ate up all the RAM and crashed) I used Github Copilot with the Claude Sonnet 4.6 model to write me a `train.py` script. Then wrapped this in a docker container. 

The script:
1. Creates embeddings for image/caption and saves the `.pt` file to `/output/cache`
2. Runs the training session, saving checkpoints to `/output`
3. When the training is finished it generates sample images using the base model, then using every checkpoint. Samples are in the `/output/samples` directory.

```bash
# Stop ALL docker containers (we need ALL that RAM)
docker stop $(docker ps -q)

hf download black-forest-labs/FLUX.1-dev --local-dir ~/gb10/models/comfyui/diffusion_models/FLUX.1-dev

cd ~/git/gb10-training/gb10-07/burgerizer
docker compose up -d
docker logs -f burgerizer
```

Testing LoRA

- Copy the checkpoint from `~/gb10/output_models` to `~/gb10/models/comfyui/loras`
- Load the `txt2img-flux1-dev-lora` workflow in ComfyUI. Change the model in the `Load LoRA (Model and CLIP)` node

Troubleshooting
```bash
# Look for Out-of-Memory errors. 
sudo dmesg -wH
[Feb21 17:13] NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359
```


