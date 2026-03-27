
Use ffmpeg to capture screenshots of a video every 10 seconds
```bash
for f in *.mkv; do ffmpeg -i "$f" -vf "fps=1/10" "/tmp/bb_images/$(echo "$f" | grep -oP 'S\d+E\d+')_%04d.png"; done
```

https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_advanced.md

# Kohya-SS Script
```bash
# Slow speed
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="bb_dataset.toml" \
  --output_dir="/output" \
  --output_name="bb_sdxl_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --network_dim=8 \
  --network_alpha=1 \
  --network_train_unet_only \
  --optimizer_type="adafactor" \
  --lr_scheduler="constant" \
  --learning_rate=4e-7 \
  --max_train_epochs=5 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_text_encoder_outputs

# Full speed/VRAM
accelerate launch --config_file accelerate_config.yaml --num_cpu_threads_per_process=12 sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="bb_dataset.toml" \
  --output_dir="/output" \
  --output_name="bob_belcher_v1" \
  --save_every_n_steps=200 \
  --save_model_as=safetensors \
  --max_train_steps=2000 \
  --learning_rate=1.0 \
  --optimizer_type="Prodigy" \
  --optimizer_args "weight_decay=0.01" "decouple=True" "use_bias_correction=True" \
  --lr_scheduler="constant" \
  --network_module=networks.lora \
  --network_dim=64 \
  --network_alpha=32 \
  --mixed_precision="bf16" \
  --full_bf16 \
  --gradient_checkpointing \
  --no_half_vae \
  --mem_eff_attn
```

# Tensorboard
```bash
docker exec -it burgerizer_sdxl /bin/bash
tensorboard --logdir=/output/logs --port=6006 --bind_all > /dev/null 2>&1 &
```