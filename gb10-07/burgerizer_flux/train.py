#!/usr/bin/env python3
"""
FLUX.1-dev LoRA fine-tuning — two-pass memory-efficient approach.

Pass 1 (precompute): Load VAE + text encoders once, encode every image/caption,
                     save tensors to CACHE_DIR, then purge encoders from VRAM.
Pass 2 (train):      Load only the transformer, wrap with LoRA, train using the
                     cached embeddings with flow-matching loss, save weights.

Usage examples:
    # Basic run with defaults
    python train.py

    # Custom paths and hyperparams
    python train.py \\
        --model_id /models/FLUX.1-dev \\
        --data_dir /input \\
        --output_dir /output \\
        --resolution 512 \\
        --lora_rank 16 \\
        --num_epochs 20 \\
        --batch_size 1 \\
        --learning_rate 1e-4

"""

import copy
import os
import argparse
import json
import time
from datetime import datetime, timedelta
import sys

import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxPipeline
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from tqdm import tqdm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARGUMENT PARSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(
        description="FLUX.1-dev LoRA fine-tuning (two-pass, memory-efficient)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Paths ---
    parser.add_argument(
        "--model_id", type=str, default="/models/FLUX.1-dev",
        help="Local path or HuggingFace hub ID of the FLUX.1-dev model",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/input",
        help="Directory containing .png/.jpg images and matching .txt captions",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/output",
        help="Root output directory for checkpoints and the final LoRA",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Cache directory for pre-computed latents (default: <output_dir>/cache)",
    )
    parser.add_argument(
        "--val_dir", type=str, default="/input_val",
        help="Directory containing held-out validation images + .txt captions. "
             "If set, validation loss is computed at each checkpoint step.",
    )
    parser.add_argument(
        "--val_cache_dir", type=str, default=None,
        help="Cache directory for pre-computed val latents (default: <output_dir>/val_cache)",
    )

    # --- Image ---
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Training resolution in pixels; must be divisible by 16",
    )

    # --- LoRA ---
    parser.add_argument(
        "--lora_rank", type=int, default=32,
        help=(
            "LoRA rank (r). Higher values increase the adapter capacity and the "
            "number of LoRA parameters roughly linearly, which increases GPU memory "
            "usage for the adapter weights and their optimizer state (AdamW keeps "
            "additional moments). Doubling `r` approximately doubles LoRA-related "
            "memory—reduce `batch_size` if you hit OOM when raising `r`. Typical "
            "values: 8-64 depending on GPU memory and desired capacity."
        ),
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help=(
            "LoRA alpha scaling factor. Scales the adapter outputs to control "
            "their effective contribution to the model. Changing `lora_alpha` "
            "affects training dynamics and signal strength but has negligible "
            "impact on VRAM (it does not change parameter count). Choose higher "
            "values to increase LoRA influence; a common heuristic is `alpha = r`."
        ),
    )

    # --- Optimisation ---
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=186,
        help="Number of full passes over the training set",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help=(
            "Training batch size. Directly controls VRAM usage — each sample in a batch "
            "requires holding its activations, intermediate tensors, and gradients in VRAM "
            "simultaneously. Doubling batch_size roughly doubles activation memory. "
            "Use 1 on consumer GPUs; the GB10's 128 GB unified memory allows larger values."
        ),
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=2,
        help=(
            "Gradient accumulation steps. Does NOT increase VRAM — instead of running one "
            "large batch, it runs (effective_batch / grad_accum_steps) sized batches "
            "sequentially, summing gradients before each optimizer step. The effective "
            "batch size (batch_size * grad_accum_steps) behaves identically to a real "
            "large batch for the optimizer, at the cost of more forward/backward passes "
            "per update step."
        ),
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Training dtype; bfloat16 recommended on Blackwell/GB10",
    )
    parser.add_argument(
        "--save_every", type=int, default=200,
        help="Save an intermediate checkpoint every N gradient steps",
    )
    # --- DataLoader ---
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help=(
            "Number of DataLoader worker subprocesses for loading batches in parallel "
            "with GPU training. Uses CPU RAM, not VRAM — but on the GB10/DGX Station "
            "where CPU and GPU share unified memory, large values combined with "
            "pin_memory=True can indirectly cause CUDA OOM by consuming memory the GPU "
            "allocator needs. Reduce this (or prefetch_factor) if you see unexpected OOM "
            "with large datasets. Set to 0 to load in the main process (useful for debugging)."
        ),
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=None,
        help=(
            "Number of batches each DataLoader worker pre-loads into CPU memory ahead of "
            "the GPU consuming them. Higher values keep the GPU fed during bursts of slow "
            "I/O at the cost of more CPU RAM. Only valid when num_workers > 0; ignored "
            "(and set to None) when num_workers=0. If left unset (None) and num_workers > 0, "
            "defaults to 4 automatically."
        ),
    )

    # --- FLUX-specific ---
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Guidance scale passed to the FLUX transformer (dev model only)",
    )
    parser.add_argument(
        "--max_t5_length", type=int, default=256,
        help="Maximum T5 token sequence length (256 recommended; 512 uses more VRAM)",
    )

    # --- Sampling ---
    parser.add_argument(
        "--sample_prompt", type=str, default="bbstyle, Bob Belcher standing in front of his restaurant",
        help="Prompt used to generate comparison images at each checkpoint (skip if empty)",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=28,
        help="Number of inference steps for sample generation (FLUX.1-dev needs 28-50 for quality output)",
    )

    # --- Cache ---
    parser.add_argument(
        "--preload_cache", action="store_true", default=True,
        help=(
            "Pre-load all cached .pt tensors into CPU RAM before training starts. "
            "Eliminates per-batch disk I/O at the cost of RAM. Strongly recommended "
            "on the GB10 / DGX Station with 128 GB unified memory. Pass "
            "--no-preload_cache to disable and revert to per-sample disk reads."
        ),
    )

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--generate_samples_only",
        action="store_true",
        default=False,
        help=(
            "Skip pre-compute and training; only generate samples using the base model "
            "and any existing checkpoints in <output_dir>."
        ),
    )
    parser.add_argument(
        "--cache_latents_only",
        action="store_true",
        default=False,
        help=(
            "Run only the pre-computation pass to cache VAE latents and text embeddings, "
            "then exit (skip training and sample generation)."
        ),
    )
    args = parser.parse_args()

    # Resolve cache_dir default
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, "cache")

    # Resolve val_cache_dir default
    if args.val_cache_dir is None:
        args.val_cache_dir = os.path.join(args.output_dir, "val_cache")

    assert args.resolution % 16 == 0, "--resolution must be divisible by 16"

    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FluxRawDataset(Dataset):
    """
    Loads raw (image, caption) pairs for Pass 1 pre-computation.
    Each image is resized & centre-cropped to `resolution`, then normalised
    to [-1, 1] as expected by the FLUX VAE.
    """

    EXTS = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(self, data_dir: str, resolution: int):
        self.data_dir = data_dir
        self.resolution = resolution

        # Collect base names that have both an image and a .txt caption
        entries = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(self.EXTS):
                base = os.path.splitext(fname)[0]
                txt = os.path.join(data_dir, base + ".txt")
                if os.path.exists(txt):
                    entries.append(base)
                else:
                    print(f"  [warn] No caption for {fname}, skipping")
        self.entries = entries

        self.transform = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),   # -> [-1, 1]
        ])
        print(f"FluxRawDataset: {len(self.entries)} image/caption pairs in {data_dir}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        base = self.entries[idx]

        # Find image file
        img_path = None
        for ext in self.EXTS:
            p = os.path.join(self.data_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break

        image = self.transform(Image.open(img_path).convert("RGB"))

        with open(os.path.join(self.data_dir, base + ".txt"), "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {"image": image, "caption": caption, "id": base}


class FluxCachedDataset(Dataset):
    """Loads pre-computed .pt cache files for Pass 2 training.

    When preload=True (default), all .pt files are loaded into RAM at init
    time so that training batches are served from memory with zero disk I/O.
    On the GB10's 128 GB unified memory this is strongly recommended.
    Set preload=False to fall back to per-sample disk reads (the original
    behaviour), which is useful when the cache is too large to fit in RAM.
    """

    def __init__(self, cache_dir: str, preload: bool = True):
        self.cache_dir = cache_dir
        files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        if len(files) == 0:
            raise RuntimeError(
                f"No .pt cache files found in '{cache_dir}'. "
                "Run Pass 1 (pre-computation) first."
            )
        print(f"FluxCachedDataset: {len(files)} cached samples from {cache_dir}")

        if preload:
            print("  Pre-loading all cache files into RAM ...")
            self.samples = [
                torch.load(os.path.join(cache_dir, f), weights_only=True)
                for f in tqdm(files, desc="Loading cache", unit="file")
            ]
            self.files = None
            print(f"  Done. {len(self.samples)} samples in RAM.")
        else:
            self.samples = None
            self.files = files

    def __len__(self):
        return len(self.samples) if self.samples is not None else len(self.files)

    def __getitem__(self, idx):
        if self.samples is not None:
            return self.samples[idx]
        return torch.load(
            os.path.join(self.cache_dir, self.files[idx]),
            weights_only=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLUX UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Reshape (B, C, H, W) -> (B, H'*W', C*p^2) for the FLUX transformer.
    FLUX uses 2x2 patches over the latent grid.
    """
    B, C, H, W = latents.shape
    pH, pW = H // patch_size, W // patch_size
    x = latents.reshape(B, C, pH, patch_size, pW, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)                          # (B, pH, pW, C, p, p)
    x = x.reshape(B, pH * pW, C * patch_size * patch_size)
    return x


def prepare_latent_image_ids(
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build row/col positional IDs for image patches.
    latent_h/latent_w are the latent spatial dims (pixel // VAE stride == 8).
    FLUX patch_size=2, so the grid is (latent_h//2, latent_w//2).
    Returns: (grid_h * grid_w, 3)
    """
    gh, gw = latent_h // 2, latent_w // 2
    ids = torch.zeros(gh, gw, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(gh, device=device, dtype=dtype).unsqueeze(1)
    ids[..., 2] = torch.arange(gw, device=device, dtype=dtype).unsqueeze(0)
    return ids.reshape(-1, 3)


def prepare_txt_ids(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Text token positional IDs: (seq_len, 3) -- all zeros for FLUX."""
    return torch.zeros(seq_len, 3, device=device, dtype=dtype)


def vae_encode(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images with the VAE and apply FLUX normalisation.
    images: (B, C, H, W) in [-1, 1]
    returns: normalised latents (B, 16, H/8, W/8)
    """
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def encode_caption(
    clip_tokenizer: CLIPTokenizer,
    clip_model: CLIPTextModel,
    t5_tokenizer: T5TokenizerFast,
    t5_model: T5EncoderModel,
    caption: str,
    max_t5_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Encode a single caption string.
    Returns:
      pooled  (1, 768)       -- CLIP pooled text embedding
      t5_seq  (1, L, 4096)   -- T5 sequence embedding
    """
    # CLIP pooled embedding
    clip_tok = clip_tokenizer(
        caption,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        pooled = clip_model(**clip_tok).pooler_output.to(dtype)   # (1, 768)

    # T5 sequence embedding
    t5_tok = t5_tokenizer(
        caption,
        padding="max_length",
        max_length=max_t5_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        t5_seq = t5_model(**t5_tok).last_hidden_state.to(dtype)   # (1, L, 4096)

    return pooled, t5_seq


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASS 1 -- PRE-COMPUTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def precompute_latents_and_embeddings(
    args,
    device: torch.device,
    dtype: torch.dtype,
    data_dir: str = None,
    cache_dir: str = None,
):
    """
    Pre-compute VAE latents and text embeddings for all images in data_dir.
    data_dir  : override for args.data_dir  (used for val set)
    cache_dir : override for args.cache_dir (used for val set)
    """
    data_dir  = data_dir  or args.data_dir
    cache_dir = cache_dir or args.cache_dir

    print("\n" + "=" * 60)
    print(f"Pass 1: Pre-computing VAE latents + text embeddings")
    print(f"  data  : {data_dir}")
    print(f"  cache : {cache_dir}")
    print("=" * 60)

    dataset = FluxRawDataset(data_dir, args.resolution)
    os.makedirs(cache_dir, exist_ok=True)

    if len(dataset) == 0:
        print(f"  [warn] No image/caption pairs found in '{data_dir}' -- nothing to cache.")
        return

    todo = [
        item_id for item_id in dataset.entries
        if not os.path.exists(os.path.join(cache_dir, f"{item_id}.pt"))
    ]
    if not todo:
        print("All items already cached -- skipping.")
        return
    print(f"{len(todo)} / {len(dataset)} items need caching.")

    todo_set = set(todo)

    # Load encoders
    print("\nLoading VAE ...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=dtype,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    print("Loading CLIP text encoder ...")
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer",
    )
    clip_model = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device)
    clip_model.requires_grad_(False)
    clip_model.eval()

    print("Loading T5 text encoder ...")
    t5_tokenizer = T5TokenizerFast.from_pretrained(
        args.model_id, subfolder="tokenizer_2",
    )
    t5_model = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2", torch_dtype=dtype,
    ).to(device)
    t5_model.requires_grad_(False)
    t5_model.eval()

    print()
    for item in tqdm(dataset, desc="Encoding", unit="img"):
        if item["id"] not in todo_set:
            continue

        cache_file = os.path.join(cache_dir, f"{item['id']}.pt")

        # VAE encode
        image = item["image"].unsqueeze(0).to(device, dtype=dtype)   # (1,C,H,W)
        latents = vae_encode(vae, image)                              # (1,16,H/8,W/8)

        # Text encode
        pooled, t5_seq = encode_caption(
            clip_tokenizer, clip_model,
            t5_tokenizer, t5_model,
            item["caption"],
            args.max_t5_length,
            device, dtype,
        )

        torch.save(
            {
                "latents":              latents.squeeze(0).cpu(),    # (16, H/8, W/8)
                "pooled_prompt_embeds": pooled.squeeze(0).cpu(),     # (768,)
                "prompt_embeds":        t5_seq.squeeze(0).cpu(),     # (L, 4096)
            },
            cache_file,
        )

    # Purge encoders to free VRAM before training/next pass
    del vae, clip_model, t5_model
    torch.cuda.empty_cache()
    print("\nEncoders purged from VRAM -- ready for training.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAMPLE GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_sample(
    args,
    device: torch.device,
    dtype: torch.dtype,
    ckpt_path: str = None,
    name: str = "base_model",
    pipe: "FluxPipeline" = None,
):
    """
    Generate a single comparison image using args.sample_prompt.

    If `pipe` is provided it is reused (no disk reload); the caller is
    responsible for loading/unloading LoRA adapters around this call.
    If `pipe` is None a fresh pipeline is loaded and destroyed afterward
    (original behaviour, but slow — prefer passing a reused pipe).

    If ckpt_path is given and pipe is None, the LoRA is loaded via PEFT,
    fused into the base weights in-place, then the whole pipe is freed.
    Saves to <output_dir>/samples/<name>.png.
    """
    if not args.sample_prompt:
        return

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    save_path = os.path.join(samples_dir, f"{name}.png")

    print(f"\n  [sample] Generating '{name}' ...")

    owns_pipe = pipe is None
    if owns_pipe:
        pipe = FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
        ).to(device)

        if ckpt_path is not None:
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer, ckpt_path,
            ).merge_and_unload()

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        image = pipe(
            prompt=args.sample_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.sample_steps,
            generator=torch.Generator(device).manual_seed(args.seed),
            width=args.resolution,
            height=args.resolution,
        ).images[0]

    image.save(save_path)
    print(f"  [sample] Saved -> {save_path}")

    if owns_pipe:
        del pipe
        torch.cuda.empty_cache()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASS 2 -- TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(args, device: torch.device, dtype: torch.dtype, val_loader=None):
    print("\n" + "=" * 60)
    print("Pass 2: Training FLUX LoRA")
    print("=" * 60)

    torch.manual_seed(args.seed)

    # --- Dataset & DataLoader ---
    dataset = FluxCachedDataset(args.cache_dir, preload=args.preload_cache)

    # When all data is in RAM, worker processes only add pickling overhead.
    # Determine prefetch_factor according to num_workers with sensible defaults.
    # Priority: explicit CLI `--prefetch_factor` > auto rules below.
    if args.preload_cache:
        # Data already in RAM -- no I/O workers needed.
        num_workers   = 0
        prefetch_factor = None
        pin_memory    = False
    else:
        num_workers = args.num_workers
        pin_memory  = True
        if num_workers == 0:
            prefetch_factor = None
        elif args.prefetch_factor is not None:
            prefetch_factor = args.prefetch_factor
        elif num_workers <= 2:
            prefetch_factor = 4
        elif num_workers <= 8:
            prefetch_factor = 2
        else:
            prefetch_factor = 1

    print(f"\nWorkers: {num_workers}  |  Prefetch factor: {prefetch_factor}  |  Pin memory: {pin_memory}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )

    # --- Load transformer (frozen base) ---
    print("\nLoading FLUX transformer ...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=dtype,
    )
    transformer.requires_grad_(False)

    # --- Wrap with LoRA ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            # Self-attention projections (double-stream blocks)
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            # Cross-attention additions (joint attention in FLUX double-stream)
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
            "attn.to_add_out",
            # Feed-forward layers in double-stream blocks
            # (critical for style/content learning — omitting these severely
            # limits what the LoRA can express)
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
            # Single-stream block projections
            "proj_mlp", "proj_out",
        ],
        bias="none",
        lora_dropout=0.05,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.to(device)
    transformer.print_trainable_parameters()

    # --- Optimizer & cosine LR scheduler ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    # total_steps must count *optimizer* steps, not DataLoader batches.
    # grad_accum_steps batches share one optimizer step; the last partial
    # accumulation window is also an optimizer step (see is_last_batch logic).
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.1,
    )

    # --- Infer latent spatial dimensions from first sample ---
    sample0 = dataset[0]
    _, lH, lW = sample0["latents"].shape     # e.g. (16, 64, 64) for 512px
    t5_seq_len = sample0["prompt_embeds"].shape[0]

    # Pre-build positional ID tensors (constant for all batches at fixed res)
    img_ids = prepare_latent_image_ids(lH, lW, device, dtype)   # (H'*W', 3)
    txt_ids = prepare_txt_ids(t5_seq_len, device, dtype)         # (L, 3)
    guidance = torch.full(
        (args.batch_size,), args.guidance_scale, device=device, dtype=dtype,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training loop ---
    global_step = 0
    print(f"  Effective batch size: {args.batch_size * args.grad_accum_steps} "
          f"({args.batch_size} x {args.grad_accum_steps} grad accum steps)")
    print()
    # Track epoch timings to estimate remaining time
    epoch_times = []
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        transformer.train()
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch")
        epoch_start = time.perf_counter()
        for local_step, batch in enumerate(pbar):
            latents = batch["latents"].to(device, dtype=dtype)              # (B, 16, H, W)
            pooled  = batch["pooled_prompt_embeds"].to(device, dtype=dtype) # (B, 768)
            t5_seq  = batch["prompt_embeds"].to(device, dtype=dtype)        # (B, L, 4096)
            B = latents.shape[0]

            # --- Flow matching ---
            # x_t = (1-t)*x_0 + t*eps   vel_target = eps - x_0
            t_batch = torch.rand(B, device=device, dtype=dtype)
            noise   = torch.randn_like(latents)
            t_4d    = t_batch.reshape(B, 1, 1, 1)

            noisy_latents   = (1.0 - t_4d) * latents + t_4d * noise
            velocity_target = noise - latents

            # --- Pack into FLUX sequence format ---
            packed_noisy  = pack_latents(noisy_latents)    # (B, N, C*4)
            packed_target = pack_latents(velocity_target)  # (B, N, C*4)

            b_guidance = guidance[:B]

            # --- Forward pass (mixed precision) ---
            # img_ids and txt_ids are passed as 2D tensors (N, 3) -- no batch dim
            with torch.autocast("cuda", dtype=dtype):
                pred = transformer(
                    hidden_states=packed_noisy,
                    timestep=t_batch,
                    guidance=b_guidance,
                    pooled_projections=pooled,
                    encoder_hidden_states=t5_seq,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]   # (B, N, C*4)

                # --- Flow-matching MSE loss (scaled for accumulation) ---
                loss = F.mse_loss(pred.float(), packed_target.float())

            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            epoch_loss += loss.item()

            is_last_batch = (local_step + 1) == len(loader)
            if (local_step + 1) % args.grad_accum_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                # --- Periodic checkpoint ---
                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    transformer.save_pretrained(ckpt_path)
                    print(f"\n  [step {global_step}] Checkpoint -> {ckpt_path}")

                    # --- Validation loss ---
                    if val_loader is not None:
                        transformer.eval()
                        val_loss_total = 0.0
                        # Fixed-seed generator so val loss is comparable across checkpoints
                        val_rng = torch.Generator(device=device)
                        val_rng.manual_seed(args.seed)
                        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                            for val_batch in val_loader:
                                vl = val_batch["latents"].to(device, dtype=dtype)
                                vp = val_batch["pooled_prompt_embeds"].to(device, dtype=dtype)
                                vt = val_batch["prompt_embeds"].to(device, dtype=dtype)
                                vB = vl.shape[0]

                                v_t    = torch.rand(vB, device=device, dtype=dtype, generator=val_rng)
                                v_noise = torch.randn(vl.shape, device=device, dtype=dtype, generator=val_rng)
                                v_t4d  = v_t.reshape(vB, 1, 1, 1)
                                v_noisy  = (1.0 - v_t4d) * vl + v_t4d * v_noise
                                v_target = v_noise - vl

                                v_pred = transformer(
                                    hidden_states=pack_latents(v_noisy),
                                    timestep=v_t,
                                    guidance=guidance[:vB],
                                    pooled_projections=vp,
                                    encoder_hidden_states=vt,
                                    txt_ids=txt_ids,
                                    img_ids=img_ids,
                                    return_dict=False,
                                )[0]

                                val_loss_total += F.mse_loss(
                                    v_pred.float(), pack_latents(v_target).float()
                                ).item()

                        val_loss_avg = val_loss_total / len(val_loader)
                        print(f"  [step {global_step}] Val loss: {val_loss_avg:.4f}")
                        transformer.train()

        avg_loss = epoch_loss / len(loader)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.num_epochs - (epoch + 1)
        est_remaining = avg_epoch * remaining_epochs

        def fmt(s):
            return str(timedelta(seconds=int(round(s))))

        print(f"  Epoch {epoch + 1}/{args.num_epochs} -- avg loss: {avg_loss:.4f}")
        print(
            f"  Epoch time: {fmt(epoch_time)}  |  Avg/epoch: {fmt(avg_epoch)}  |  "
            f"Estimated remaining: {fmt(est_remaining)} ({remaining_epochs} epochs)"
        )

    # --- Save final LoRA adapter ---
    final_path = os.path.join(args.output_dir, "flux-lora-final")
    transformer.save_pretrained(final_path)
    print(f"\n{'=' * 60}")
    print(f"Training complete.  LoRA weights saved to {final_path}")
    print(f"{'=' * 60}\n")

    # Free transformer VRAM before post-training sampling
    del transformer
    torch.cuda.empty_cache()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    args = parse_args()

    # Log all parsed CLI arguments for debugging/record-keeping
    print("Parsed CLI args:")
    try:
        print(json.dumps(vars(args), indent=2))
    except Exception:
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE  = {"float16": torch.float16,
              "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]

    print(f"\nDevice     : {DEVICE}")
    print(f"Dtype      : {DTYPE} ({args.dtype})")
    print(f"Model      : {args.model_id}")
    print(f"Data       : {args.data_dir}")
    print(f"Output     : {args.output_dir}")
    print(f"Cache      : {args.cache_dir}")
    print(f"Resolution : {args.resolution}px")
    print(f"LoRA rank  : {args.lora_rank}  alpha: {args.lora_alpha}")
    print(f"Epochs     : {args.num_epochs}  batch: {args.batch_size}  lr: {args.learning_rate}")

    # Decide whether to run precompute/train or only generate samples
    if args.generate_samples_only:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] generate_samples_only flag set - skipping precompute and training.")
    elif args.cache_latents_only:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] cache_latents_only flag set - running precompute only.")
        precompute_latents_and_embeddings(args, DEVICE, DTYPE)
        if args.val_dir:
            precompute_latents_and_embeddings(
                args, DEVICE, DTYPE,
                data_dir=args.val_dir, cache_dir=args.val_cache_dir,
            )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed precompute_latents_and_embeddings...")
        sys.exit(0)
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started training...")
        precompute_latents_and_embeddings(args, DEVICE, DTYPE)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed precompute_latents_and_embeddings...")

        # Pre-compute validation cache (if a val set is provided)
        val_loader = None
        if args.val_dir:
            precompute_latents_and_embeddings(
                args, DEVICE, DTYPE,
                data_dir=args.val_dir, cache_dir=args.val_cache_dir,
            )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed val precompute...")
            val_pt_files = [
                f for f in os.listdir(args.val_cache_dir)
                if f.endswith(".pt")
            ] if os.path.isdir(args.val_cache_dir) else []
            if val_pt_files:
                val_dataset = FluxCachedDataset(args.val_cache_dir, preload=args.preload_cache)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    prefetch_factor=None,
                )
                print(f"  Validation set: {len(val_dataset)} samples, {len(val_loader)} batches")
            else:
                print(
                    f"  [warn] --val_dir set but no cached samples found in "
                    f"'{args.val_cache_dir}'. "
                    "Check that val images have matching .txt caption files. "
                    "Validation loss will be skipped."
                )

        train(args, DEVICE, DTYPE, val_loader=val_loader)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed training...")

    # Generate samples for base model, each checkpoint, and the final LoRA.
    # Load the pipeline ONCE and swap LoRA adapters between calls to avoid
    # reloading ~30 GB of weights from disk for every single sample.
    if args.sample_prompt and not args.cache_latents_only:
        print("\nLoading pipeline for sample generation (single load for all samples) ...")
        sample_pipe = FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=DTYPE,
        ).to(DEVICE)

        # Base model (no LoRA)
        generate_sample(args, DEVICE, DTYPE, ckpt_path=None, name="base_model", pipe=sample_pipe)

        # Keep a reference to the bare transformer so we can restore it after
        # each PEFT adapter load (PEFT format, not diffusers LoRA format).
        #
        # IMPORTANT: merge_and_unload() modifies the base model's weights IN-PLACE
        # and returns it. We must deep-copy the base transformer before wrapping it
        # in PeftModel each time, otherwise every subsequent checkpoint sample starts
        # from an already-modified base and accumulates all previous LoRA deltas
        # (causing progressively worse / pure-noise outputs).
        base_transformer = sample_pipe.transformer

        checkpoints = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        for ckpt_name in checkpoints:
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            fresh_transformer = copy.deepcopy(base_transformer)
            sample_pipe.transformer = PeftModel.from_pretrained(
                fresh_transformer, ckpt_path,
            ).merge_and_unload()
            generate_sample(args, DEVICE, DTYPE, ckpt_path=None, name=ckpt_name, pipe=sample_pipe)
            # Restore the untouched base (fresh_transformer absorbed the merge)
            sample_pipe.transformer = base_transformer

        final_path = os.path.join(args.output_dir, "flux-lora-final")
        if os.path.isdir(final_path):
            fresh_transformer = copy.deepcopy(base_transformer)
            sample_pipe.transformer = PeftModel.from_pretrained(
                fresh_transformer, final_path,
            ).merge_and_unload()
            generate_sample(args, DEVICE, DTYPE, ckpt_path=None, name="flux-lora-final", pipe=sample_pipe)
            sample_pipe.transformer = base_transformer

        del sample_pipe
        torch.cuda.empty_cache()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed sample generation...")