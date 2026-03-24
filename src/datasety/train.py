"""LoRA fine-tuning for image generation models (FLUX Klein, SDXL, Qwen)."""

import sys
from pathlib import Path

# ── Dataset ───────────────────────────────────────────────────────────────────


def _build_dataset(input_dir: str, image_size: int = 512):
    """Build a simple paired image-caption dataset from a directory.

    Expects the directory to contain image files alongside same-name .txt
    caption files (e.g. ``photo.jpg`` + ``photo.txt``).

    Returns a list of ``{"image": PIL.Image, "caption": str}`` dicts.
    """
    from PIL import Image

    from datasety.common import get_image_files

    image_files = get_image_files(Path(input_dir), formats=["jpg", "jpeg", "png", "webp"])
    if not image_files:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)

    samples = []
    missing_captions = 0
    for img_path in image_files:
        caption_path = img_path.with_suffix(".txt")
        caption = ""
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            missing_captions += 1

        try:
            img = Image.open(img_path).convert("RGB")
            # Center-crop to square, then resize
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((image_size, image_size), Image.LANCZOS)
            samples.append({"image": img, "caption": caption, "path": str(img_path)})
        except Exception as e:
            print(f"Warning: skipping {img_path.name}: {e}")

    if missing_captions:
        print(
            f"Warning: {missing_captions}/{len(image_files)} images have no .txt caption "
            f"(empty string will be used)"
        )

    print(f"Dataset: {len(samples)} images loaded from {input_dir}")
    return samples


# ── Save LoRA ─────────────────────────────────────────────────────────────────


def _save_lora(model, output_path: str, family: str):
    """Save LoRA adapter weights in diffusers-compatible safetensors format.

    For FLUX models the transformer LoRA keys are saved under the ``transformer.``
    prefix so that ``pipeline.load_lora_weights()`` can load them directly.
    PEFT's ``base_model.model.`` wrapper is stripped as diffusers does not use it.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Error: safetensors is required. Run: pip install safetensors")
        sys.exit(1)

    from peft import get_peft_model_state_dict

    state_dict = get_peft_model_state_dict(model)

    # Remap PEFT keys to diffusers format:
    # "base_model.model.X" -> "<component>.X"
    # For FLUX: component = "transformer"; for SDXL: component = "unet"
    component = "unet" if family == "sdxl" else "transformer"
    remapped = {}
    for k, v in state_dict.items():
        # Strip PEFT wrapper prefix if present
        stripped = k
        if stripped.startswith("base_model.model."):
            stripped = stripped[len("base_model.model."):]
        remapped[f"{component}.{stripped}"] = v.contiguous().float()

    save_file(remapped, output_path)
    print(f"LoRA saved → {output_path}")


# ── Training utilities ────────────────────────────────────────────────────────


def _sample_timestep(device, dtype, timestep_type="sigmoid"):
    """Sample a random timestep t ∈ (0,1) for flow-matching training.

    ``sigmoid`` (default, per ai-toolkit): biases toward middle timesteps
    where perceptual learning is densest, accelerating LoRA convergence.
    ``lognorm``: logit-normal distribution with scale 0.5.
    ``linear``: uniform — matches original behaviour.
    """
    import torch

    if timestep_type == "sigmoid":
        return torch.sigmoid(torch.randn(1, device=device, dtype=dtype))
    elif timestep_type == "lognorm":
        u = torch.randn(1, device=device, dtype=dtype)
        return torch.sigmoid(u * 0.5).clamp(0.001, 0.999)
    else:  # linear / uniform
        return torch.rand(1, device=device, dtype=dtype)


def _maybe_dropout_caption(caption: str, rate: float) -> str:
    """Randomly replace caption with empty string (unconditional training)."""
    import random

    if rate > 0 and random.random() < rate:
        return ""
    return caption


def _build_optimizer(params, lr: float, optimizer_type: str = "adamw"):
    """Build AdamW or 8-bit AdamW optimizer (falls back if bitsandbytes absent)."""
    import torch

    if optimizer_type == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-4, eps=1e-6)
        except ImportError:
            print(
                "Warning: bitsandbytes not found, falling back to AdamW. "
                "Run: pip install bitsandbytes"
            )
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def _build_lr_scheduler(
    optimizer, total_steps: int, sched_type: str = "constant", warmup_steps: int = 0
):
    """Build a LambdaLR scheduler with optional linear warmup."""
    import math

    from torch.optim.lr_scheduler import LambdaLR

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def linear_with_warmup(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, 1.0 - float(step - warmup_steps) / max(1, total_steps - warmup_steps))

    def constant_with_warmup(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0

    fn = {"cosine": cosine_with_warmup, "linear": linear_with_warmup}.get(
        sched_type, constant_with_warmup
    )
    return LambdaLR(optimizer, fn)


def _min_snr_loss(loss_unreduced, timesteps, alphas_cumprod, gamma):
    """Apply min-SNR-γ loss weighting (Hang et al. 2023) to unreduced SDXL loss.

    Clamps per-timestep SNR to ``gamma``, reducing over-weighting of easy
    (high-noise) timesteps and stabilising DDPM training.
    """
    if gamma is None or gamma <= 0:
        return loss_unreduced.mean()
    alpha_t = alphas_cumprod[timesteps].to(loss_unreduced.device).float()
    snr = alpha_t / (1.0 - alpha_t + 1e-8)
    weights = (snr.clamp(max=gamma) / snr).view(-1, *([1] * (loss_unreduced.ndim - 1)))
    return (loss_unreduced * weights).mean()


# ── FLUX Klein LoRA training ──────────────────────────────────────────────────


def _train_flux_klein(args):
    """Train a LoRA adapter on Flux2KleinPipeline (flow-matching, text-to-image)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("Error: PyTorch not installed.")
        sys.exit(1)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Error: peft is required for LoRA training. Run: pip install peft")
        sys.exit(1)

    from datasety.common import resolve_device
    from datasety.synthetic import _MODEL_VRAM_GB, _detect_model_family

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    print(f"Loading pipeline: {args.model}")
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)

    # VRAM check for auto-offload
    if device == "cuda":
        torch.cuda.empty_cache()
        family = _detect_model_family(args.model)
        free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        needed_gb = _MODEL_VRAM_GB.get(family, 16)
        if free_gb >= needed_gb:
            pipe.to(device)
        elif total_gb >= needed_gb:
            pipe.enable_model_cpu_offload()
            print(f"Model CPU offload enabled ({free_gb:.1f} GB free, need ~{needed_gb} GB)")
        else:
            pipe.enable_sequential_cpu_offload()
            print(f"Sequential CPU offload enabled ({free_gb:.1f} GB free, need ~{needed_gb} GB)")
    else:
        pipe.to(device)

    # Freeze everything except the transformer.
    # The 9B model uses a Qwen3 text encoder (text_encoder_2 attribute); freeze
    # both text encoders when present.
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)

    # Discover which projection module names exist in this model's transformer.
    # FLUX.2-klein 4B uses to_qkv_mlp_proj; some 9B checkpoints expose
    # to_out.0 instead of to_add_out.  We probe once and build the list.
    _probe = pipe.transformer
    _candidate_extra = []
    for name, _ in _probe.named_modules():
        leaf = name.split(".")[-1]
        if leaf in ("to_add_out", "to_qkv_mlp_proj") and leaf not in _candidate_extra:
            _candidate_extra.append(leaf)
    target_modules = ["to_q", "to_k", "to_v"] + _candidate_extra
    if not _candidate_extra:
        # Fallback: standard attention output projection
        target_modules.append("to_out.0")

    # Apply LoRA to transformer attention layers.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Gradient checkpointing reduces VRAM at a small compute cost
    if getattr(args, "gradient_checkpointing", False):
        pipe.transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Resume from checkpoint
    if args.resume:
        from safetensors.torch import load_file

        print(f"Resuming from checkpoint: {args.resume}")
        state = load_file(args.resume)
        # Strip component prefix (diffusers format: "transformer.X" -> "X")
        # Also handle old lora_family prefix for backward compatibility
        cleaned = {}
        for k, v in state.items():
            for prefix in ("transformer.", "lora_flux2-klein.", "lora_flux.", "base_model.model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
        pipe.transformer.load_state_dict(cleaned, strict=False)

    # Dataset
    samples = _build_dataset(args.input, image_size=args.image_size)
    if len(samples) == 0:
        print("Error: No samples to train on.")
        sys.exit(1)

    # Validation split
    val_samples = []
    if args.validation_split > 0:
        import random as _rnd

        _rnd.seed(args.seed if args.seed is not None else 42)
        _rnd.shuffle(samples)
        n_val = max(1, int(len(samples) * args.validation_split))
        val_samples = samples[:n_val]
        samples = samples[n_val:]
        print(f"Validation split: {len(val_samples)} val, {len(samples)} train")

    # Optimizer + LR scheduler
    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer_type = getattr(args, "optimizer", "adamw")
    optimizer = _build_optimizer(trainable, args.lr, optimizer_type)
    sched_type = getattr(args, "lr_scheduler", "constant")
    warmup_steps = getattr(args, "lr_warmup_steps", 0)
    scheduler = _build_lr_scheduler(optimizer, args.steps, sched_type, warmup_steps)
    accum_steps = max(1, getattr(args, "gradient_accumulation_steps", 1))
    caption_dropout = getattr(args, "caption_dropout", 0.0)
    timestep_type = getattr(args, "timestep_type", "sigmoid")

    exec_device = pipe._execution_device if hasattr(pipe, "_execution_device") else device

    print("\nStarting FLUX Klein LoRA training")
    print(f"  steps: {args.steps}, lr: {args.lr}, rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  optimizer: {optimizer_type}, scheduler: {sched_type}, warmup: {warmup_steps}")
    print(f"  timestep_type: {timestep_type}, caption_dropout: {caption_dropout}, "
          f"grad_accum: {accum_steps}")
    print(f"  dataset: {len(samples)} images, image_size: {args.image_size}")
    print("-" * 60)

    pipe.transformer.train()
    seed = args.seed if args.seed is not None else 42
    generator = torch.Generator(device=exec_device).manual_seed(seed)
    optimizer.zero_grad()

    for step in range(args.steps):
        # Pick a sample (cycle through dataset)
        sample = samples[step % len(samples)]
        caption = _maybe_dropout_caption(sample["caption"], caption_dropout)

        # Encode text (no_grad).
        # encode_prompt returns (prompt_embeds, text_ids) for all Flux2Klein variants.
        with torch.no_grad():
            enc_out = pipe.encode_prompt(
                prompt=caption,
                device=exec_device,
                num_images_per_prompt=1,
            )
            # Unpack; some variants return a longer tuple — take first two
            prompt_embeds, text_ids = enc_out[0], enc_out[1]

        # Encode image to latents (no_grad on VAE)
        img_tensor = _pil_to_tensor(sample["image"], dtype=dtype, device=exec_device)
        with torch.no_grad():
            # Use pipeline helper: returns patchified latents [1, C, H/2, W/2]
            image_latents = pipe._encode_vae_image(img_tensor, generator=generator)
            # Pack: [B, C, H, W] → [B, H*W, C]
            packed_latents = pipe._pack_latents(image_latents)  # [1, seq_len, C]
            # Prepare positional IDs for latents
            latent_ids = pipe._prepare_latent_ids(image_latents).to(exec_device)

        # Flow matching: x_t = (1-t)*x_0 + t*noise
        # sigmoid timestep sampling biases toward middle timesteps (ai-toolkit default)
        t_scalar = _sample_timestep(exec_device, dtype, timestep_type)
        noise = torch.randn_like(packed_latents)
        noisy_latents = (1.0 - t_scalar) * packed_latents + t_scalar * noise

        # Timestep for transformer (scaled 0→1000)
        timestep = (t_scalar * 1000.0).expand(1).to(dtype=dtype)

        # Forward through transformer (with grad)
        noise_pred = pipe.transformer(
            hidden_states=noisy_latents.to(dtype),
            timestep=timestep / 1000.0,
            guidance=None,
            encoder_hidden_states=prompt_embeds.to(dtype),
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]

        # Velocity target: v = noise - x_0 (flow matching)
        target = noise - packed_latents
        # Predict only for the denoising tokens (not conditioning tokens)
        seq_len = packed_latents.shape[1]
        loss = F.mse_loss(noise_pred[:, :seq_len], target.to(dtype)) / accum_steps
        loss.backward()

        is_update = (step + 1) % accum_steps == 0 or (step + 1) == args.steps
        if is_update:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            lr_now = scheduler.get_last_lr()[0]
            loss_val = loss.item() * accum_steps
            print(f"  step {step + 1:4d}/{args.steps}  loss={loss_val:.6f}  lr={lr_now:.2e}")

        if args.save_every and (step + 1) % args.save_every == 0:
            stem = Path(args.output).stem
            mid_path = str(Path(args.output).parent / f"{stem}_step{step+1}.safetensors")
            _save_lora(pipe.transformer, mid_path, "flux2-klein")

    # Save final LoRA
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"
    _save_lora(pipe.transformer, output_path, "flux2-klein")

    # Validation loss
    if val_samples:
        pipe.transformer.eval()
        val_losses = []
        with torch.no_grad():
            for vs in val_samples:
                enc_out = pipe.encode_prompt(
                    prompt=vs["caption"], device=exec_device, num_images_per_prompt=1,
                )
                prompt_embeds, text_ids = enc_out[0], enc_out[1]
                vt = _pil_to_tensor(vs["image"], dtype=dtype, device=exec_device)
                vl = pipe._encode_vae_image(vt, generator=generator)
                vp = pipe._pack_latents(vl)
                vid = pipe._prepare_latent_ids(vl).to(exec_device)
                t = torch.rand(1, device=exec_device, dtype=dtype)
                n = torch.randn_like(vp)
                noisy = (1.0 - t) * vp + t * n
                ts = (t * 1000.0).expand(1).to(dtype=dtype)
                pred = pipe.transformer(
                    hidden_states=noisy.to(dtype), timestep=ts / 1000.0,
                    guidance=None, encoder_hidden_states=prompt_embeds.to(dtype),
                    txt_ids=text_ids, img_ids=vid, return_dict=False,
                )[0]
                target = n - vp
                sl = vp.shape[1]
                val_losses.append(F.mse_loss(pred[:, :sl], target.to(dtype)).item())
        avg_val = sum(val_losses) / len(val_losses)
        print(f"\nValidation loss: {avg_val:.6f} (over {len(val_samples)} images)")

    print("\nTraining complete!")


def _pil_to_tensor(pil_image, dtype, device):
    """Convert PIL image to normalized float tensor [1, C, H, W] in [-1, 1]."""
    import numpy as np
    import torch

    arr = np.array(pil_image).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


# ── SDXL LoRA training ────────────────────────────────────────────────────────


def _train_sdxl(args):
    """Train a LoRA adapter on StableDiffusionXLPipeline (DDPM, text-to-image)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("Error: PyTorch not installed.")
        sys.exit(1)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Error: peft is required for LoRA training. Run: pip install peft")
        sys.exit(1)

    from datasety.common import resolve_device

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    print(f"Loading SDXL pipeline: {args.model}")
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=dtype)

    if device == "cuda":
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
        if free_gb >= 7:
            pipe.to(device)
        else:
            pipe.enable_model_cpu_offload()
            print(f"Model CPU offload enabled ({free_gb:.1f} GB free)")
    else:
        pipe.to(device)

    # Freeze VAE and text encoders
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    # Apply LoRA to UNet attention layers
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    # Gradient checkpointing reduces VRAM at a small compute cost
    if getattr(args, "gradient_checkpointing", False):
        pipe.unet.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Resume from checkpoint
    if args.resume:
        from safetensors.torch import load_file

        print(f"Resuming from checkpoint: {args.resume}")
        state = load_file(args.resume)
        cleaned = {}
        for k, v in state.items():
            if k.startswith("lora_sdxl."):
                k = k[len("lora_sdxl."):]
            cleaned[k] = v
        pipe.unet.load_state_dict(cleaned, strict=False)

    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    # Dataset
    samples = _build_dataset(args.input, image_size=args.image_size)

    trainable = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer_type = getattr(args, "optimizer", "adamw")
    optimizer = _build_optimizer(trainable, args.lr, optimizer_type)
    sched_type = getattr(args, "lr_scheduler", "constant")
    warmup_steps = getattr(args, "lr_warmup_steps", 0)
    scheduler = _build_lr_scheduler(optimizer, args.steps, sched_type, warmup_steps)
    accum_steps = max(1, getattr(args, "gradient_accumulation_steps", 1))
    caption_dropout = getattr(args, "caption_dropout", 0.0)
    noise_offset = getattr(args, "noise_offset", 0.0)
    min_snr_gamma = getattr(args, "min_snr_gamma", None)

    exec_device = device

    print("\nStarting SDXL LoRA training")
    print(f"  steps: {args.steps}, lr: {args.lr}, rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  optimizer: {optimizer_type}, scheduler: {sched_type}, warmup: {warmup_steps}")
    print(f"  caption_dropout: {caption_dropout}, grad_accum: {accum_steps}")
    print(f"  noise_offset: {noise_offset}, min_snr_gamma: {min_snr_gamma}")
    print(f"  dataset: {len(samples)} images, image_size: {args.image_size}")
    print("-" * 60)

    pipe.unet.train()
    optimizer.zero_grad()

    for step in range(args.steps):
        sample = samples[step % len(samples)]
        caption = _maybe_dropout_caption(sample["caption"], caption_dropout)

        # Encode image to latents
        img_tensor = _pil_to_tensor(sample["image"], dtype=dtype, device=exec_device)
        with torch.no_grad():
            latents = pipe.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        # Encode text with both SDXL text encoders
        with torch.no_grad():
            text_input_1 = pipe.tokenizer(
                [caption],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(exec_device)
            text_input_2 = pipe.tokenizer_2(
                [caption],
                padding="max_length",
                max_length=pipe.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(exec_device)

            enc1 = pipe.text_encoder(text_input_1, output_hidden_states=True)
            enc2 = pipe.text_encoder_2(text_input_2, output_hidden_states=True)
            # SDXL uses penultimate hidden states from both encoders
            text_embeds = torch.cat(
                [enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1
            )
            pooled_text_embeds = enc2[0]  # pooled output from text_encoder_2

        # Build add_time_ids (SDXL conditioning: original_size, crop_coords, target_size)
        add_time_ids = _get_sdxl_time_ids(
            args.image_size, args.image_size, exec_device, dtype
        )

        # Add noise (with optional offset for dark/bright image coverage)
        noise = torch.randn_like(latents)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=exec_device, dtype=dtype
            )
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=exec_device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # UNet forward
        model_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_text_embeds,
                "time_ids": add_time_ids,
            },
        ).sample

        # v-prediction vs noise prediction
        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # Compute loss — apply min-SNR-γ weighting if requested
        loss_unreduced = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = _min_snr_loss(
            loss_unreduced, timesteps, noise_scheduler.alphas_cumprod, min_snr_gamma
        ) / accum_steps
        loss.backward()

        is_update = (step + 1) % accum_steps == 0 or (step + 1) == args.steps
        if is_update:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            lr_now = scheduler.get_last_lr()[0]
            loss_val = loss.item() * accum_steps
            print(f"  step {step + 1:4d}/{args.steps}  loss={loss_val:.6f}  lr={lr_now:.2e}")

        if args.save_every and (step + 1) % args.save_every == 0:
            stem = Path(args.output).stem
            mid_path = str(Path(args.output).parent / f"{stem}_step{step+1}.safetensors")
            _save_lora(pipe.unet, mid_path, "sdxl")

    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"
    _save_lora(pipe.unet, output_path, "sdxl")
    print("\nTraining complete!")


def _get_sdxl_time_ids(height, width, device, dtype):
    """Build SDXL additional conditioning time_ids tensor."""
    import torch

    original_size = (height, width)
    crops_coords_top_left = (0, 0)
    target_size = (height, width)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
    return add_time_ids


# ── Qwen LoRA training ────────────────────────────────────────────────────────


def _qwen_pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack VAE latents into transformer sequence tokens.

    Mirrors ``QwenImageEditPlusPipeline._pack_latents``.  Accepts both 4-D
    (B, C, H, W) and 5-D (B, C, T, H, W) tensors — the temporal dimension is
    absorbed by the reshape when T == 1.
    """
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def _train_qwen(args):
    """Train a LoRA adapter on QwenImageEditPlusPipeline (flow-matching, image-editing).

    Uses the same image as both source and target (reconstruction training).
    This teaches the model identity/style without requiring paired before/after
    images.  Pass ``--image-size 512`` (default) or multiples of 16.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("Error: PyTorch not installed.")
        sys.exit(1)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Error: peft is required for LoRA training. Run: pip install peft")
        sys.exit(1)

    from datasety.common import resolve_device

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    # Image size must be divisible by vae_scale_factor * patch_size * 2 = 8*2*2=32
    # We round down to nearest 32 so latent packing works.
    image_size = (args.image_size // 32) * 32
    if image_size != args.image_size:
        print(f"Note: --image-size rounded to {image_size} (must be divisible by 32 for Qwen)")

    print(f"Loading QwenImageEditPlusPipeline: {args.model}")
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model, torch_dtype=dtype)

    if device == "cuda":
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
        needed_gb = 30  # ~30 GB for 7B transformer + 7B text encoder in BF16
        if free_gb >= needed_gb:
            pipe.to(device)
        else:
            pipe.enable_model_cpu_offload()
            print(f"Model CPU offload enabled ({free_gb:.1f} GB free, need ~{needed_gb} GB)")
    else:
        pipe.to(device)

    # Freeze VAE and text encoder — only train transformer via LoRA
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Discover LoRA-compatible attention module names from the actual transformer.
    # QwenImageTransformer2DModel uses both image-stream (to_q/k/v, to_out.0) and
    # text-stream (add_q/k/v_proj, to_add_out) attention projections.
    _candidates = ["to_q", "to_k", "to_v", "to_out.0",
                   "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    _found = set()
    for n, _ in pipe.transformer.named_modules():
        leaf = n.split(".")[-1]
        if leaf in _candidates:
            _found.add(leaf)
        # also capture "to_out.0" as a module path segment
        if n.endswith("to_out.0"):
            _found.add("to_out.0")
    target_modules = [m for m in _candidates if m in _found] or ["to_q", "to_k", "to_v", "to_out.0"]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Gradient checkpointing reduces VRAM at a small compute cost
    if getattr(args, "gradient_checkpointing", False):
        pipe.transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Resume from checkpoint
    if args.resume:
        from safetensors.torch import load_file

        print(f"Resuming from checkpoint: {args.resume}")
        state = load_file(args.resume)
        cleaned = {}
        for k, v in state.items():
            for prefix in ("transformer.", "diffusion_model.", "base_model.model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
        pipe.transformer.load_state_dict(cleaned, strict=False)

    # Dataset
    samples = _build_dataset(args.input, image_size=image_size)
    if not samples:
        print("Error: No samples to train on.")
        sys.exit(1)

    # Validation split
    val_samples = []
    if args.validation_split > 0:
        import random as _rnd

        _rnd.seed(args.seed if args.seed is not None else 42)
        _rnd.shuffle(samples)
        n_val = max(1, int(len(samples) * args.validation_split))
        val_samples = samples[:n_val]
        samples = samples[n_val:]
        print(f"Validation split: {len(val_samples)} val, {len(samples)} train")

    # Pre-compute VAE normalization constants (on CPU, move later)
    latent_channels = pipe.vae.config.z_dim  # 16
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, latent_channels, 1, 1, 1
    )
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(
        1, latent_channels, 1, 1, 1
    )

    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer_type = getattr(args, "optimizer", "adamw")
    optimizer = _build_optimizer(trainable, args.lr, optimizer_type)
    sched_type = getattr(args, "lr_scheduler", "constant")
    warmup_steps = getattr(args, "lr_warmup_steps", 0)
    scheduler = _build_lr_scheduler(optimizer, args.steps, sched_type, warmup_steps)
    accum_steps = max(1, getattr(args, "gradient_accumulation_steps", 1))
    caption_dropout = getattr(args, "caption_dropout", 0.0)
    timestep_type = getattr(args, "timestep_type", "sigmoid")

    exec_device = pipe._execution_device if hasattr(pipe, "_execution_device") else device

    print("\nStarting Qwen LoRA training (image-editing, flow-matching)")
    print(f"  steps: {args.steps}, lr: {args.lr}, rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  optimizer: {optimizer_type}, scheduler: {sched_type}, warmup: {warmup_steps}")
    print(f"  timestep_type: {timestep_type}, caption_dropout: {caption_dropout}, "
          f"grad_accum: {accum_steps}")
    print(f"  dataset: {len(samples)} images, image_size: {image_size}")
    print(f"  target modules: {target_modules}")
    print("-" * 60)

    pipe.transformer.train()
    seed = args.seed if args.seed is not None else 42
    generator = torch.Generator(device=exec_device).manual_seed(seed)

    lm = latents_mean.to(exec_device, dtype=dtype)
    ls = latents_std.to(exec_device, dtype=dtype)
    optimizer.zero_grad()

    for step in range(args.steps):
        sample = samples[step % len(samples)]
        pil_img = sample["image"]  # PIL.Image
        caption = _maybe_dropout_caption(sample["caption"], caption_dropout)

        # ── Text + visual conditioning (no grad) ──────────────────────────────
        with torch.no_grad():
            # encode_prompt for edit+ takes a list of PIL control images
            prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                caption,
                image=[pil_img],
                device=exec_device,
                num_images_per_prompt=1,
            )

        # ── VAE encoding of target (= source for reconstruction LoRA) ─────────
        img_tensor = _pil_to_tensor(pil_img, dtype=dtype, device=exec_device)  # (1,3,H,W)
        img_5d = img_tensor.unsqueeze(2)  # (1, 3, 1, H, W)  — Wan VAE needs T dim

        with torch.no_grad():
            raw = pipe.vae.encode(img_5d).latent_dist.sample(generator=generator)
            # raw: (1, z_dim, 1, H_lat, W_lat)
            target_latents = (raw - lm) / ls  # normalized

            # For edit+ training: control = same image (reconstruction objective)
            # This trains the model to recognize and preserve the source content.
            ctrl_latents = target_latents.clone()

            # Latent spatial dims (shape[3:] accounts for temporal dim)
            lat_H, lat_W = target_latents.shape[3], target_latents.shape[4]

            packed_target = _qwen_pack_latents(target_latents, 1, latent_channels, lat_H, lat_W)
            packed_ctrl = _qwen_pack_latents(ctrl_latents, 1, latent_channels, lat_H, lat_W)

        # img_shapes: list[list[(n, h2, w2)]] — one entry per image type per batch
        # h2/w2 = latent_H // patch_size (patch_size=2)
        h2, w2 = lat_H // 2, lat_W // 2
        img_shapes = [[(1, h2, w2), (1, h2, w2)]]  # [target_shape, ctrl_shape]

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )

        # ── Flow matching ──────────────────────────────────────────────────────
        # sigmoid timestep sampling biases toward middle timesteps (ai-toolkit default)
        t_scalar = _sample_timestep(exec_device, dtype, timestep_type)
        noise = torch.randn_like(packed_target)
        noisy_target = (1.0 - t_scalar) * packed_target + t_scalar * noise
        # Control tokens are NOT noised — they are always clean conditioning
        noisy_input = torch.cat([noisy_target, packed_ctrl], dim=1)
        timestep = (t_scalar * 1000.0).expand(1).to(dtype=dtype)

        # ── Transformer forward ────────────────────────────────────────────────
        noise_pred = pipe.transformer(
            hidden_states=noisy_input.to(dtype),
            timestep=timestep / 1000.0,
            guidance=None,
            encoder_hidden_states=prompt_embeds.to(dtype),
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # Slice to target token length only (first half)
        noise_pred = noise_pred[:, : packed_target.shape[1]]

        # Velocity target: v = noise - x0  (flow-matching convention)
        velocity_target = (noise - packed_target).to(dtype)
        loss = F.mse_loss(noise_pred, velocity_target) / accum_steps
        loss.backward()

        is_update = (step + 1) % accum_steps == 0 or (step + 1) == args.steps
        if is_update:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            lr_now = scheduler.get_last_lr()[0]
            loss_val = loss.item() * accum_steps
            print(f"  step {step + 1:4d}/{args.steps}  loss={loss_val:.6f}  lr={lr_now:.2e}")

        if args.save_every and (step + 1) % args.save_every == 0:
            stem = Path(args.output).stem
            mid_path = str(Path(args.output).parent / f"{stem}_step{step+1}.safetensors")
            _save_lora(pipe.transformer, mid_path, "qwen")

    # ── Save final LoRA ────────────────────────────────────────────────────────
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"
    _save_lora(pipe.transformer, output_path, "qwen")

    # ── Validation loss ────────────────────────────────────────────────────────
    if val_samples:
        pipe.transformer.eval()
        val_losses = []
        with torch.no_grad():
            for vs in val_samples:
                pe, pm = pipe.encode_prompt(
                    vs["caption"], image=[vs["image"]],
                    device=exec_device, num_images_per_prompt=1,
                )
                vt = _pil_to_tensor(vs["image"], dtype=dtype, device=exec_device)
                vraw = pipe.vae.encode(vt.unsqueeze(2)).latent_dist.sample(generator=generator)
                vl = (vraw - lm) / ls
                vlH, vlW = vl.shape[3], vl.shape[4]
                pvl = _qwen_pack_latents(vl, 1, latent_channels, vlH, vlW)
                pvctrl = pvl.clone()
                vn = torch.randn_like(pvl)
                vt_s = torch.rand(1, device=exec_device, dtype=dtype)
                vnoisy = (1.0 - vt_s) * pvl + vt_s * vn
                vinput = torch.cat([vnoisy, pvctrl], dim=1)
                vts = (vt_s * 1000.0).expand(1).to(dtype=dtype)
                vh2, vw2 = vlH // 2, vlW // 2
                vis = [[(1, vh2, vw2), (1, vh2, vw2)]]
                vtsl = pm.sum(dim=1).tolist() if pm is not None else None
                vpred = pipe.transformer(
                    hidden_states=vinput.to(dtype), timestep=vts / 1000.0,
                    guidance=None, encoder_hidden_states=pe.to(dtype),
                    encoder_hidden_states_mask=pm,
                    img_shapes=vis, txt_seq_lens=vtsl, return_dict=False,
                )[0][:, :pvl.shape[1]]
                val_losses.append(F.mse_loss(vpred, (vn - pvl).to(dtype)).item())
        avg_val = sum(val_losses) / len(val_losses)
        print(f"\nValidation loss: {avg_val:.6f} (over {len(val_samples)} images)")

    print("\nTraining complete!")


# ── Command dispatch ──────────────────────────────────────────────────────────


def cmd_train(args):
    """Execute the train command: LoRA fine-tuning."""
    # Warn if user passed a distilled (inference) model — training should use base models
    model_lower = args.model.lower()
    is_klein = "klein" in model_lower
    is_distilled = is_klein and "base" not in model_lower
    if is_distilled:
        print(
            f"Warning: '{args.model}' appears to be a step-distilled model. "
            f"Training on distilled models produces degraded results. "
            f"Use the corresponding base model instead, e.g. "
            f"'black-forest-labs/FLUX.2-klein-base-4B' or "
            f"'black-forest-labs/FLUX.2-klein-base-9B'."
        )

    # Detect family from model name if not explicitly set
    family = args.family
    if family is None:
        if is_klein or "flux.2" in model_lower or "flux2" in model_lower:
            family = "flux"
        elif "qwen" in model_lower:
            family = "qwen"
        elif "firered" in model_lower:
            family = "qwen"  # FireRed shares Qwen architecture
        else:
            family = "sdxl"
        print(f"Auto-detected model family: {family}")
    else:
        print(f"Model family: {family}")

    if family == "flux":
        _train_flux_klein(args)
    elif family == "sdxl":
        _train_sdxl(args)
    elif family == "qwen":
        _train_qwen(args)
    else:
        print(f"Error: Unknown family '{family}'. Choose from: flux, sdxl, qwen")
        sys.exit(1)


def register_parser(subparsers):
    """Register the train subcommand."""
    train_parser = subparsers.add_parser(
        "train",
        help="Train a LoRA adapter for image generation models (FLUX, SDXL, Qwen)",
    )
    train_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input dataset directory (images + .txt captions with matching filenames)",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        default="lora.safetensors",
        help="Output LoRA file path (default: lora.safetensors)",
    )
    train_parser.add_argument(
        "--model",
        "-m",
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help=(
            "Model HF repo ID for training. Use the BASE (undistilled) model "
            "for LoRA training — distilled models produce degraded results. "
            "(default: black-forest-labs/FLUX.2-klein-base-4B)"
        ),
    )
    train_parser.add_argument(
        "--family",
        choices=["flux", "sdxl", "qwen"],
        default=None,
        help="Model family — auto-detected from --model if not set",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    train_parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha (default: 16.0)",
    )
    train_parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate (default: 0.0)",
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Training image resolution (center-cropped to square, default: 512)",
    )
    train_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use (default: auto)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: only save at the end)",
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from a checkpoint .safetensors file",
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Hold out this fraction of images for validation (e.g., 0.1 = 10%%)",
    )
    train_parser.add_argument(
        "--timestep-type",
        choices=["sigmoid", "lognorm", "linear"],
        default="sigmoid",
        help=(
            "Timestep sampling distribution for flow-matching (flux/qwen). "
            "'sigmoid' (default, ai-toolkit): biases toward middle timesteps. "
            "'lognorm': logit-normal distribution. 'linear': uniform."
        ),
    )
    train_parser.add_argument(
        "--caption-dropout",
        type=float,
        default=0.05,
        help=(
            "Probability of replacing a caption with empty string "
            "(unconditional training). Default: 0.05"
        ),
    )
    train_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to reduce VRAM at a small speed cost",
    )
    train_parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw8bit"],
        default="adamw",
        help=(
            "Optimizer type. 'adamw8bit' halves optimizer memory "
            "(requires bitsandbytes). Default: adamw"
        ),
    )
    train_parser.add_argument(
        "--lr-scheduler",
        choices=["constant", "cosine", "linear"],
        default="constant",
        help="Learning rate schedule. Default: constant",
    )
    train_parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Number of linear warmup steps before reaching the target LR. Default: 0",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over N steps before an optimizer update "
            "(simulates larger batches). Default: 1"
        ),
    )
    train_parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=None,
        help=(
            "Min-SNR-γ loss weighting for SDXL (Hang et al. 2023). "
            "Recommended: 5.0. Default: disabled"
        ),
    )
    train_parser.add_argument(
        "--noise-offset",
        type=float,
        default=0.0,
        help=(
            "Noise offset for SDXL training — adds per-channel offset to improve "
            "dark/bright image coverage. Recommended: 0.05–0.1. Default: 0.0"
        ),
    )
    train_parser.set_defaults(func=cmd_train)
