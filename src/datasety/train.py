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
    """Save LoRA adapter weights to a .safetensors file."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Error: safetensors is required. Run: pip install safetensors")
        sys.exit(1)

    from peft import get_peft_model_state_dict

    state_dict = get_peft_model_state_dict(model)
    # Prefix keys with the family so users know the source model
    prefixed = {f"lora_{family}.{k}": v.contiguous().float() for k, v in state_dict.items()}
    save_file(prefixed, output_path)
    print(f"LoRA saved → {output_path}")


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

    # Freeze everything except the transformer
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Apply LoRA to transformer attention layers.
    # Note: to_out is a ModuleList in Flux2, so we target to_add_out and the
    # combined qkv projection instead.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_add_out", "to_qkv_mlp_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Resume from checkpoint
    if args.resume:
        from safetensors.torch import load_file

        print(f"Resuming from checkpoint: {args.resume}")
        state = load_file(args.resume)
        # Strip lora_ family prefix if present
        cleaned = {}
        for k, v in state.items():
            for prefix in ("lora_flux2-klein.", "lora_flux."):
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

    # Optimizer — only update LoRA parameters
    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    exec_device = pipe._execution_device if hasattr(pipe, "_execution_device") else device

    print("\nStarting FLUX Klein LoRA training")
    print(f"  steps: {args.steps}, lr: {args.lr}, rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  dataset: {len(samples)} images, image_size: {args.image_size}")
    print("-" * 60)

    pipe.transformer.train()
    seed = args.seed if args.seed is not None else 42
    generator = torch.Generator(device=exec_device).manual_seed(seed)

    for step in range(args.steps):
        # Pick a sample (cycle through dataset)
        sample = samples[step % len(samples)]
        caption = sample["caption"]

        # Encode text (no_grad)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=caption,
                device=exec_device,
                num_images_per_prompt=1,
            )

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
        t_scalar = torch.rand(1, device=exec_device, dtype=dtype)
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
        loss = F.mse_loss(noise_pred[:, :seq_len], target.to(dtype))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            print(f"  step {step + 1:4d}/{args.steps}  loss={loss.item():.6f}")

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
                prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=vs["caption"], device=exec_device, num_images_per_prompt=1,
                )
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
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    exec_device = device

    print("\nStarting SDXL LoRA training")
    print(f"  steps: {args.steps}, lr: {args.lr}, rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  dataset: {len(samples)} images, image_size: {args.image_size}")
    print("-" * 60)

    pipe.unet.train()

    for step in range(args.steps):
        sample = samples[step % len(samples)]
        caption = sample["caption"]

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

        # Add noise
        noise = torch.randn_like(latents)
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

        loss = F.mse_loss(model_pred.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            print(f"  step {step + 1:4d}/{args.steps}  loss={loss.item():.6f}")

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


# ── Qwen stub ─────────────────────────────────────────────────────────────────


def _train_qwen(args):
    """Placeholder for Qwen LoRA training — uses the generic path."""
    print("Qwen LoRA training is not yet implemented.")
    print("The Qwen model family requires custom LoRA support via its VLM architecture.")
    print("Use --family flux or --family sdxl for currently supported families.")
    sys.exit(1)


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
        elif "qwen" in model_lower or "firered" in model_lower:
            print(
                "Error: Qwen LoRA training is not yet supported. "
                "Use a FLUX or SDXL model instead."
            )
            sys.exit(1)
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
        help="Train a LoRA adapter for image generation models (FLUX, SDXL)",
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
        choices=["flux", "sdxl"],
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
    train_parser.set_defaults(func=cmd_train)
