"""Generate synthetic images using image editing models."""

import sys
from pathlib import Path

from PIL import Image

from datasety.common import (
    _resolve_gguf_path,
    _resolve_hf_file,
    _resolve_io_mode,
    get_image_files,
    resolve_device,
)

_MODEL_VRAM_GB = {
    "qwen": 32,  # ~31 GB peak with offload; needs sequential offload on 32 GB
    "flux-kontext": 33,  # ~33 GB non-offloaded; triggers offload on 32 GB cards
    # Distilled (inference) variants — fast few-step generation, NOT for training
    "flux2-klein": 8,  # 4B bf16 distilled
    "flux2-klein-9b": 18,  # 9B bf16 distilled
    "flux2-klein-9b-kv": 29,  # 9B KV-cache optimized, ~29 GB VRAM
    "flux2-klein-4b-fp8": 5,  # 4B fp8 distilled — ~4-5 GB (local: falls back to bf16)
    "flux2-klein-9b-fp8": 10,  # 9B fp8 distilled — ~9-10 GB (local: falls back to bf16)
    # Base (undistilled) variants — recommended for LoRA fine-tuning
    "flux2-klein-base": 8,  # 4B bf16 base
    "flux2-klein-base-9b": 18,  # 9B bf16 base
    "flux2-klein-base-4b-fp8": 5,  # 4B fp8 base (local: falls back to bf16 base)
    "flux2-klein-base-9b-fp8": 10,  # 9B fp8 base (local: falls back to bf16 base)
    "flux2-dev": 24,
    "longcat": 18,
    "sdxl": 7,
    "hunyuan": 48,
}


def _detect_model_family(model_name: str) -> str:
    """Detect model family from model name/path.

    Distinguishes between distilled inference models (flux2-klein) and
    undistilled base models (flux2-klein-base) used for training.
    """
    name_lower = model_name.lower()
    if "kontext" in name_lower:
        return "flux-kontext"
    if "klein" in name_lower:
        is_base = "base" in name_lower
        is_9b = "9b" in name_lower
        is_fp8 = "fp8" in name_lower
        is_kv = "kv" in name_lower
        if is_base:
            if is_fp8:
                return "flux2-klein-base-9b-fp8" if is_9b else "flux2-klein-base-4b-fp8"
            return "flux2-klein-base-9b" if is_9b else "flux2-klein-base"
        else:
            if is_kv and is_9b:
                return "flux2-klein-9b-kv"
            if is_fp8:
                return "flux2-klein-9b-fp8" if is_9b else "flux2-klein-4b-fp8"
            return "flux2-klein-9b" if is_9b else "flux2-klein"
    if "flux.2" in name_lower or "flux2" in name_lower:
        if "dev" in name_lower:
            return "flux2-dev"
        return "flux2-klein"
    if "longcat" in name_lower:
        return "longcat"
    if "firered" in name_lower:
        return "qwen"
    if "stable-diffusion-xl" in name_lower or "sdxl" in name_lower:
        return "sdxl"
    if "hunyuan" in name_lower:
        return "hunyuan"
    return "qwen"


def _load_synthetic_pipeline(model_name, family, device, torch_dtype, gguf_path, cpu_offload):
    """Load the appropriate diffusion pipeline for the model family."""
    if family == "qwen":
        from diffusers import QwenImageEditPlusPipeline

        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import GGUFQuantizationConfig, QwenVLTransformer2DModel

            transformer = QwenVLTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=model_name,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer
        pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name, **kwargs)

    elif family == "flux-kontext":
        from diffusers import FluxKontextPipeline

        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=model_name,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer
        pipeline = FluxKontextPipeline.from_pretrained(model_name, **kwargs)

    elif family == "flux2-klein-9b-kv":
        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig

            transformer = Flux2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=model_name,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer
        try:
            from diffusers import Flux2KleinKVPipeline
        except ImportError:
            import subprocess

            print(
                "Flux2KleinKVPipeline not found, upgrading diffusers from "
                "official HuggingFace repo..."
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "git+https://github.com/huggingface/diffusers.git",
                ]
            )
            import importlib

            for _key in [k for k in sys.modules if k.startswith("diffusers")]:
                del sys.modules[_key]
            importlib.invalidate_caches()
            from diffusers import Flux2KleinKVPipeline
        pipeline = Flux2KleinKVPipeline.from_pretrained(model_name, **kwargs)

    elif family in (
        "flux2-klein",
        "flux2-klein-9b",
        "flux2-klein-4b-fp8",
        "flux2-klein-9b-fp8",
        "flux2-klein-base",
        "flux2-klein-base-9b",
        "flux2-klein-base-4b-fp8",
        "flux2-klein-base-9b-fp8",
    ):
        kwargs = {"torch_dtype": torch_dtype}
        # FP8 single-file repos (BFL proprietary format) cannot be loaded directly by
        # diffusers. Automatically fall back to the equivalent BF16 model.
        _fp8_to_bf16 = {
            "flux2-klein-4b-fp8": "black-forest-labs/FLUX.2-klein-4B",
            "flux2-klein-9b-fp8": "black-forest-labs/FLUX.2-klein-9B",
            "flux2-klein-base-4b-fp8": "black-forest-labs/FLUX.2-klein-base-4B",
            "flux2-klein-base-9b-fp8": "black-forest-labs/FLUX.2-klein-base-9B",
        }
        if family in _fp8_to_bf16:
            bf16_model = _fp8_to_bf16[family]
            print(
                f"Note: FP8 single-file format is not yet loadable by diffusers locally. "
                f"Using equivalent BF16 model: {bf16_model}. "
                f"For true FP8 inference use --image-api."
            )
            model_name = bf16_model
        if gguf_path:
            from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig

            transformer = Flux2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=model_name,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer
        # Flux2KleinPipeline requires diffusers >= 0.37.0
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            import subprocess

            print(
                "Flux2KleinPipeline not found, upgrading diffusers from "
                "official HuggingFace repo..."
            )
            print("Installing: git+https://github.com/huggingface/diffusers.git")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "git+https://github.com/huggingface/diffusers.git",
                ]
            )
            # Clear cached diffusers modules so the upgraded version is loaded
            import importlib

            for _key in [k for k in sys.modules if k.startswith("diffusers")]:
                del sys.modules[_key]
            importlib.invalidate_caches()
            from diffusers import Flux2KleinPipeline
        pipeline = Flux2KleinPipeline.from_pretrained(model_name, **kwargs)

    elif family == "flux2-dev":
        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig

            transformer = Flux2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=model_name,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer
        from diffusers import Flux2Pipeline

        pipeline = Flux2Pipeline.from_pretrained(model_name, **kwargs)

    elif family == "longcat":
        from diffusers import LongCatImageEditPipeline

        pipeline = LongCatImageEditPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

    elif family == "sdxl":
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )

    elif family == "hunyuan":
        from diffusers import HunyuanImagePipeline

        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import GGUFQuantizationConfig, HunyuanVideo3DTransformerModel

            transformer = HunyuanVideo3DTransformerModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=model_name,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer
        pipeline = HunyuanImagePipeline.from_pretrained(model_name, **kwargs)

    else:
        raise ValueError(f"Unknown model family: {family}")

    if cpu_offload:
        import torch

        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        needed_gb = _MODEL_VRAM_GB.get(family, 16)
        # Sequential CPU offload is incompatible with GGUF quantized models
        if needed_gb >= total_gb and not gguf_path:
            pipeline.enable_sequential_cpu_offload()
            print("Sequential CPU offload enabled")
        else:
            pipeline.enable_model_cpu_offload()
            print("Model CPU offload enabled")
    else:
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def _run_synthetic_pipeline(pipeline, family, image, args, device, cpu_offload):
    """Run the pipeline with family-specific parameter mapping."""
    import torch

    gen_device = "cpu" if cpu_offload else device

    gen_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "num_images_per_prompt": args.num_images,
    }

    if args.seed is not None:
        gen_kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(args.seed)

    if family == "qwen":
        gen_kwargs["image"] = [image]
        gen_kwargs["negative_prompt"] = args.negative_prompt
        gen_kwargs["guidance_scale"] = args.cfg_scale
        gen_kwargs["true_cfg_scale"] = args.true_cfg_scale

    elif family == "flux-kontext":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale

    elif family == "flux2-klein-9b-kv":
        # Flux2KleinKVPipeline does not accept guidance_scale
        gen_kwargs["image"] = [image]

    elif family in (
        "flux2-klein",
        "flux2-klein-9b",
        "flux2-klein-4b-fp8",
        "flux2-klein-9b-fp8",
        "flux2-klein-base",
        "flux2-klein-base-9b",
        "flux2-klein-base-4b-fp8",
        "flux2-klein-base-9b-fp8",
    ):
        gen_kwargs["image"] = [image]
        gen_kwargs["guidance_scale"] = args.cfg_scale

    elif family == "flux2-dev":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale
        gen_kwargs["strength"] = args.strength

    elif family == "longcat":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale
        if args.negative_prompt and args.negative_prompt.strip():
            gen_kwargs["negative_prompt"] = args.negative_prompt

    elif family == "sdxl":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale
        gen_kwargs["strength"] = args.strength
        if args.negative_prompt and args.negative_prompt.strip():
            gen_kwargs["negative_prompt"] = args.negative_prompt

    elif family == "hunyuan":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale

    with torch.inference_mode():
        output = pipeline(**gen_kwargs)

    return output


def _parse_lora_spec(spec):
    """Parse a LoRA specification string.

    Formats:
      - "path/to/lora.safetensors" -> (path, 1.0)
      - "path/to/lora.safetensors:0.8" -> (path, 0.8)
      - "https://huggingface.co/user/repo/resolve/main/lora.safetensors:0.5"
      - "user/repo:0.8" (HF repo ID with weight)

    The weight suffix is only recognized if the part after the last ':' is a
    valid float.  This avoids ambiguity with HF URLs containing colons.
    """
    # Try splitting on last ':'
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        try:
            weight = float(tail)
            return head, weight
        except ValueError:
            pass
    return spec, 1.0


def _load_lora_adapters(pipeline, lora_specs):
    """Load one or more LoRA adapters into the pipeline.

    Each spec is parsed by _parse_lora_spec.  HF URLs (both /blob/ and /resolve/
    forms) are downloaded automatically.
    """
    adapter_names = []
    adapter_weights = []

    for i, spec in enumerate(lora_specs):
        path_or_repo, weight = _parse_lora_spec(spec)

        # Resolve HF URLs to local paths
        resolved = _resolve_hf_file(path_or_repo)

        adapter_name = f"lora_{i}"
        load_kwargs = {"adapter_name": adapter_name}

        # If it's a local .safetensors file, pass as weight_name with directory
        if resolved != path_or_repo or Path(resolved).is_file():
            local_path = Path(resolved)
            if local_path.is_file():
                load_kwargs["weight_name"] = local_path.name
                resolved = str(local_path.parent)

        print(f"Loading LoRA [{adapter_name}]: {path_or_repo} (weight={weight})")
        pipeline.load_lora_weights(resolved, **load_kwargs)

        adapter_names.append(adapter_name)
        adapter_weights.append(weight)

    if len(adapter_names) > 1:
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        print(
            f"Activated {len(adapter_names)} LoRA adapters: "
            f"{list(zip(adapter_names, adapter_weights))}"
        )
    elif len(adapter_names) == 1 and adapter_weights[0] != 1.0:
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        print(f"LoRA weight set to {adapter_weights[0]}")


def cmd_synthetic(args):
    """Execute the synthetic image generation command."""
    single_files, output_path, is_single = _resolve_io_mode(args)

    if is_single:
        image_files = single_files
        output_dir = output_path.parent if output_path else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        input_dir = Path(args.input)
        output_dir = output_path

        if not input_dir.exists():
            print(f"Error: Input directory '{input_dir}' does not exist.")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

    # Find images early so dry-run can preview without loading models
    if not is_single:
        formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
        image_files = get_image_files(Path(args.input), formats, recursive=args.recursive)

        if not image_files:
            print(f"No images found in '{args.input}'")
            sys.exit(0)

    # Detect model family
    family = _detect_model_family(args.model)

    dry_run = args.dry_run
    if dry_run:
        print("=== DRY RUN (no files will be written) ===")
        mode = "API" if args.image_api else f"local ({family})"
        print(f"Model: {args.model} (mode: {mode})")
        print(f"Found {len(image_files)} images")
        print(f"Prompt: {args.prompt}")
        if not args.image_api:
            print(f"Steps: {args.steps}, CFG: {args.cfg_scale}")
        print("-" * 50)
        out_ext = args.output_format.lower()
        for idx, img_path in enumerate(image_files, 1):
            if is_single and output_path:
                print(f"  [{idx}/{len(image_files)}] {img_path.name} -> {output_path.name}")
            elif args.num_images > 1:
                for j in range(args.num_images):
                    p = output_dir / f"{img_path.stem}_{j + 1}.{out_ext}"
                    print(f"  [{idx}/{len(image_files)}] {img_path.name} -> {p.name}")
            else:
                out_path = output_dir / f"{img_path.stem}.{out_ext}"
                print(f"  [{idx}/{len(image_files)}] {img_path.name} -> {out_path.name}")
        print("-" * 50)
        print(f"\nRun without --dry-run to generate {len(image_files)} image(s).")
        return

    # ── Cloud API path ──
    if args.image_api:
        from datasety.llm import _generate_image_via_api, resolve_llm_api_config

        # Map model names to their canonical API IDs (lowercase) used by OpenRouter
        # and other compatible endpoints. Both distilled and base models map to the
        # same API endpoint (the API serves the distilled variant for speed).
        _api_model_map = {
            "black-forest-labs/FLUX.2-klein-4b-fp8": "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/FLUX.2-klein-9b-fp8": "black-forest-labs/flux.2-klein-9b",
            "black-forest-labs/FLUX.2-klein-4B": "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/FLUX.2-klein-9B": "black-forest-labs/flux.2-klein-9b",
            "black-forest-labs/FLUX.2-klein-9b-kv": "black-forest-labs/flux.2-klein-9b",
            "black-forest-labs/FLUX.2-klein-base-4B": "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/FLUX.2-klein-base-9B": "black-forest-labs/flux.2-klein-9b",
            "black-forest-labs/FLUX.2-klein-base-4b-fp8": "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/FLUX.2-klein-base-9b-fp8": "black-forest-labs/flux.2-klein-9b",
        }
        api_model_name = _api_model_map.get(args.model, args.model) if args.model else None
        api_key, base_url, model = resolve_llm_api_config(api_model_name or None)
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is required for --image-api")
            sys.exit(1)

        print(f"Using image API: {base_url}")
        print(f"Model: {model}")
        print(f"Found {len(image_files)} images")
        print(f"Prompt: {args.prompt}")
        print("-" * 50)

        processed = 0
        total = len(image_files)
        out_ext = args.output_format.lower()

        for idx, img_path in enumerate(image_files, 1):
            try:
                with Image.open(img_path) as img:
                    input_image = img.convert("RGB").copy()

                result = _generate_image_via_api(
                    args.prompt,
                    api_key,
                    base_url,
                    model,
                    input_image=input_image,
                    seed=args.seed,
                    aspect_ratio=getattr(args, "api_aspect_ratio", None),
                    image_size=getattr(args, "api_image_size", None),
                )

                if is_single and output_path:
                    out_path = output_path
                else:
                    out_path = output_dir / f"{img_path.stem}.{out_ext}"

                result.save(out_path)
                print(f"[{idx}/{total}] [OK] {img_path.name} -> {out_path.name}")
                processed += 1

            except Exception as e:
                print(f"[{idx}/{total}] [ERROR] {img_path.name}: {e}")

        print("-" * 50)
        print(f"Done! Processed: {processed} images")

        if processed == 0 and image_files:
            print("Error: All images failed to process.")
            sys.exit(1)
        return

    # ── Local pipeline path ──
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed.")
        print("Run: pip install 'datasety[synthetic]'")
        sys.exit(1)

    # Determine device
    device = resolve_device(args.device)

    torch_dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    print(f"Loading model: {args.model} (family: {family})")
    print(f"Device: {device}")

    # Auto-detect cpu_offload if not explicitly set
    if device == "cuda" and not args.cpu_offload:
        torch.cuda.empty_cache()  # Free cached allocator memory before checking
        free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        needed_gb = _MODEL_VRAM_GB.get(family, 16)
        if free_vram_gb >= needed_gb:
            # Enough free VRAM — load directly, no offload
            cpu_offload = False
        elif total_vram_gb >= needed_gb:
            # Model fits in total VRAM but currently occupied — use model_cpu_offload
            cpu_offload = True
            print(
                f"Auto-enabling model CPU offload: {free_vram_gb:.1f} GB free, "
                f"model needs ~{needed_gb} GB (total: {total_vram_gb:.1f} GB)"
            )
        else:
            # Model exceeds total VRAM — must use sequential (layer-by-layer) offload
            cpu_offload = True
            print(
                f"Auto-enabling sequential CPU offload: {free_vram_gb:.1f} GB free, "
                f"model needs ~{needed_gb} GB (total: {total_vram_gb:.1f} GB)"
            )
    else:
        cpu_offload = args.cpu_offload

    gguf_path = _resolve_gguf_path(getattr(args, "gguf", None))

    try:
        pipeline = _load_synthetic_pipeline(
            args.model,
            family,
            device,
            torch_dtype,
            gguf_path,
            cpu_offload,
        )
    except ImportError as e:
        print(f"Error: Missing dependency for {family} pipeline: {e}")
        print("Make sure you have the latest diffusers: pip install -U diffusers")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Inject fine-tuned weights if specified (Qwen-specific)
    if args.weights:
        if family != "qwen":
            print("Warning: --weights is only supported for Qwen models, ignoring.")
        else:
            import gc

            try:
                from safetensors.torch import load_file
            except ImportError:
                print("Error: safetensors is required for --weights.")
                print("Run: pip install safetensors")
                sys.exit(1)

            # Support HF URLs, repo_id:filename, or local paths
            weights_val = args.weights
            if weights_val.startswith(("https://", "http://")):
                weight_path = _resolve_hf_file(weights_val)
            elif ":" in weights_val:
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError:
                    print(
                        "Error: huggingface_hub is required for --weights with "
                        "repo_id:filename format."
                    )
                    print("Run: pip install 'huggingface_hub>=0.20.0'")
                    sys.exit(1)

                repo_id, filename = weights_val.split(":", 1)
                print(f"Downloading weights: {repo_id} / {filename}")
                weight_path = hf_hub_download(repo_id, filename)
            else:
                weight_path = weights_val

            print("Loading weight file...")
            state_dict = load_file(weight_path)

            # Sort weights by key prefix into component dicts
            transformer_weights = {}
            vae_weights = {}
            text_encoder_weights = {}

            for key, value in state_dict.items():
                if key.startswith(("model.diffusion_model.", "transformer.")):
                    for prefix in ("model.diffusion_model.", "transformer."):
                        if key.startswith(prefix):
                            transformer_weights[key[len(prefix) :]] = value
                            break
                elif key.startswith(("first_stage_model.", "vae.")):
                    for prefix in ("first_stage_model.", "vae."):
                        if key.startswith(prefix):
                            vae_weights[key[len(prefix) :]] = value
                            break
                elif "text_encoder" in key or "conditioner" in key:
                    text_encoder_weights[key] = value

            if transformer_weights:
                print(f"Injecting {len(transformer_weights)} transformer weights")
                pipeline.transformer.load_state_dict(
                    transformer_weights,
                    strict=False,
                    assign=True,
                )

            if vae_weights:
                print(f"Injecting {len(vae_weights)} VAE weights")
                pipeline.vae.load_state_dict(vae_weights, strict=False, assign=True)

            if text_encoder_weights:
                print(f"Injecting {len(text_encoder_weights)} text encoder weights")
                pipeline.text_encoder.load_state_dict(
                    text_encoder_weights,
                    strict=False,
                    assign=True,
                )

            del state_dict, transformer_weights, vae_weights, text_encoder_weights
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            print("Weights injected successfully")

    # Load LoRA adapters if specified
    if getattr(args, "lora", None):
        _load_lora_adapters(pipeline, args.lora)

    print(f"Found {len(image_files)} images")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, CFG: {args.cfg_scale}")
    if family == "qwen":
        print(f"True CFG: {args.true_cfg_scale}")
    print("-" * 50)

    processed = 0
    total = len(image_files)

    out_ext = args.output_format.lower()

    for idx, img_path in enumerate(image_files, 1):
        try:
            # Skip if output already exists
            if getattr(args, "skip_existing", False) and not is_single:
                out_check = output_dir / f"{img_path.stem}.{out_ext}"
                if out_check.exists():
                    print(f"[{idx}/{total}] [SKIP] {img_path.name} (output exists)")
                    continue

            with Image.open(img_path) as img:
                image = img.convert("RGB").copy()

            output = _run_synthetic_pipeline(pipeline, family, image, args, device, cpu_offload)

            # Save output image(s)
            for out_idx, out_img in enumerate(output.images):
                if is_single and output_path:
                    out_path = output_path
                elif args.num_images > 1:
                    out_path = output_dir / f"{img_path.stem}_{out_idx + 1}.{out_ext}"
                else:
                    out_path = output_dir / f"{img_path.stem}.{out_ext}"

                out_img.save(out_path)

            print(f"[{idx}/{total}] [OK] {img_path.name} -> {len(output.images)} image(s)")
            processed += 1

            # Flush GPU memory periodically
            batch_size = getattr(args, "batch_size", 0)
            if batch_size and processed % batch_size == 0:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        except Exception as e:
            print(f"[{idx}/{total}] [ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")

    if processed == 0 and image_files:
        print("Error: All images failed to process.")
        sys.exit(1)


def register_parser(subparsers):
    """Register the synthetic subcommand."""
    synthetic_parser = subparsers.add_parser(
        "synthetic", help="Generate synthetic images using image editing models"
    )
    synthetic_parser.add_argument(
        "--input", "-i", default="", help="Input directory containing images"
    )
    synthetic_parser.add_argument(
        "--output", "-o", default="", help="Output directory for generated images"
    )
    synthetic_parser.add_argument(
        "--input-image", default=None, help="Single input image path (alternative to --input dir)"
    )
    synthetic_parser.add_argument(
        "--output-image", default=None, help="Single output image path (use with --input-image)"
    )
    synthetic_parser.add_argument(
        "--prompt", "-p", required=True, help="Edit prompt (e.g., 'add a winter hat to the person')"
    )
    synthetic_parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4b-fp8",
        help="Model to use (default: black-forest-labs/FLUX.2-klein-4b-fp8)",
    )
    synthetic_parser.add_argument(
        "--weights",
        default=None,
        help="Fine-tuned weights as 'repo_id:filename' "
        "(e.g., 'Phr00t/Qwen-Image-Edit-Rapid-AIO:v23/model.safetensors')",
    )
    synthetic_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU/MPS)",
    )
    synthetic_parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Force CPU offload (auto-detected by default based on available VRAM)",
    )
    synthetic_parser.add_argument(
        "--steps", type=int, default=4, help="Number of inference steps (default: 4)"
    )
    synthetic_parser.add_argument(
        "--cfg-scale", type=float, default=2.5, help="Guidance scale (default: 2.5)"
    )
    synthetic_parser.add_argument(
        "--true-cfg-scale", type=float, default=4.0, help="True CFG scale (default: 4.0)"
    )
    synthetic_parser.add_argument(
        "--negative-prompt", default=" ", help="Negative prompt (default: ' ')"
    )
    synthetic_parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate per input (default: 1)",
    )
    synthetic_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    synthetic_parser.add_argument(
        "--gguf", default=None, help="Path or URL to GGUF file for quantized transformer loading"
    )
    synthetic_parser.add_argument(
        "--lora",
        action="append",
        default=None,
        help="LoRA adapter: path, HF repo, or URL to .safetensors file. "
        "Optionally append :WEIGHT (e.g., 'adapter.safetensors:0.8'). "
        "Can be specified multiple times for multiple LoRAs.",
    )
    synthetic_parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="Img2img strength for SDXL/FLUX.2 (0.0-1.0, default: 0.7)",
    )
    synthetic_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)",
    )
    synthetic_parser.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    synthetic_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts and output paths without loading models or generating images",
    )
    synthetic_parser.add_argument(
        "--image-api",
        action="store_true",
        help="Use OpenAI-compatible API for image generation (needs OPENAI_API_KEY)",
    )
    synthetic_parser.add_argument(
        "--api-aspect-ratio",
        default=None,
        metavar="RATIO",
        help="Aspect ratio for --image-api (e.g. '16:9', '1:1', '3:2'). "
        "Auto-detected from input image if not set.",
    )
    synthetic_parser.add_argument(
        "--api-image-size",
        default=None,
        choices=["0.5K", "1K", "2K", "4K"],
        help="Output resolution for --image-api (default: 1K).",
    )
    synthetic_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar instead of per-file output",
    )
    synthetic_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Flush GPU memory every N images (default: 0 = no flushing)",
    )
    synthetic_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output file already exists",
    )
    synthetic_parser.set_defaults(func=cmd_synthetic)
