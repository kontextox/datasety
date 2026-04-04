"""LoRA fine-tuning for image generation models and TTS models."""


def cmd_train(args):
    """Execute the train command."""
    from datasety.upload import detect_dataset_type

    # Auto-detect task from explicit family or dataset content
    dt = "image"
    if args.family in ["flux", "sdxl", "qwen"]:
        dt = "image"
    elif args.family in ["piper", "coqui", "f5-tts"] or args.backend in [
        "piper",
        "coqui",
        "f5-tts",
    ]:
        dt = "audio"
    else:
        dt = detect_dataset_type(args.input)

    if dt == "audio":
        from datasety.train_audio import cmd_train_audio

        cmd_train_audio(args)
    else:
        from datasety.train_image import cmd_train_image

        cmd_train_image(args)


def register_parser(subparsers):
    """Register the unified train subcommand."""
    train_parser = subparsers.add_parser(
        "train",
        help="Train a LoRA adapter (images) or a TTS model (audio)",
    )

    # Common / Image Arguments
    train_parser.add_argument("--input", "-i", required=True, help="Input dataset directory")
    train_parser.add_argument(
        "--output", "-o", default="lora.safetensors", help="Output path or directory"
    )
    train_parser.add_argument(
        "--model",
        "-m",
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="Model HF repo ID (optionally with :subfolder for Piper) or local path",
    )
    train_parser.add_argument(
        "--family", default=None, help="Model family (flux, sdxl, qwen, piper, etc.)"
    )
    train_parser.add_argument(
        "--steps", type=int, default=100, help="Training steps (or epochs for audio)"
    )
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    train_parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout rate")
    train_parser.add_argument(
        "--image-size", type=int, default=512, help="Training image resolution"
    )
    train_parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Device"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--save-every", type=int, default=None, help="Save checkpoint every N steps"
    )
    train_parser.add_argument("--resume", default=None, help="Resume training from a checkpoint")
    train_parser.add_argument(
        "--validation-split", type=float, default=0.0, help="Validation fraction"
    )
    train_parser.add_argument(
        "--timestep-type", choices=["sigmoid", "lognorm", "linear"], default="sigmoid"
    )
    train_parser.add_argument(
        "--caption-dropout", type=float, default=0.05, help="Caption dropout rate"
    )
    train_parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    train_parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    train_parser.add_argument(
        "--lr-scheduler", choices=["constant", "cosine", "linear"], default="constant"
    )
    train_parser.add_argument("--lr-warmup-steps", type=int, default=0)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--min-snr-gamma", type=float, default=None)
    train_parser.add_argument("--noise-offset", type=float, default=0.0)

    # Audio & PyTorch Lightning Specific Arguments
    train_parser.add_argument(
        "--backend",
        choices=["piper", "coqui", "f5-tts"],
        default="piper",
        help="Audio training backend",
    )
    train_parser.add_argument(
        "--test-text", default=None, help="Background inference test text during training"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size (audio)")
    train_parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Audio sample rate (audio)"
    )
    train_parser.add_argument(
        "--accelerator", default="auto", help="PyTorch Lightning accelerator (auto, gpu, cpu)"
    )
    train_parser.add_argument(
        "--devices",
        default="auto",
        help="PyTorch Lightning devices (e.g., auto, 1, 2, -1 for all GPUs)",
    )

    train_parser.set_defaults(func=cmd_train)
