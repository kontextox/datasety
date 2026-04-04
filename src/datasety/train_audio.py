"""TTS Model Training using Piper (and future backends)."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _ensure_piper_installed() -> Path:
    """Smart Auto-Installer: Compiles Piper and monotonic_align automatically if missing."""
    cache_dir = Path.home() / ".cache" / "datasety"
    piper_dir = cache_dir / "piper1-gpl"
    src_dir = piper_dir / "src"
    align_dir = src_dir / "piper" / "train" / "vits" / "monotonic_align"
    ma_target = align_dir / "monotonic_align"

    # Check if successfully compiled
    is_compiled = False
    if ma_target.exists():
        so_files = list(ma_target.glob("core*.so"))
        if so_files:
            is_compiled = True

    if piper_dir.exists() and is_compiled:
        return src_dir

    print("=" * 60)
    print("Setting up Piper TTS training environment (First-time only)...")
    print("This will download and compile the necessary C/Cython extensions.")
    print("=" * 60)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not piper_dir.exists():
        print("Cloning the fixed kontextox Piper repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/kontextox/piper1-gpl.git", str(piper_dir)],
            check=True,
        )

    print("Installing build dependencies (scikit-build, cython)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-build", "cython"], check=True)

    print("Installing Piper Python package...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[train]"], cwd=str(piper_dir), check=True
    )

    # --- AUTO-PATCH PIPER BUGS ---
    dataset_py = src_dir / "piper" / "train" / "vits" / "dataset.py"
    if dataset_py.exists():
        content = dataset_py.read_text(encoding="utf-8")
        if "phonemes_to_ids(sentence_phonemes)" in content and "id_map=" not in content:
            content = content.replace(
                "phonemes_to_ids(sentence_phonemes)",
                "phonemes_to_ids(sentence_phonemes, id_map=self.piper_config.phoneme_id_map)",
            )
            dataset_py.write_text(content, encoding="utf-8")
            print("Patched dataset.py to fix missing phoneme map bug.")

    export_py = src_dir / "piper" / "train" / "export_onnx.py"
    if export_py.exists():
        content = export_py.read_text(encoding="utf-8")
        if "dynamo=False" not in content and "opset_version=" in content:
            content = content.replace("opset_version=", "dynamo=False, opset_version=")
            export_py.write_text(content, encoding="utf-8")
            print("Patched export_onnx.py with dynamo=False to fix PyTorch 2.6+ bug.")
    # -----------------------------

    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(piper_dir), check=True
    )

    print("Compiling monotonic_align...")
    ma_target.mkdir(exist_ok=True)

    c_file = align_dir / "core.c"
    if c_file.exists():
        c_file.unlink()

    subprocess.run(["cythonize", "-i", "core.pyx"], cwd=str(align_dir), check=True)

    # Move the compiled .so object into the internal module directory
    for so in align_dir.glob("core*.so"):
        shutil.move(str(so), str(ma_target / so.name))

    print("Piper environment setup complete!\n")
    return src_dir


def _resolve_piper_model(model_str: str):
    """Resolve local directory or HF repo for Piper base model and config."""
    p = Path(model_str)
    if p.is_dir():
        ckpts = list(p.glob("*.ckpt"))
        cfg = p / "config.json"
        if not ckpts or not cfg.exists():
            raise ValueError(f"Local directory {model_str} must contain a .ckpt and config.json")
        return ckpts[0], cfg

    repo, _, subfolder = model_str.partition(":")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading base Piper model from {repo} (folder: {subfolder})...")
    path = snapshot_download(
        repo_id=repo, allow_patterns=f"{subfolder}/*" if subfolder else ["*.ckpt", "config.json"]
    )
    base_dir = Path(path)
    if subfolder:
        base_dir = base_dir / subfolder

    ckpts = list(base_dir.glob("*.ckpt"))
    cfg = base_dir / "config.json"
    if not ckpts or not cfg.exists():
        raise ValueError(f"No .ckpt or config.json found in HF repo {repo}")
    return ckpts[0], cfg


def _start_piper_watcher(output_dir: str, test_text: str, env_vars: dict):
    """Watches output for new Lightning checkpoints, exports to ONNX, and runs TTS inference."""

    def watch():
        seen = {}
        out_path = Path(output_dir)
        print(f"[Watcher] Started monitoring {out_path} for new .ckpt files to test...")
        while True:
            time.sleep(15)
            for ckpt in out_path.rglob("*.ckpt"):
                try:
                    mtime = ckpt.stat().st_mtime
                except FileNotFoundError:
                    continue

                if ckpt not in seen or seen[ckpt] != mtime:
                    seen[ckpt] = mtime
                    onnx_path = ckpt.with_suffix(".onnx")
                    wav_path = ckpt.with_suffix(".wav")
                    print(f"\n[Watcher] New checkpoint found: {ckpt.name}. Exporting to ONNX...")
                    try:
                        # Export to ONNX
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "piper.train.export_onnx",
                                "--checkpoint",
                                str(ckpt),
                                "--output-file",
                                str(onnx_path),
                            ],
                            env=env_vars,
                            check=True,
                            capture_output=True,
                        )

                        # Run Piper inference safely using stdin injection (cross-platform safe)
                        print(f"[Watcher] Generating test audio for {ckpt.name}...")
                        cmd = [
                            sys.executable,
                            "-m",
                            "piper",
                            "--model",
                            str(onnx_path),
                            "--output_file",
                            str(wav_path),
                        ]
                        subprocess.run(
                            cmd,
                            input=test_text.encode("utf-8"),
                            env=env_vars,
                            check=True,
                            capture_output=True,
                        )

                        print(f"[Watcher] Success! Test audio saved to: {wav_path}\n")
                    except Exception as e:
                        print(f"[Watcher] Error processing {ckpt.name}: {e}\n")

    t = threading.Thread(target=watch, daemon=True)
    t.start()


def cmd_train_audio(args):
    """Dispatch TTS training to requested backend."""
    if args.backend == "piper":
        _train_piper(args)
    else:
        print(f"Backend '{args.backend}' is planned for a future update. Using 'piper' for now.")
        _train_piper(args)


def _train_piper(args):
    """Main workflow to train Piper."""
    input_dir = Path(args.input)
    if not input_dir.exists() or not (input_dir / "metadata.csv").exists():
        print(f"Error: Valid TTS dataset directory (with metadata.csv) not found at {input_dir}.")
        sys.exit(1)

    # 1. Smart Auto-Installer
    # Compiles Cython/C extensions natively without bothering the user
    piper_src_dir = _ensure_piper_installed()

    try:
        ckpt_path, config_path = _resolve_piper_model(args.model)
    except Exception as e:
        print(f"Error resolving base model: {e}")
        sys.exit(1)

    # 2. Setup environment so python can locate the dynamically compiled `piper.train` module
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{piper_src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # 3. Extract custom phoneme map cleanly and read config settings
    phonemes_path = input_dir / "phonemes.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    phoneme_type = cfg.get("phoneme_type", "espeak") # Default to standard espeak
    espeak_voice = cfg.get("espeak_voice", "en-us")

    if not phonemes_path.exists():
        print("Extracting phonemes.json from base config...")
        if "phoneme_id_map" in cfg:
            with open(phonemes_path, "w", encoding="utf-8") as f:
                json.dump(cfg["phoneme_id_map"], f, ensure_ascii=False, indent=2)

    # 4. Handle paths gracefully
    output_dir = Path(args.output)
    if output_dir.suffix == ".safetensors":
        output_dir = output_dir.parent / output_dir.stem

    cache_dir = input_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build the Piper PyTorch Lightning run command
    cmd =[
        sys.executable,
        "-m",
        "piper.train",
        "fit",
        "--data.voice_name",
        "datasety_voice",
        "--data.csv_path",
        str(input_dir / "metadata.csv"),
        "--data.audio_dir",
        str(input_dir / "wavs"),
        "--model.sample_rate",
        str(args.sample_rate),
        "--data.phoneme_type",
        phoneme_type,
        "--data.dataset_type",
        "text",
        "--data.phonemes_path",
        str(phonemes_path),
        "--data.config_path",
        str(config_path),
        "--data.cache_dir",
        str(cache_dir),
        "--data.batch_size",
        str(args.batch_size),
        "--model.vocoder_warmstart_ckpt",
        str(ckpt_path),
        "--trainer.max_epochs",
        str(args.steps),
        "--trainer.default_root_dir",
        str(output_dir),
        "--trainer.precision",
        "16-mixed",
        "--trainer.accelerator",
        str(args.accelerator),
        "--trainer.devices",
        str(args.devices),
    ]

    if phoneme_type == "espeak":
        cmd.extend(["--data.espeak_voice", espeak_voice])

    # Auto-resume logic
    if output_dir.exists():
        last_ckpt = list(output_dir.rglob("last.ckpt"))
        if last_ckpt:
            print(f"Auto-resuming from {last_ckpt[0]}")
            cmd.extend(["--ckpt_path", str(last_ckpt[0])])

    # 6. Fire up the test background thread
    if getattr(args, "test_text", None):
        _start_piper_watcher(str(output_dir), args.test_text, env)

    # 7. Pre-flight RTX 5090 / CUDA 12.8 check
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 12:
            archs = torch.cuda.get_arch_list()
            if archs and not any(a.startswith("sm_12") for a in archs):
                print("\n" + "!" * 60)
                print(f"CRITICAL WARNING: Your GPU (Compute {major}.{minor}, e.g. RTX 5090)")
                print("is not natively supported by this PyTorch build.")
                print("Please upgrade PyTorch to a version supporting cu128:")
                print(
                    "pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
                )
                print("!" * 60 + "\n")

    # 8. Execute Training
    print("\n" + "=" * 60)
    print("🚀 Starting Piper TTS training...")
    if args.devices in ["auto", "-1"] or (str(args.devices).isdigit() and int(args.devices) > 1):
        print("⚡ Multi-GPU Detected: PyTorch Lightning will utilize all available GPUs via DDP.")
    print("Command:", " ".join(cmd))
    print("=" * 60)

    # Launch subprocess with the injected PYTHONPATH
    subprocess.run(cmd, env=env)
