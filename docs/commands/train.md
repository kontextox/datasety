# train — LoRA Fine-Tuning

Train a LoRA adapter for image generation models from a local dataset of image + caption pairs.

Supported model families: **FLUX.2-klein** (flow-matching), **SDXL** (DDPM), and **Qwen** (flow-matching, image-editing).

```bash
datasety train --input ./dataset --output lora.safetensors
```

## Dataset Format

The input directory must contain image files alongside matching `.txt` caption files:

```
dataset/
  001.jpg
  001.txt     ← "ohwx person wearing a red jacket"
  002.png
  002.txt     ← "ohwx person smiling outdoors"
  ...
```

Images are center-cropped to a square and resized to `--image-size` (default 512 px). Use `datasety resize`, `datasety caption`, and the other preparation commands to build the dataset before training.

## Base vs Distilled Models

> **Always use the base (undistilled) model for LoRA training.**

| Model                                    | Type                       | Use for             |
| ---------------------------------------- | -------------------------- | ------------------- |
| `black-forest-labs/FLUX.2-klein-4B`      | Step-distilled (4–8 steps) | Inference only      |
| `black-forest-labs/FLUX.2-klein-9B`      | Step-distilled (4–8 steps) | Inference only      |
| `black-forest-labs/FLUX.2-klein-base-4B` | Base (undistilled)         | **LoRA training** ✓ |
| `black-forest-labs/FLUX.2-klein-base-9B` | Base (undistilled)         | **LoRA training** ✓ |

The tool will print a warning if you pass a distilled model.

## Options

| Option                          | Description                                               | Default                                  |
| ------------------------------- | --------------------------------------------------------- | ---------------------------------------- |
| `--input`, `-i`                 | Dataset directory (images + `.txt` captions)              | required                                 |
| `--output`, `-o`                | Output LoRA `.safetensors` path                           | `lora.safetensors`                       |
| `--model`, `-m`                 | HuggingFace repo ID (base model)                          | `black-forest-labs/FLUX.2-klein-base-4B` |
| `--family`                      | Model family: `flux`, `sdxl`, `qwen`                      | auto-detected                            |
| `--steps`                       | Number of training steps                                  | `100`                                    |
| `--lr`                          | Learning rate                                             | `1e-4`                                   |
| `--lora-rank`                   | LoRA rank                                                 | `16`                                     |
| `--lora-alpha`                  | LoRA alpha                                                | `16.0`                                   |
| `--lora-dropout`                | LoRA dropout rate                                         | `0.0`                                    |
| `--image-size`                  | Training resolution (square crop)                         | `512`                                    |
| `--device`                      | `auto`, `cpu`, `cuda`, `mps`                              | `auto`                                   |
| `--seed`                        | Random seed                                               | `42`                                     |
| `--save-every`                  | Save checkpoint every N steps                             | end only                                 |
| `--resume`                      | Resume from a LoRA checkpoint (.safetensors)              |                                          |
| `--validation-split`            | Fraction of dataset for validation (0.0-0.5)              |                                          |
| `--timestep-type`               | Timestep sampling: `sigmoid`, `lognorm`, `linear`         | `sigmoid`                                |
| `--caption-dropout`             | Probability of dropping caption (unconditional)           | `0.05`                                   |
| `--gradient-checkpointing`      | Enable gradient checkpointing (saves VRAM)                | off                                      |
| `--optimizer`                   | `adamw` or `adamw8bit` (requires bitsandbytes)            | `adamw`                                  |
| `--lr-scheduler`                | LR schedule: `constant`, `cosine`, `linear`               | `constant`                               |
| `--lr-warmup-steps`             | Linear warmup steps before target LR                      | `0`                                      |
| `--gradient-accumulation-steps` | Accumulate gradients over N steps                         | `1`                                      |
| `--min-snr-gamma`               | Min-SNR-γ loss weighting for SDXL (recommended: 5.0)      | disabled                                 |
| `--noise-offset`                | Per-channel noise offset for SDXL (recommended: 0.05–0.1) | `0.0`                                    |

## Training Best Practices

The following defaults reflect ai-toolkit recommendations for fast, stable LoRA convergence:

| Technique                 | Option                            | Recommended value | Effect                                                               |
| ------------------------- | --------------------------------- | ----------------- | -------------------------------------------------------------------- |
| Sigmoid timestep sampling | `--timestep-type sigmoid`         | default           | Biases toward mid-timesteps where most learning happens (vs uniform) |
| Caption dropout           | `--caption-dropout 0.05`          | default           | 5% unconditional steps, improves CFG adherence                       |
| 8-bit optimizer           | `--optimizer adamw8bit`           | opt-in            | Halves optimizer state memory; requires `bitsandbytes`               |
| Cosine LR decay           | `--lr-scheduler cosine`           | opt-in            | Prevents late-stage oscillation                                      |
| Warmup                    | `--lr-warmup-steps 50`            | opt-in            | Avoids early large gradient steps                                    |
| Gradient accumulation     | `--gradient-accumulation-steps 4` | opt-in            | Simulates larger batch size without extra VRAM                       |
| Gradient checkpointing    | `--gradient-checkpointing`        | opt-in            | Reduces VRAM ~30% at ~20% speed cost                                 |
| Min-SNR-γ (SDXL only)     | `--min-snr-gamma 5.0`             | opt-in            | Stabilises DDPM loss across timesteps                                |
| Noise offset (SDXL only)  | `--noise-offset 0.05`             | opt-in            | Improves dark/bright image generation                                |

## Examples

### FLUX.2-klein LoRA with all best practices

```bash
datasety train \
    --input ./dataset \
    --output ./lora/flux_lora.safetensors \
    --model black-forest-labs/FLUX.2-klein-base-4B \
    --steps 500 \
    --lr 1e-4 \
    --lora-rank 16 \
    --timestep-type sigmoid \
    --caption-dropout 0.05 \
    --optimizer adamw8bit \
    --lr-scheduler cosine \
    --lr-warmup-steps 50 \
    --gradient-accumulation-steps 2 \
    --gradient-checkpointing
```

### FLUX.2-klein LoRA (recommended)

Prepare a dataset first, then train:

```bash
# 1. Prepare dataset
datasety resize -i ./raw -o ./dataset -r 512x512
datasety caption -i ./dataset -o ./dataset --trigger-word "ohwx person,"

# 2. Train LoRA on FLUX.2-klein-base-4B (~8 GB VRAM)
datasety train \
    --input ./dataset \
    --output ./lora/flux_lora.safetensors \
    --model black-forest-labs/FLUX.2-klein-base-4B \
    --steps 500 \
    --lr 1e-4 \
    --lora-rank 16

# 3. Use the trained LoRA with synthetic editing
datasety synthetic \
    --input-image photo.jpg \
    --output-image result.png \
    --prompt "ohwx person wearing sunglasses" \
    --lora ./lora/flux_lora.safetensors:0.8
```

### SDXL LoRA

```bash
datasety train \
    --input ./dataset \
    --output sdxl_lora.safetensors \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --family sdxl \
    --steps 500 \
    --lr 1e-4 \
    --lora-rank 16 \
    --image-size 1024
```

### Qwen Image-Edit LoRA (~30 GB VRAM)

Train a LoRA for `Qwen/Qwen-Image-Edit-2511` (or `2509`). The dataset needs only
image + caption pairs — the same image is used as both source and target
(reconstruction training), which teaches the model to preserve identity/style.
Image size must be a multiple of 32.

```bash
datasety train \
    --input ./dataset \
    --output qwen_lora.safetensors \
    --model Qwen/Qwen-Image-Edit-2511 \
    --steps 500 \
    --lr 5e-5 \
    --lora-rank 16 \
    --image-size 512
```

The resulting LoRA loads directly with `--lora` in `datasety synthetic`:

```bash
datasety synthetic \
    --input-image photo.jpg --output-image edited.png \
    --model Qwen/Qwen-Image-Edit-2511 \
    --lora qwen_lora.safetensors:0.8 \
    --prompt "ohwx person wearing a winter hat"
```

### Quick test run (20 steps)

Verify the training loop works before a full run:

```bash
datasety train \
    --input ./dataset \
    --output test_lora.safetensors \
    --steps 20 \
    --save-every 10
```

### Resume from checkpoint

```bash
datasety train \
    --input ./dataset \
    --output lora.safetensors \
    --resume lora_step200.safetensors \
    --steps 500
```

### Training with validation

```bash
datasety train \
    --input ./dataset \
    --output lora.safetensors \
    --steps 500 \
    --validation-split 0.1    # 10% of images held out for validation loss
```

### Save checkpoints during training

```bash
datasety train \
    --input ./dataset \
    --output lora.safetensors \
    --steps 1000 \
    --save-every 200    # saves lora_step200.safetensors, lora_step400.safetensors, ...
```

## VRAM Requirements

| Model                     | VRAM   | Notes                               |
| ------------------------- | ------ | ----------------------------------- |
| FLUX.2-klein-base-4B      | ~8 GB  | Default, auto CPU-offload if needed |
| FLUX.2-klein-base-9B      | ~18 GB | Higher quality                      |
| SDXL                      | ~7 GB  | Good for object/style LoRAs         |
| Qwen/Qwen-Image-Edit-2511 | ~30 GB | Image-editing LoRA, flow-matching   |

CPU offload is applied automatically when free VRAM is below the required amount.

## LoRA Parameters Guide

| Parameter      | Recommended range       | Effect                                          |
| -------------- | ----------------------- | ----------------------------------------------- |
| `--lora-rank`  | 4–64                    | Higher = more capacity, larger file             |
| `--lora-alpha` | Equal to rank (default) | Controls effective learning rate scale          |
| `--steps`      | 100–2000                | More steps = more fitting (risk of overfitting) |
| `--lr`         | `1e-5` – `1e-3`         | Too high causes divergence; too low is slow     |
| `--image-size` | 512 or 1024             | Match your target inference resolution          |

## Output

The trained LoRA is saved as a `.safetensors` file in **diffusers-compatible format** — keys use the `transformer.` prefix for FLUX and Qwen models and `unet.` for SDXL, so the file loads directly with `pipeline.load_lora_weights()` and with `--lora` in datasety commands:

```bash
datasety synthetic -i ./images -o ./output \
    --prompt "ohwx person in a park" \
    --lora lora.safetensors:0.8
```

The LoRA weight (`:0.8`) controls blend strength — typically `0.6`–`1.0`.

### Target modules

| Model                     | Trainable modules                                                                          | File size (rank 16) |
| ------------------------- | ------------------------------------------------------------------------------------------ | ------------------- |
| FLUX.2-klein-base-4B      | `to_q`, `to_k`, `to_v`, `to_out.0`                                                         | ~38 MB              |
| FLUX.2-klein-base-9B      | `to_q`, `to_k`, `to_v`, `to_qkv_mlp_proj`                                                  | ~77 MB              |
| SDXL                      | `to_q`, `to_k`, `to_v`, `to_out.0`                                                         | ~25 MB              |
| Qwen/Qwen-Image-Edit-2511 | `to_q`, `to_k`, `to_v`, `to_out.0`, `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out` | ~45 MB              |

The 9B FLUX model uses fused `to_qkv_mlp_proj` projections (single-transformer blocks). Qwen targets both image-stream and text-stream attention projections across 60 transformer blocks.
