# synthetic

Generate synthetic variations of images using image editing models.

## Usage

```bash
datasety synthetic --input ./images --output ./synthetic --prompt "add a winter hat"
```

## Supported Models

| Model Family | Example Model ID | Key Params |
| --- | --- | --- |
| **FLUX.2 klein** (default) | `black-forest-labs/FLUX.2-klein-4B` | `--strength`, steps=4 |
| **FLUX.2 dev** | `black-forest-labs/FLUX.2-dev` | `--strength`, steps=28 |
| **Qwen** | `Qwen/Qwen-Image-Edit-2511` | `--true-cfg-scale`, steps=40 |
| **FireRed** | `FireRedTeam/FireRed-Image-Edit-1.0` | `--true-cfg-scale`, steps=40 |
| **FLUX Kontext** | `black-forest-labs/FLUX.1-Kontext-dev` | steps=28 |
| **LongCat** | `meituan-longcat/LongCat-Image-Edit-Turbo` | steps=50 |
| **SDXL** | `stabilityai/stable-diffusion-xl-*` | `--strength`, steps=30 |
| **HunyuanImage** | `tencent/HunyuanImage-3.0` | steps=50 |

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--input`, `-i` | Input directory | (required*) |
| `--output`, `-o` | Output directory | (required*) |
| `--input-image` | Single input image | |
| `--output-image` | Single output image | |
| `--prompt`, `-p` | Edit prompt | (required) |
| `--model` | Model (auto-detects family) | `black-forest-labs/FLUX.2-klein-4B` |
| `--weights` | Fine-tuned weights | (none) |
| `--lora` | LoRA adapter (repeatable, `:WEIGHT`) | (none) |
| `--device` | `auto`, `cpu`, or `cuda` | `auto` |
| `--cpu-offload` | Force CPU offload | `false` |
| `--steps` | Inference steps | `40` |
| `--cfg-scale` | Guidance scale | `1.0` |
| `--true-cfg-scale` | True CFG (Qwen only) | `4.0` |
| `--negative-prompt` | Negative prompt | `" "` |
| `--num-images` | Images per input | `1` |
| `--seed` | Random seed | (random) |
| `--gguf` | GGUF path/URL for quantized loading | (none) |
| `--strength` | Img2img strength (0.0-1.0) | `0.7` |
| `--output-format` | `png`, `jpg`, `webp` | `png` |

## Examples

```bash
# FLUX.2 klein (fast, 8 GB VRAM)
datasety synthetic -i ./dataset -o ./synthetic \
    --prompt "add a winter hat" --steps 4 --cfg-scale 2.5

# Single image
datasety synthetic --input-image photo.jpg --output-image edited.png \
    --prompt "add sunglasses" --steps 4

# Qwen with LoRA
datasety synthetic -i ./dataset -o ./synthetic \
    --model "Qwen/Qwen-Image-Edit-2511" \
    --lora "adapter.safetensors:0.8" \
    --prompt "add a winter hat" --steps 40

# GGUF quantized model
datasety synthetic -i ./dataset -o ./synthetic \
    --model "black-forest-labs/FLUX.1-Kontext-dev" \
    --gguf "path/to/model.gguf" \
    --prompt "add a winter hat" --cpu-offload
```
