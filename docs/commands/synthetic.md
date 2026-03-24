# synthetic

Generate synthetic variations of images using image editing models.

## Usage

```bash
datasety synthetic --input ./images --output ./synthetic --prompt "add a winter hat"
```

## Supported Models

| Model Family                   | Example Model ID                           | Key Params                              |
| ------------------------------ | ------------------------------------------ | --------------------------------------- |
| **FLUX.2 klein FP8** (default) | `black-forest-labs/FLUX.2-klein-4b-fp8`    | `--strength`, steps=4, ~5 GB            |
| **FLUX.2 klein 9B FP8**        | `black-forest-labs/FLUX.2-klein-9b-fp8`    | Better face fidelity, ~10 GB            |
| **FLUX.2 klein BF16**          | `black-forest-labs/FLUX.2-klein-4B`        | `--strength`, steps=4                   |
| **FLUX.2 klein 9B KV**         | `black-forest-labs/FLUX.2-klein-9b-kv`     | KV-cache pipeline, no guidance, ~29 GB  |
| **FLUX.2 dev**                 | `black-forest-labs/FLUX.2-dev`             | `--strength`, steps=28                  |
| **Qwen**                       | `Qwen/Qwen-Image-Edit-2511`                | `--true-cfg-scale`, steps=40            |
| **FireRed**                    | `FireRedTeam/FireRed-Image-Edit-1.0`       | `--true-cfg-scale`, steps=40            |
| **FLUX Kontext**               | `black-forest-labs/FLUX.1-Kontext-dev`     | steps=28                                |
| **LongCat**                    | `meituan-longcat/LongCat-Image-Edit-Turbo` | steps=50                                |
| **SDXL**                       | `stabilityai/stable-diffusion-xl-*`        | `--strength`, steps=30                  |
| **HunyuanImage**               | `tencent/HunyuanImage-3.0`                 | steps=50                                |

### Cloud API models (`--image-api`)

When using `--image-api`, pass any OpenRouter (or OpenAI-compatible) image-generation model via `--model`:

| Model ID | Type | Notes |
| -------- | ---- | ----- |
| `google/gemini-2.5-flash-image` | text+image output | Supports i2i; use `--api-aspect-ratio` |
| `black-forest-labs/flux.2-flex` | image output | t2i and i2i |
| `black-forest-labs/flux.2-pro` | image output | t2i |
| `sourceful/riverflow-v2-fast` | image output | i2i with super-resolution references |

## Options

| Option              | Description                              | Default                                 |
| ------------------- | ---------------------------------------- | --------------------------------------- |
| `--input`, `-i`     | Input directory                          | (required\*)                            |
| `--output`, `-o`    | Output directory                         | (required\*)                            |
| `--input-image`     | Single input image                       |                                         |
| `--output-image`    | Single output image                      |                                         |
| `--prompt`, `-p`    | Edit prompt                              | (required)                              |
| `--model`           | Model (auto-detects family or API model) | `black-forest-labs/FLUX.2-klein-4b-fp8` |
| `--image-api`       | Use OpenAI-compatible API for generation | `false`                                 |
| `--api-aspect-ratio` | Aspect ratio for `--image-api` (e.g. `16:9`, `1:1`, `9:16`) | auto |
| `--api-image-size`  | Resolution for `--image-api`: `0.5K`, `1K`, `2K`, `4K`      | `1K` |
| `--weights`         | Fine-tuned weights                       | (none)                                  |
| `--lora`            | LoRA adapter (repeatable, `:WEIGHT`)     | (none)                                  |
| `--device`          | `auto`, `cpu`, `cuda`, or `mps`          | `auto`                                  |
| `--cpu-offload`     | Force CPU offload                        | `false`                                 |
| `--steps`           | Inference steps                          | `4`                                     |
| `--cfg-scale`       | Guidance scale                           | `2.5`                                   |
| `--true-cfg-scale`  | True CFG (Qwen only)                     | `4.0`                                   |
| `--negative-prompt` | Negative prompt                          | `" "`                                   |
| `--num-images`      | Images per input                         | `1`                                     |
| `--seed`            | Random seed                              | (random)                                |
| `--gguf`            | GGUF path/URL for quantized loading      | (none)                                  |
| `--strength`        | Img2img strength (0.0-1.0)               | `0.7`                                   |
| `--recursive`, `-R` | Search input directory recursively       | `false`                                 |
| `--output-format`   | `png`, `jpg`, `webp`                     | `png`                                   |
| `--skip-existing`   | Skip images with existing output         | `false`                                 |
| `--batch-size`      | Flush GPU memory every N images          | `0` (off)                               |
| `--progress`        | Show tqdm progress bar                   | `false`                                 |
| `--dry-run`         | Preview without loading models           | `false`                                 |

## Examples

```bash
# FLUX.2 klein (fast, 8 GB VRAM)
datasety synthetic -i ./dataset -o ./synthetic \
    --prompt "add a winter hat" --steps 4 --cfg-scale 2.5

# Single image
datasety synthetic --input-image photo.jpg --output-image edited.png \
    --prompt "add sunglasses" --steps 4

# Cloud API — FLUX.2-flex image-to-image
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  datasety synthetic -i ./images -o ./synthetic \
  --prompt "add a winter hat" --image-api --model black-forest-labs/flux.2-flex \
  --api-aspect-ratio 1:1

# Cloud API — Gemini 2.5 Flash (text+image, supports image-to-image)
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  datasety synthetic -i ./images -o ./synthetic \
  --prompt "transform into oil painting style" \
  --model google/gemini-2.5-flash-image --image-api \
  --api-aspect-ratio 3:4 --api-image-size 2K

# Qwen with LoRA
datasety synthetic -i ./dataset -o ./synthetic \
    --model "Qwen/Qwen-Image-Edit-2511" \
    --lora "adapter.safetensors:0.8" \
    --prompt "add a winter hat" --steps 40

# FLUX.2-klein-9b-kv (KV-cache pipeline, ~29 GB VRAM, no guidance_scale)
datasety synthetic -i ./dataset -o ./synthetic \
    --model "black-forest-labs/FLUX.2-klein-9b-kv" \
    --prompt "add sunglasses" --steps 4

# GGUF quantized model
datasety synthetic -i ./dataset -o ./synthetic \
    --model "black-forest-labs/FLUX.1-Kontext-dev" \
    --gguf "path/to/model.gguf" \
    --prompt "add a winter hat" --cpu-offload

# Dry run (preview without processing)
datasety synthetic -i ./images -o ./synthetic --prompt "add a hat" --dry-run
```
