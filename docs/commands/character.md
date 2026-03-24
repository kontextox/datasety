# character

Generate character datasets using LLM-generated prompts + text-to-image (FLUX.2-klein local or cloud API).

## Usage

```bash
datasety character --output ./dataset --llm-ollama qwen3.5:4b
```

## Notes

- **Thinking models** (e.g. `Qwen3.5-4B`, `Qwen3-4B`): These models expose chain-of-thought reasoning as plain text. datasety filters meta-commentary lines automatically and trims the output to exactly `--num-images` prompts. For cleanest results use an API backend or a non-thinking instruction model.
- **FLUX.2-klein-9b-kv**: When using `--model black-forest-labs/FLUX.2-klein-9b-kv`, the KV-cache pipeline is selected automatically and `--cfg-scale` is ignored (not supported by this pipeline).

## LLM Backends (for prompt generation)

| Flag                 | Backend               | Requirements             |
| -------------------- | --------------------- | ------------------------ |
| `--llm-api`          | OpenAI-compatible API | `OPENAI_API_KEY` env var |
| `--llm-ollama MODEL` | Local Ollama          | Ollama running locally   |
| `--llm-gguf PATH`    | Local GGUF model      | `llama-cpp-python`       |
| `--llm-model REPO`   | HuggingFace model     | `transformers`           |

## Image Generation

By default, images are generated locally using FLUX.2-klein (requires GPU + PyTorch). Use `--image-api` to generate images via an OpenAI-compatible API instead (no GPU required).

## Options

| Option                    | Description                                        | Default                                 |
| ------------------------- | -------------------------------------------------- | --------------------------------------- |
| `--reference`, `-r`       | Reference face image(s) (optional, prompt context) |                                         |
| `--output`, `-o`          | Output directory                                   | (required)                              |
| `--num-images`, `-n`      | Number of images                                   | `10`                                    |
| `--model`                 | Model for generation (local HF or API model ID)    | `black-forest-labs/FLUX.2-klein-4b-fp8` |
| `--gguf`                  | GGUF path/URL for quantized loading                |                                         |
| `--image-api`             | Use OpenAI-compatible API for image generation     | `false`                                 |
| `--api-aspect-ratio`      | Aspect ratio for `--image-api` (e.g. `9:16`, `1:1`) | derived from `--width`/`--height`     |
| `--api-image-size`        | Resolution for `--image-api`: `0.5K`, `1K`, `2K`, `4K` |                                    |
| `--character-description` | Text description of character                      |                                         |
| `--style`                 | Style guidance                                     |                                         |
| `--prompts-only`          | Only generate prompts                              | `false`                                 |
| `--prompts-file`          | Load prompts from file                             |                                         |
| `--device`                | `auto`, `cpu`, `cuda`, or `mps`                    | `auto`                                  |
| `--steps`                 | Inference steps                                    | `4`                                     |
| `--cfg-scale`             | Guidance scale                                     | `4.0`                                   |
| `--seed`                  | Random seed                                        | (random)                                |
| `--height`                | Output image height                                | `1024`                                  |
| `--width`                 | Output image width                                 | `1024`                                  |
| `--output-format`         | `png`, `jpg`, `webp`                               | `png`                                   |
| `--batch-size`            | Flush GPU memory every N images                    | `0` (off)                               |
| `--dry-run`               | Preview prompts without generating images          | `false`                                 |

## Examples

```bash
# Local pipeline with Ollama prompts
datasety character -o ./dataset --llm-ollama qwen3.5:4b --num-images 20

# Cloud API for images — FLUX.2-flex (no GPU needed)
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  datasety character -o ./dataset --prompts-file prompts.txt \
  --image-api --model black-forest-labs/flux.2-flex --api-aspect-ratio 2:3

# Cloud API for images — Gemini 2.5 Flash
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  datasety character -o ./dataset \
  --character-description "young warrior in samurai armor" --style "detailed anime" \
  --llm-api --image-api --model google/gemini-2.5-flash-image \
  --api-aspect-ratio 9:16 --num-images 10

# Preview prompts only
datasety character -o ./dataset --llm-api --prompts-only

# Use pre-written prompts with local pipeline
datasety character -o ./dataset --prompts-file prompts.txt

# Dry run
datasety character -o ./dataset --prompts-file prompts.txt --dry-run
```
