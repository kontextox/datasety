# caption

Generate captions for images using Florence-2 or OpenAI-compatible vision APIs.

## Usage

```bash
# Florence-2 (default: base model)
datasety caption --input ./images --output ./captions

# Vision API
datasety caption --input ./images --output ./captions --llm-api --model gpt-5-nano
```

## Options

| Option               | Description                        | Default                   |
| -------------------- | ---------------------------------- | ------------------------- |
| `--input`, `-i`      | Input directory                    | (required\*)              |
| `--output`, `-o`     | Output directory for .txt files    | (required\*)              |
| `--input-image`      | Single input image                 |                           |
| `--output-caption`   | Single output .txt path            |                           |
| `--device`           | `auto`, `cpu`, `cuda`, or `mps`    | `auto`                    |
| `--trigger-word`     | Text to prepend to captions        | (none)                    |
| `--prompt`           | Florence-2 task prompt             | `<MORE_DETAILED_CAPTION>` |
| `--model`            | HF model or API model ID           | (none)                    |
| `--num-beams`        | Beam search width (1 = greedy)     | `3`                       |
| `--florence-2-base`  | Use base model (0.23B, faster)     | (default)                 |
| `--florence-2-large` | Use large model (0.77B, better)    |                           |
| `--llm-api`          | Use OpenAI-compatible vision API   |                           |
| `--max-tokens`       | Max response tokens (API mode)     | `300`                     |
| `--temperature`      | Temperature (API mode)             | `0.3`                     |
| `--skip-existing`    | Skip images with existing .txt     | `false`                   |
| `--append`           | Append text to existing captions   |                           |
| `--prepend`          | Prepend text to existing captions  |                           |
| `--recursive`, `-R`  | Search input directory recursively | `false`                   |
| `--progress`         | Show tqdm progress bar             | `false`                   |
| `--dry-run`          | Preview without writing files      | `false`                   |

## Environment Variables

| Variable          | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `OPENAI_API_KEY`  | API key (required for `--llm-api`)                                 |
| `OPENAI_BASE_URL` | Custom API endpoint                                                |
| `OPENAI_API_BASE` | Legacy fallback for base URL                                       |
| `OPENAI_MODEL`    | Default model when `--model` not specified (default: `gpt-5-nano`) |

## Florence-2 Prompts

| Prompt                    | Description             |
| ------------------------- | ----------------------- |
| `<CAPTION>`               | Brief caption           |
| `<DETAILED_CAPTION>`      | Detailed caption        |
| `<MORE_DETAILED_CAPTION>` | Most detailed (default) |

## Examples

```bash
# Florence-2 base with trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "photo of sks person,"

# Florence-2 large
datasety caption -i ./dataset -o ./dataset --florence-2-large --device cuda

# OpenAI vision API
datasety caption -i ./dataset -o ./dataset --llm-api --model gpt-5-nano

# Custom provider via env vars
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_API_KEY=your-key \
datasety caption -i ./dataset -o ./dataset --llm-api --model x-ai/grok-4.1-fast
```
