# character

Generate identity-preserving character datasets from reference face images using LLM-generated prompts and IP-Adapter.

## Usage

```bash
datasety character --reference face.jpg --output ./dataset --llm-ollama llama3.2
```

## LLM Backends

| Flag                 | Backend               | Requirements             |
| -------------------- | --------------------- | ------------------------ |
| `--llm-api`          | OpenAI-compatible API | `OPENAI_API_KEY` env var |
| `--llm-ollama MODEL` | Local Ollama          | Ollama running locally   |
| `--llm-gguf PATH`    | Local GGUF model      | `llama-cpp-python`       |
| `--llm-model REPO`   | HuggingFace model     | `transformers`           |

## Options

| Option                    | Description                   | Default                             |
| ------------------------- | ----------------------------- | ----------------------------------- |
| `--reference`, `-r`       | Reference face image(s)       | (required)                          |
| `--output`, `-o`          | Output directory              | (required)                          |
| `--num-images`, `-n`      | Number of images              | `10`                                |
| `--model`                 | Base model for generation     | `black-forest-labs/FLUX.2-klein-4B` |
| `--ip-adapter`            | IP-Adapter model              | (auto-detected)                     |
| `--ip-adapter-scale`      | Conditioning strength 0.0-1.0 | `0.6`                               |
| `--character-description` | Text description of character |                                     |
| `--style`                 | Style guidance                |                                     |
| `--prompts-only`          | Only generate prompts         | `false`                             |
| `--prompts-file`          | Load prompts from file        |                                     |
| `--device`                | `auto`, `cpu`, or `cuda`      | `auto`                              |
| `--steps`                 | Inference steps               | `4`                                 |
| `--cfg-scale`             | Guidance scale                | `2.5`                               |
| `--seed`                  | Random seed                   | (random)                            |
| `--output-format`         | `png`, `jpg`, `webp`          | `png`                               |

## Examples

```bash
# Generate with OpenAI API
datasety character -r face1.jpg face2.jpg -o ./dataset \
    --llm-api --num-images 20 --style "photorealistic"

# Preview prompts only
datasety character -r face.jpg -o ./dataset --llm-ollama llama3.2 --prompts-only

# Use pre-written prompts
datasety character -r face.jpg -o ./dataset --prompts-file prompts.txt
```
