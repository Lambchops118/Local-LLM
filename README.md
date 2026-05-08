# Local LLM Chat for Apple Silicon

This project is a local-only Python CLI chat app for running chat-oriented Hugging Face models on a Mac with Apple Silicon. It now uses a pluggable backend architecture:

- `gemma_3_4b_it`: MLX + 4-bit text-tuned Gemma 3 4B
- `gemma_3_12b_it`: MLX + 4-bit text-tuned Gemma 3 12B
- `gemma_3_27b_it`: MLX + 4-bit text-tuned Gemma 3 27B
- `gemma_9b_it`: MLX + 4-bit quantized Gemma 2 9B for the fastest local Apple Silicon path
- `gemma_2b_it`: the existing Transformers + PyTorch path for Gemma 2 2B
- `gemma_9b_it_transformers`: a legacy 9B FP16 fallback profile
- `llama3_8b`: Transformers + PyTorch path for Meta Llama 3 8B Instruct
- `mistral_7b`: Transformers + PyTorch path for Mistral 7B Instruct
- `phi_3`: Transformers + PyTorch path for Phi-3 Mini 4K Instruct
- `mixtral_8x7b`: Transformers + PyTorch path for Mixtral 8x7B Instruct
- `gemma_4_e4b_it`: Transformers + PyTorch path for Gemma 4 E4B Instruct
- `gemma_4_26b_a4b_it`: Transformers + PyTorch path for Gemma 4 26B A4B Instruct

Multi-turn chat behavior is preserved, `/reset` now clears both chat history and backend-side prompt cache state, and all inference stays fully local once model files are on disk.

Gemma 3 in this CLI is currently wired up through the `mlx-lm` text models, so these new profiles are usable for text chat only. The upstream Google Gemma 3 checkpoints are multimodal, but this app does not yet accept image input.

## What Was Slowing Gemma 9B Down

The original 9B path used `transformers` + PyTorch on `mps` with the full FP16 checkpoint. That leaves a few major bottlenecks on an M3:

- about 18 GB of FP16 weights before runtime overhead
- full-prompt recomputation on every turn
- no persistent KV/prompt cache reuse between turns
- a generic backend path that is correct, but not the fastest Apple Silicon-native inference stack

For Gemma 9B on macOS, the best practical local path is usually MLX with an MLX-converted 4-bit checkpoint.

## What Changed

- Added a backend registry so models are selected by profile, not hard-coded loader logic.
- Split the runtime layer into:
  - `transformers` backend for the existing PyTorch/MPS flow
  - `mlx` backend for Apple Silicon-optimized inference
- Repointed the `gemma_9b_it` profile to `mlx-community/gemma-2-9b-it-4bit`.
- Added incremental multi-turn prompt-cache reuse in the MLX backend so follow-up turns do not re-prefill the entire conversation.
- Enabled configurable KV-cache quantization for the MLX profile to reduce longer-chat memory pressure.
- Kept Gemma 2B support intact and added `gemma_9b_it_transformers` as a compatibility fallback.

## Realistic Performance Target

A warm 4-bit MLX Gemma 9B setup on an M3 can be much faster than the old FP16 Transformers path, especially on short follow-ups because prompt prefill is reused across turns.

That said, around **300 ms for a complete 9B answer** is usually not realistic for normal multi-sentence generations. Around **300 ms to first visible output** or **very short warm-cache replies** can be realistic; full answers will still commonly land in the sub-second to multi-second range depending on prompt length, response length, and memory bandwidth.

The biggest wins here are:

- lower memory footprint
- better Apple Silicon kernel performance through MLX
- much less repeated prompt work on follow-up turns

## Project Layout

```text
.
├── config/
│   └── models.json
├── requirements.txt
├── pyproject.toml
├── README.md
└── src/
    └── local_llm_chat/
        ├── backends/
        │   ├── base.py
        │   ├── mlx_backend.py
        │   └── transformers_backend.py
        ├── cli.py
        ├── config.py
        ├── modeling.py
        └── session.py
```

## Requirements

- macOS on Apple Silicon
- Python 3.9+
- enough unified memory for the selected model
- a Hugging Face account for Gemma-licensed model downloads
- Python 3.10+ and `transformers>=5.5.0` if you want to use the Gemma 4 profiles

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install -e .
   ```

3. Accept the Gemma license and authenticate locally if needed:

   ```bash
   huggingface-cli login
   ```

4. Run the app:

   ```bash
   local-llm-chat
   ```

## Profiles

The shipped config currently defaults to `gemma_9b_it`, and you can switch to any profile at launch:

```bash
local-llm-chat --model-profile gemma_3_4b_it
local-llm-chat --model-profile gemma_3_12b_it
local-llm-chat --model-profile gemma_3_27b_it
local-llm-chat --model-profile gemma_9b_it
local-llm-chat --model-profile llama3_8b
local-llm-chat --model-profile mistral_7b
local-llm-chat --model-profile phi_3
local-llm-chat --model-profile mixtral_8x7b
local-llm-chat --model-profile gemma_4_e4b_it
local-llm-chat --model-profile gemma_4_26b_a4b_it
```

Available profiles:

- `gemma_3_4b_it`: MLX 4-bit Gemma 3 4B text chat
- `gemma_3_12b_it`: MLX 4-bit Gemma 3 12B text chat
- `gemma_3_27b_it`: MLX 4-bit Gemma 3 27B text chat
- `gemma_9b_it`: MLX 4-bit Gemma 2 9B
- `gemma_2b_it`: Transformers Gemma 2 2B
- `gemma_9b_it_transformers`: legacy FP16 9B fallback
- `llama3_8b`: Transformers Meta Llama 3 8B Instruct
- `mistral_7b`: Transformers Mistral 7B Instruct
- `phi_3`: Transformers Microsoft Phi-3 Mini 4K Instruct
- `mixtral_8x7b`: Transformers Mixtral 8x7B Instruct
- `gemma_4_e4b_it`: Transformers Gemma 4 E4B Instruct
- `gemma_4_26b_a4b_it`: Transformers Gemma 4 26B A4B Instruct

## Example Usage

```text
$ local-llm-chat --model-profile gemma_3_4b_it
Local LLM Chat
Google Gemma 3 4B Instruct MLX 4-bit (mlx-community/gemma-3-text-4b-it-4bit) on apple-silicon via mlx using 4-bit quantized weights
Commands: /help, /reset, /history, /exit
Enter a message to begin.

You> Explain what a mutex does in one paragraph.
Assistant> A mutex is a synchronization primitive that allows only one thread at a time to enter a critical section...
```

## Configuration

Profiles are defined in [`config/models.json`](/Users/jacksal1/Desktop/Local LLM/config/models.json).

Key fields:

- `backend`: `transformers` or `mlx`
- `model_id`: Hugging Face repo or local model path
- `backend_options`: backend-specific tuning knobs

The Gemma 3 MLX profiles use the same prompt-cache and KV-cache tuning approach as the optimized Gemma 2 9B profile:

```json
{
  "backend": "mlx",
  "model_id": "mlx-community/gemma-3-text-4b-it-4bit",
  "backend_options": {
    "prefill_step_size": 1024,
    "kv_bits": 4,
    "kv_group_size": 64,
    "quantized_kv_start": 1024
  }
}
```

## CLI Options

```bash
local-llm-chat --help
local-llm-chat --model-profile gemma_3_4b_it
local-llm-chat --model-profile gemma_3_12b_it
local-llm-chat --model-profile gemma_3_27b_it
local-llm-chat --model-profile gemma_9b_it
local-llm-chat --model-profile gemma_2b_it
local-llm-chat --model-profile llama3_8b
local-llm-chat --model-profile mistral_7b
local-llm-chat --model-profile phi_3
local-llm-chat --model-profile mixtral_8x7b
local-llm-chat --model-profile gemma_4_e4b_it
local-llm-chat --model-profile gemma_4_26b_a4b_it
local-llm-chat --local-files-only
local-llm-chat --offline
local-llm-chat --max-new-tokens 128
local-llm-chat --temperature 0.2 --top-p 0.9
```

## Managing Downloaded Models

You can inspect and delete locally cached model downloads without starting the chat UI:

```bash
local-llm-chat --list-downloaded-models
local-llm-chat --list-models
local-llm-chat --delete-downloaded-model gemma_3_27b_it
local-llm-chat --delete-downloaded-model mlx-community/gemma-3-text-27b-it-4bit --yes
```

- `--list-downloaded-models` shows configured profiles and whether each model is `downloaded`, `deleted`, or `not downloaded`.
- `--list-models` is an alias for `--list-downloaded-models`.
- `--delete-downloaded-model` accepts either a profile name or an exact Hugging Face model id.
- `--yes` skips the deletion confirmation prompt.

## Local-Only Behavior

- Inference is local after model files exist in the local Hugging Face cache.
- The first load of an uncached model may download weights from Hugging Face.
- `--local-files-only` and `--offline` still work for both backends.

## Notes

- MLX is the preferred backend for large Apple Silicon-local Gemma models.
- Gemma 3 support here uses the `mlx-community/gemma-3-text-*-it-4bit` conversions because the CLI currently handles text chat, not image input.
- The added Llama, Mistral, Phi, and Mixtral profiles use the generic `transformers` backend and rely on each model repo's chat template support.
- The Gemma 4 profiles currently use the generic `transformers` backend for text chat. The upstream models are multimodal, but this CLI does not yet pass image or audio inputs.
- Gemma 4 support depends on Python 3.10+ and a newer Hugging Face `transformers` build than older Gemma 2 and Llama setups. If a Gemma 4 profile errors with an unrecognized `gemma4` architecture, recreate the venv on Python 3.10+ and then upgrade inside it with `pip install --upgrade 'transformers>=5.5.0'`.
- The MLX backend reuses prompt cache across turns, so short follow-up messages benefit more than cold-start prompts.
- If you want the old PyTorch/MPS path for 9B, use `--model-profile gemma_9b_it_transformers`.
- Longer chats still consume more memory and latency, especially if you keep unlimited context.

## Troubleshooting

- If the MLX backend cannot find the quantized model locally, run once without `--local-files-only` to download it.
- If Gemma access fails, accept the model terms on Hugging Face and run `huggingface-cli login`.
- If Llama or another licensed model fails to download, accept its Hugging Face terms and authenticate with `huggingface-cli login`.
- If you hit memory pressure, reduce `--max-new-tokens`, reset the chat, or switch to a smaller profile such as `gemma_3_4b_it` or `gemma_2b_it`.
