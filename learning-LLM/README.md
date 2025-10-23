# Self-Improving TinyLlama (LoRA, single GPU)

This project fine-tunes a small chat model (TinyLlama-1.1B) with LoRA adapters on a single GPU and includes a self-improvement loop that collects your feedback and performs quick online updates.

## What it does
- Loads TinyLlama in 4-bit (QLoRA) or 8/16-bit depending on your setup
- Adds LoRA adapters to attention/MLP layers for efficient training
- Interactive chat that asks for feedback (-1/0/1)
- Trains on positively-rated responses and saves adapters
- Resumes with saved adapters automatically

## Install

Note: On Windows, `bitsandbytes` may not install. If it fails, run with `--no-4bit`.

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If `bitsandbytes` fails to install on Windows, you can still run:

```
python LORA.py --mode chat --no-4bit
```

## Run

Chat only:
```
python LORA.py --mode chat
```

Self-improvement loop (fine-tunes after N positive samples):
```
python LORA.py --mode loop --update-every 4
```

Train from a JSONL file with pairs `{ "prompt": str, "response": str }`:
```
python LORA.py --mode train --data data/training.jsonl --epochs 1 --batch-size 2
```

You can set a system prompt to steer behavior:
```
python LORA.py --mode chat --system "You are a concise, helpful assistant."
```

Change model or adapter directory via env or flags:
```
set MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set ADAPTER_DIR=adapters-tinyllama
python LORA.py --mode loop
```

## How it works
- Uses `tokenizer.apply_chat_template` when available to format chat turns.
- During training, masks the prompt tokens with -100 so only assistant tokens contribute to the loss (standard SFT).
- Saves just LoRA adapters (few MB) to `./adapters-tinyllama` by default.

## Tips
- If you hit CUDA OOM, reduce `--batch-size`, increase `gradient_accumulation_steps` in code, or pass `--no-4bit` if 4-bit unsupported.
- Consider a smaller model if needed (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0` is already compact, alternatives include `Qwen2-1.5B-Instruct`, `Phi-2` for non-commercial).

## Troubleshooting
- `bitsandbytes` install error on Windows: run with `--no-4bit`.
- `apply_chat_template` missing: the code falls back to a simple prompt format.
- If generation is slow, try `--use-8bit` or run without quantization if your GPU has enough memory.
