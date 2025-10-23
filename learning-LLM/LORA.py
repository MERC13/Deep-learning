"""
Self-improving TinyLlama with LoRA on a single GPU.

Key features:
- Loads TinyLlama chat model with optional 4-bit/8-bit quantization for single-GPU training.
- Applies PEFT LoRA adapters to a targeted set of layers.
- Interactive chat loop that collects feedback and performs periodic online SFT updates.
- Masks prompt tokens so only assistant response is trained (standard SFT objective).
- Saves and reloads LoRA adapters between runs.

Run examples:
  python LORA.py --mode chat
  python LORA.py --mode loop --update-every 4
  python LORA.py --mode train --data data/training.jsonl --epochs 1
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


DEFAULT_MODEL = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./adapters-tinyllama")


def build_bnb_config(load_in_4bit: bool = True, load_in_8bit: bool = False) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config if available; return None on failure.
    Falls back gracefully on Windows where bitsandbytes may be unavailable.
    """
    if not load_in_4bit and not load_in_8bit:
        return None
    try:
        # bnb_4bit_compute_dtype must be a torch.dtype
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit and not load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except Exception:
        # e.g., Windows without compatible bitsandbytes wheel
        return None


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    adapter_dir: str = ADAPTER_DIR,
    use_4bit: bool = True,
    use_8bit: bool = False,
) -> tuple:
    """Load base model with quantization if possible, attach LoRA if present."""
    bnb_config = build_bnb_config(load_in_4bit=use_4bit, load_in_8bit=use_8bit)

    kwargs: Dict[str, Any] = {}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
    else:
        # No quantization: still place on GPU if available
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Ensure pad token exists for Trainer padding
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Gradient checkpointing improves memory
    model.gradient_checkpointing_enable()
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # Safe to ignore if not using k-bit or unsupported platform
        pass

    # If adapters exist, load them; else attach fresh LoRA config
    if os.path.isdir(adapter_dir) and any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(adapter_dir)):
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # Common LLaMA-family target modules
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    # Freeze base params (LoRA layers remain trainable)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    return model, tokenizer


def apply_chat_template(tokenizer, user_text: str, system_text: Optional[str] = None) -> Dict[str, Any]:
    """Return tokenized prompt-only (for masking) and prompt+assistant placeholders.

    We build two encodings:
    - prompt_ids: tokens up to the assistant turn (masked during training)
    - full_ids: tokens including the assistant response placeholder + response text
    """
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    # Tokens up to assistant start (no assistant content)
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # ends with assistant header
            tokenize=True,
            return_tensors=None,
        )
    except Exception:
        # Fallback simple template
        prompt_str = (f"System: {system_text}\n" if system_text else "") + f"User: {user_text}\nAssistant: "
        prompt_ids = tokenizer(prompt_str, add_special_tokens=True)["input_ids"]

    return {"prompt_ids": prompt_ids}


def build_sft_dataset(
    tokenizer,
    samples: List[Dict[str, str]],
    system_text: Optional[str] = None,
    max_length: int = 1024,
) -> Dataset:
    """Create a dataset with input_ids and labels where labels mask the prompt tokens."""
    input_ids_list = []
    labels_list = []

    for ex in samples:
        prompt = ex["prompt"].strip()
        response = ex["response"].strip()

        enc = apply_chat_template(tokenizer, prompt, system_text)
        prompt_ids = enc["prompt_ids"]
        resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        full_ids = prompt_ids + resp_ids
        full_ids = full_ids[:max_length]

        # Labels: mask the prompt portion with -100, keep response tokens
        labels = [-100] * min(len(prompt_ids), len(full_ids))
        if len(full_ids) > len(prompt_ids):
            labels += full_ids[len(prompt_ids):]
        labels = labels[:max_length]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    ds = Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})
    return ds


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad input_ids and labels to the longest sequence in the batch
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def train_once(
    model,
    tokenizer,
    train_samples: List[Dict[str, str]],
    system_text: Optional[str],
    output_dir: str,
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 2,
):
    ds = build_sft_dataset(tokenizer, train_samples, system_text)
    collator = DataCollatorForCausalLMWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=1,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        save_steps=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    # Save LoRA adapter weights only
    model.save_pretrained(output_dir)


def generate_response(model, tokenizer, user_text: str, system_text: Optional[str] = None, max_new_tokens: int = 256) -> str:
    model.eval()
    enc = apply_chat_template(tokenizer, user_text, system_text)
    input_ids = torch.tensor(enc["prompt_ids"], dtype=torch.long).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated part
    gen_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def interactive_chat(model, tokenizer, system_text: Optional[str] = None):
    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = generate_response(model, tokenizer, user_input, system_text)
        print(f"Assistant: {reply}\n")


def self_improve_loop(
    model,
    tokenizer,
    adapter_dir: str,
    system_text: Optional[str] = None,
    update_every: int = 4,
):
    """Chat, collect feedback, and fine-tune after N positive samples."""
    print("Enter feedback after each answer as -1, 0, or 1. Type 'exit' to quit.")
    buffer: List[Dict[str, str]] = []
    total_updates = 0
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        reply = generate_response(model, tokenizer, user_input, system_text)
        print(f"Assistant: {reply}")
        try:
            fb = int(input("Feedback (-1,0,1): ").strip())
        except Exception:
            fb = 0
        if fb > 0:
            buffer.append({"prompt": user_input, "response": reply})

        if len(buffer) >= update_every:
            total_updates += 1
            print(f"\n[Training] Updating on {len(buffer)} samples (update #{total_updates})...\n")
            train_once(
                model=model,
                tokenizer=tokenizer,
                train_samples=buffer,
                system_text=system_text,
                output_dir=adapter_dir,
                epochs=1,
                lr=1e-4,
                batch_size=2,
            )
            buffer.clear()
            print("[Training] Done. Adapters saved.\n")


def load_jsonl(path: str) -> List[Dict[str, str]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chat", "loop", "train"], default="chat")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters", default=ADAPTER_DIR)
    parser.add_argument("--system", default=None, help="Optional system prompt to steer behavior")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--data", default=None, help="JSONL with {prompt,response}")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        adapter_dir=args.adapters,
        use_4bit=not args.no_4bit,
        use_8bit=args.use_8bit,
    )

    if args.mode == "chat":
        interactive_chat(model, tokenizer, args.system)
    elif args.mode == "loop":
        self_improve_loop(model, tokenizer, args.adapters, args.system, args.update_every)
    elif args.mode == "train":
        if not args.data or not os.path.isfile(args.data):
            raise SystemExit("--data JSONL file required for train mode")
        samples = load_jsonl(args.data)
        train_once(
            model=model,
            tokenizer=tokenizer,
            train_samples=samples,
            system_text=args.system,
            output_dir=args.adapters,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
