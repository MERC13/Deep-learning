import transformers
import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").cuda()

for param in base_model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(base_model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Prepare few-shot labeled examples
data = Dataset.from_dict({
    "text": ["Use more numbers", "What’s the weather?", ...],
    "label": [1, 0, ...]
})
clf = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
args = TrainingArguments(output_dir="clf_out", per_device_train_batch_size=8, num_train_epochs=3)
trainer = Trainer(model=clf, args=args, train_dataset=data)
trainer.train()

clf.eval()
clf = clf.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

clf_tokenizer = ClfTokenizer.from_pretrained("distilbert-base-uncased")
def is_feedback(msg: str) -> bool:
    inputs = clf_tokenizer(msg, return_tensors="pt").to("cuda")
    logits = clf(**inputs).logits
    return logits.argmax(-1).item() == 1

def online_update(user_msg: str, prompt: str, corrected: str=None):
    # 1. Generate response
    in_tok = tokenizer(prompt, return_tensors="pt").to("cuda")
    gen = model.generate(**in_tok)

    # 2. Detect feedback
    feedback_flag = is_feedback(user_msg)
    lr = 1e-5 if feedback_flag else 1e-7
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # 3. Prepare training inputs
    if feedback_flag and corrected:
        inp = tokenizer(f"{prompt} [FEEDBACK] {user_msg}", return_tensors="pt").to("cuda")
        lbl = tokenizer(corrected, return_tensors="pt").input_ids.to("cuda")
    else:
        inp, lbl = in_tok.input_ids, gen

    # 4. Update adapters
    model.train()
    outputs = model(input_ids=inp, labels=lbl)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return tokenizer.decode(gen[0], skip_special_tokens=True)


while True:
    user_msg = input("User: ")
    if user_msg.startswith("exit"):
        break
    # Optionally obtain a corrected target if feedback_flag is True
    corrected = None
    if is_feedback(user_msg):
        corrected = input("Corrected target: ")
    response = online_update(user_msg, user_msg, corrected)
    print("LLM:", response)
