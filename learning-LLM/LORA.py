# pip install transformers peft datasets bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "meta-llama/Llama-2-7b-hf"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization
    bnb_4bit_compute_dtype="bfloat16"
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": "cuda"}
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for k-bit training (low-bit fine-tuning)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Configure and add LoRA adapters
lora_config = LoraConfig(
    r=8,                   # LoRA rank (controls adapter size)
    lora_alpha=16,         # Scaling factor for the update
    lora_dropout=0.1,      # Dropout for regularization
    bias="none",           # Only train LoRA params
    task_type="CAUSAL_LM", # For chatbots
    target_modules=["q_proj", "v_proj"]  # LLaMA attention layers
)

model = get_peft_model(model, lora_config)

# Freeze all parameters except LoRA adapters
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

def chat(model, tokenizer):
    while True:
        user_input = input("User: ")
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", response)
        feedback = input("Rate response (-1, 0, 1): ")
        yield user_input, response, int(feedback)

from transformers import Trainer, TrainingArguments
from datasets import Dataset

def online_update(model, tokenizer, feedback_data):
    # Prepare dataset
    dataset = Dataset.from_dict(feedback_data)
    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        output_dir="./lora_online",
        report_to=None
    )
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()

feedback_data = {"input_ids": [], "labels": []}

for user_input, response, feedback in chat(model, tokenizer):
    if feedback == 1:
        # Positive feedback: reinforce response
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        labels = tokenizer(response, return_tensors="pt")["input_ids"].to(model.device)
        feedback_data["input_ids"].append(inputs["input_ids"][0])
        feedback_data["labels"].append(labels[0])
    if len(feedback_data["input_ids"]) >= 4:  # Update every 4 feedbacks
        online_update(model, tokenizer, feedback_data)
        feedback_data = {"input_ids": [], "labels": []}
