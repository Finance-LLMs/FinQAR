import os
import torch
from datasets import load_dataset

from huggingface_hub import login

# Provide your huggingface token here (replace with your actual token string)
HF_TOKEN = "hf_TOKEN" 

login(token=HF_TOKEN)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# === USER SETTINGS ===
CSV_PATH       = "qar_pages_BOOK.csv"              # your CSV file
MODEL_NAME     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR     = "qar_finetuned"
BATCH_SIZE     = 2
GRAD_ACCUM     = 8
MAX_SEQ_LEN    = 512                             # max tokens per example
LEARNING_RATE  = 2e-4
EPOCHS         = 3
DEVICE         = "cuda"
# =====================

# 1. Load and preprocess dataset
raw_ds = load_dataset("csv", data_files=CSV_PATH)["train"]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    padding_side="right",
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess(example):
    # Combine question, answer, reasoning into one sequence
    prompt = (
        "<|QUESTION|>\n"   + example["question"].strip()   +
        "\n<|ANSWER|>\n"    + example["answer"].strip()     +
        "\n<|REASONING|>\n" + example["reasoning"].strip()  +
        "\n<|END|>"
    )
    toks = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks

train_ds = raw_ds.map(
    preprocess,
    remove_columns=raw_ds.column_names,
    batched=False,
)

# 2. Load base model with 4-bit quantization
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)

# 3. Prepare model for k-bit training and attach LoRA
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# 4. Training arguments
steps_per_epoch = len(train_ds) // BATCH_SIZE // GRAD_ACCUM
total_steps = steps_per_epoch * EPOCHS

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=total_steps,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=20,
    save_steps=200,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
)

# 5. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 6. Launch training
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuned model saved to {OUTPUT_DIR}")
