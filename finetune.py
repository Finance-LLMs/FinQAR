import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set preferred device BEFORE imports

# from datasets import load_dataset //if you want to use online CSV data
from unsloth import FastLanguageModel
import torch
import pandas as pd

# Step 1: Load dataset
dataset = pd.read_csv(csv_path)
print("Dataset loaded successfully!")

# Step 2: Model configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Step 3: Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Step 4: Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=True,
)

# Step 5: Format dataset
def formatting_prompts_func(examples):
    texts = [f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
             for question, answer in zip(examples['question'], examples['answer'])]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Step 6: Training arguments
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="./llama3-2-7b-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=50,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
)

# Step 7: Create trainer (automatically uses specified GPU)
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,
    train_dataset=dataset['train'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)

# Step 8: Start training
trainer.train()

# Step 9: Save
model_path = os.path.join(os.getcwd(),"data","models","lamma_3_2")
tokenizer_path = os.path.join(os.getcwd(),"data","tokenizer","lamma_3_2")
model.save_pretrained(os.path.join(model_path,"llama3-2-1b-finetuned"))
tokenizer.save_pretrained(tokenizer_path,"llama3-2-1b-finetuned")

# Define paths
model_path = os.path.join(os.getcwd(), "data", "models", "lamma_3_2", "llama-2-7b")
tokenizer_path = os.path.join(os.getcwd(), "data", "tokenizer", "lamma_3_2", "llama-2-7b")

# Create directories if they don't exist
os.makedirs(model_path, exist_ok=True)
os.makedirs(tokenizer_path, exist_ok=True)

model = model.merge_and_unload()

# Save model and tokenizer
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

print(f"Model saved to: {model_path}")
print(f"Tokenizer saved to: {tokenizer_path}")
