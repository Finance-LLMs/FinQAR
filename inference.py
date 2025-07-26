# Compare model's performance on Original vs Finetuned model

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# === USER SETTINGS ===
HF_TOKEN = "hf_KEY"
FINETUNED_DIR = "qar_finetuned"                # Directory of your fine-tuned model
ORIGINAL_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.25

# === Authenticate with Hugging Face ===
login(token=HF_TOKEN)

# === 4-bit quantization + CPU offload configuration ===
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

def load_model(path: str):
    """Load tokenizer and quantized model with offloading."""
    tokenizer = AutoTokenizer.from_pretrained(path, use_auth_token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_cfg,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        use_auth_token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, question: str) -> str:
    """Generate a response for a given question."""
    prompt = f"<|QUESTION|>\n You are Graham a Finance Expert, and the Author of the book Intelligent Investor.Answer the following question within 250 words. \n{question.strip()}\n<|ANSWER|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    # Load both models once
    print("Loading fine-tuned model")
    ft_tokenizer, ft_model = load_model(FINETUNED_DIR)
    print("Loading original model")
    orig_tokenizer, orig_model = load_model(ORIGINAL_MODEL)

    while True:
        question = input("\nEnter your question (or type NO to exit): ").strip()
        if question.upper() == "NO":
            print("Goodbye!")
            break

        print("\n--- Fine-Tuned Model Response ---")
        print(generate(ft_tokenizer, ft_model, question))

        print("\n--- Original Model Response ---")
        print(generate(orig_tokenizer, orig_model, question))

if __name__ == "__main__":
    main()
