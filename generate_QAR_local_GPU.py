import os, glob, time, re, gc
import numpy as np, pandas as pd
import easyocr
from pdf2image import convert_from_bytes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# === USER SETTINGS ===
PDF_DIR      = os.path.expanduser("~/DigitalTwin/dataset_PDFs/PDFs_3090_2")
PARA_TXT_ROOT= os.path.expanduser("~/DigitalTwin/paragraph_texts")
OUTPUT_ROOT  = "qar_dataset"
QUESTIONS_PER_PAGE = 10
HF_TOKEN     = "YOUR_HUGGINGFACE_TOKEN"
DEVICE       = torch.device("cuda")
MODEL_ID     = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" # OR Any instruct model
MIN_WORDS    = 50
# =====================

# System Prompts
SYSTEM_QUESTION = """
You are a senior financial analyst. Generate exactly one focused question (max 50 words) 
about valuation, risk analysis, market behavior, or business strategy from the user's text excerpt.
Only output the question itself.
"""
SYSTEM_ANSWER = """
You are a Senior financial analyst. Answer the user's question clearly in less than 200 words using ONLY explicit information from the text.
If not stated, reply '[Not Directly Stated]'. Output only the answer.
"""
SYSTEM_REASONING = """
You are a CFA charterholder. Provide step-by-step reasoning of how you reached at this answer:
1. Start: "Based solely on the text:"
2. Analyze numbers/text
3. Connect to finance principles
4. End with "Takeaway: [one-sentence insight]"
Use only provided text/numbers. Output only the reasoning in 230 words.
"""

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=True)

from transformers.utils.quantization_config import BitsAndBytesConfig

# Monkey-patch BitsAndBytesConfig to add get_loading_attributes
def _get_loading_attributes(self):
    # Return only the relevant flag attributes expected by merge_quantization_configs
    attrs = {}
    for attr in [
        "load_in_4bit", "load_in_8bit",
        "llm_int8_enable_fp32_cpu_offload",
        "llm_int8_has_fp16_weight", "llm_int8_skip_modules", "llm_int8_threshold",
        "bnb_4bit_quant_type","bnb_4bit_use_double_quant","bnb_4bit_compute_dtype"
    ]:
        if hasattr(self, attr):
            attrs[attr] = getattr(self, attr)
    return attrs

BitsAndBytesConfig.get_loading_attributes = _get_loading_attributes


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, token=HF_TOKEN, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from transformers import AutoModelForCausalLM

# GPUs visible as cuda:0, cuda:1, cuda:2 Order according to PCIe slot numbers
max_mem = {
    0: "48GiB",   # physical PCIe id 1 (A6000)
    #1: "23GiB"  # physical PCIe id 2 (3090 #1)
    #2: "23GiB",   # physical PCIe id 3 (3090 #2)
    #"cpu": "5GiB"
}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    load_in_8bit=True,         
    device_map="auto",
    max_memory=max_mem,
    torch_dtype=torch.float16
)
model.eval()




def pdf_to_images(path): 
    with open(path,"rb") as f: return convert_from_bytes(f.read())

def run_ocr_and_dump(path):
    os.makedirs(PARA_TXT_ROOT, exist_ok=True)
    pages, base = {}, os.path.splitext(os.path.basename(path))[0]
    for i, img in enumerate(pdf_to_images(path), start=1):
        arr = np.array(img)
        paras = reader.readtext(arr, detail=0, paragraph=True, x_ths=0.5, y_ths=0.2)
        valid = [p for p in paras if MIN_WORDS <= len(p.split()) <= 500]
        pages[i] = valid
        for j, p in enumerate(valid, start=1):
            fn = os.path.join(PARA_TXT_ROOT, f"{base}_p{i}_{j}.txt")
            with open(fn, "w", encoding="utf-8") as f: f.write(p)
    return pages

def generate_with_system(sys_prompt, user_content, temp, max_tokens):
    msgs = [{"role":"system","content":sys_prompt.strip()},
            {"role":"user",  "content":user_content.strip()}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=131072).to(model.device)
    with torch.inference_mode():
        outs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temp>0),
            temperature=temp or None,
            top_p=0.9 if temp>0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    gen = outs[0, inputs.input_ids.shape[1]:]
    result = tokenizer.decode(gen, skip_special_tokens=True).strip()
    del inputs, outs, gen
    torch.cuda.empty_cache(); gc.collect()
    return result

def generate_qar_for_page(context, page_context):
    num_tokens = len(tokenizer(page_context, return_tensors="pt").input_ids[0])
    num_questions = max(1, num_tokens // 100)
    triplets = []
    for i in range(num_questions):
        print(f"Question {i} of {num_questions}")
        q = generate_with_system(SYSTEM_QUESTION, page_context, 0.9, 64)
        a = generate_with_system(SYSTEM_ANSWER, f"Text: {context}\nQuestion: {q}", 0.1, 256)
        answerable = True
        if a == '[Not Directly Stated]':
            a2 = generate_with_system(SYSTEM_ANSWER, f"Text: {context}\nQuestion: {q}", 0.2, 256)
            if a2 != '[Not Directly Stated]':
                a = a2
            else:
                answerable = False
        r = generate_with_system(SYSTEM_REASONING, f"Text: {context}\nQuestion: {q}\nAnswer: {a}", 0.1, 256)
        triplets.append({"question": q, "answer": a, "reasoning": r, "answerable": answerable})
        time.sleep(0.1)
    return triplets

def slugify(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+','_', name)
    return re.sub(r'_+','_',name).strip('_')

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for pdf_path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        pdf_name = slugify(os.path.basename(pdf_path).rsplit(".",1)[0])
        out_dir = os.path.join(OUTPUT_ROOT, pdf_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "qar_pages.csv")
        pages = run_ocr_and_dump(pdf_path)
        all_triples = []; cumulative_text = []
        for pnum in sorted(pages):
            page_text = "\n\n".join(pages[pnum])
            cumulative_text.extend(pages[pnum])
            context = "\n\n".join(cumulative_text)
            print(f"Processing page {pnum}/{len(pages)} â€¦")
            try:
                page_triples = generate_qar_for_page(context, page_text)
                if page_triples:
                    pd.DataFrame(page_triples).to_csv(
                        csv_path, mode="a",
                        header=(pnum==1), index=False
                    )
                    all_triples.extend(page_triples)
                    print(f" â†’ {len(page_triples)} QARs generated")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); gc.collect()
                    print("OOM: retrying with same full contextâ€¦")
                    page_triples = generate_qar_for_page(context)
                    pd.DataFrame(page_triples).to_csv(csv_path, mode="a", header=False,index=False)
                    all_triples.extend(page_triples)
        print(f"Completed {pdf_name}: {len(all_triples)} total QARs")

if __name__=="__main__":
    main()
