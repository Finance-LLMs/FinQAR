# This code scans PDF using easyocr, stores them (per page format) in a dictionary
# Using the system prompts (for generating high quality responses)
# For each page, we will generate set of questions, the question will be answered based on the total pages visited till that point
# As the model used in this code supports 128k context, I have limited it to the last 120k tokens, discarding the oldest visited context
# A question is generated for every 100 tokens on that page using the SYSTEM_QUESTION prompt as system prompt
# That question is then passed along with the context for generating high quality response from the given context with low temperature to avoid hallunications
# The Question and Answer is then passed along with the context to generate the thought process OR reasoning for that answer based on that context

import os
import glob
import time
import numpy as np
import pandas as pd
import easyocr
from pdf2image import convert_from_bytes
import requests
import json
import gc
import re
import tiktoken

# === USER SETTINGS ===
PDF_DIR = os.path.expanduser("/folderContainingPDF")
PARA_TXT_ROOT = os.path.expanduser("/folderToDumpOCRpages")
OUTPUT_ROOT = "qar_dataset"
TOKENS_PER_QUESTION = 100  # 1 question per 100 tokens on the page, change is as per your need
NVIDIA_API_KEY = "nvapi-........" # put your NVIDIA API Key here
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Use a model that supports 128k context
MODEL_NAME = "meta/llama-3.1-70b-instruct"  # 128k context
# Alternative models with 128k+ context: nvidia/llama-3.1-nemotron-70b-instruct

MIN_WORDS = 50
MAX_CONTEXT_TOKENS = 120000  # Reserve some tokens for system prompts and output
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
You are a CFA charterholder. Provide step-by-step reasoning:
1. Start: "Based solely on the text:"
2. Analyze numbers/text
3. Connect to finance principles
4. End with "Takeaway: [one-sentence insight]"
Use only provided text/numbers. Output only the reasoning.
"""

# Initialize OCR and tokenizer
reader = easyocr.Reader(['en'], gpu=True)
tokenizer = tiktoken.encoding_for_model("gpt-4")  # Use for token counting

def count_tokens(text):
    """Count tokens in text using tiktoken"""
    return len(tokenizer.encode(text))

def truncate_to_token_limit(text, max_tokens):
    """Truncate text to stay within token limit"""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate from the beginning to keep the most recent context
    truncated_tokens = tokens[-max_tokens:]
    return tokenizer.decode(truncated_tokens)

def pdf_to_images(path: str):
    with open(path, "rb") as f:
        return convert_from_bytes(f.read())

# The below function is for doing OCR 'per paragraph' and then dump it.

# def run_ocr_and_dump(path: str) -> dict:
#     """
#     OCR each page, dump paragraphs, and return dict: {page_num: [paras...] }
#     """
#     os.makedirs(PARA_TXT_ROOT, exist_ok=True)
#     images = pdf_to_images(path)
#     pages = {}
#     base = os.path.splitext(os.path.basename(path))[0]
    
#     for i, img in enumerate(images, start=1):
#         print(f"Processing page {i}/{len(images)} with OCR...")
#         arr = np.array(img)
#         results = reader.readtext(arr, detail=0, paragraph=True, x_ths=0.5, y_ths=0.2)
#         valid = [p for p in results if MIN_WORDS <= len(p.split()) <= 500]
#         pages[i] = valid
        
#         # Persist each paragraph text
#         for j, p in enumerate(valid, start=1):
#             fn = os.path.join(PARA_TXT_ROOT, f"{base}_p{i}_{j}.txt")
#             with open(fn, "w", encoding="utf-8") as f:
#                 f.write(p)
    
#     return pages

def ocr_pages_to_dict(pdf_path: str) -> dict[int, str]:
    """
    OCR every page and return a dictionary: {page_number: full_page_text}.
    • No paragraph / word-count filtering
    • Optionally saves one .txt file per page (remove if not needed)
    """
    images = pdf_to_images(pdf_path)
    pages = {}
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    dump_dir = os.path.join(PARA_TXT_ROOT, base_name)          # keep existing root
    os.makedirs(dump_dir, exist_ok=True)

    for i, img in enumerate(images, start=1):
        print(f"Processing page {i}/{len(images)} …")
        page_lines = reader.readtext(np.array(img), detail=0)   # line-level OCR
        page_text = " ".join(page_lines).strip()                # whole page

        pages[i] = page_text                                    # store as ONE string

        # --- optional: persist full page text ---
        txt_name = os.path.join(dump_dir, f"{base_name}_page{i}.txt")
        with open(txt_name, "w", encoding="utf-8") as f:
            f.write(page_text)

    return pages


def call_nvidia_api(system_prompt: str, user_content: str, temp: float, max_tokens: int) -> str:
    """
    Call NVIDIA API with system prompt and user content
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Combine system prompt and user content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temp,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return f"[API Error: {response.status_code}]"
            
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return f"[API Error: {str(e)}]"

def generate_with_system(system_prompt: str, user_content: str, temp: float, max_tokens: int) -> str:
    """
    Generate text using NVIDIA API with system prompt and context length management
    """
    # Calculate total prompt tokens
    total_prompt = system_prompt + "\n\n" + user_content
    prompt_tokens = count_tokens(total_prompt)
    
    # Check if we need to truncate
    if prompt_tokens > MAX_CONTEXT_TOKENS:
        print(f"Context too long ({prompt_tokens} tokens), truncating to {MAX_CONTEXT_TOKENS} tokens")
        # Reserve tokens for system prompt
        system_tokens = count_tokens(system_prompt)
        available_tokens = MAX_CONTEXT_TOKENS - system_tokens - 100  # Buffer
        user_content = truncate_to_token_limit(user_content, available_tokens)
        print(f"Truncated context to {count_tokens(user_content)} tokens")
    
    return call_nvidia_api(system_prompt, user_content, temp, max_tokens)

def generate_qar_for_page(cumulative_context: str, page_text: str, page_num: int, total_pages: int) -> list:
    """
    Generate QAR triplets based on page tokens (1 question per 100 tokens on current page).
    Uses cumulative context for generating questions but calculates count from current page tokens.
    Handles question_count == 0 by skipping generation.
    """
    # Calculate tokens for current page only to determine number of questions
    page_tokens = count_tokens(page_text)
    if page_tokens > 50:
        questions_count = max(1, page_tokens // TOKENS_PER_QUESTION)  # Allow zero questions
    else:
        question_count = 0
    
    print(f"Page {page_num}/{total_pages}: Page has {page_tokens} tokens, generating {questions_count} questions")
    
    # If no questions to generate, return empty list immediately
    if questions_count == 0:
        return []
    
    # Prepare cumulative context (truncate if needed)
    context_tokens = count_tokens(cumulative_context)
    context_for_generation = cumulative_context
    if context_tokens > MAX_CONTEXT_TOKENS:
        context_for_generation = truncate_to_token_limit(cumulative_context, MAX_CONTEXT_TOKENS)
    
    triplets = []
    for i in range(questions_count):
        # Generate question
        q = generate_with_system(SYSTEM_QUESTION, page_text, temp=0.9, max_tokens=64)
        
        # Generate answer
        answer_ctx = f"Text: {context_for_generation}\nQuestion: {q}"
        a = generate_with_system(SYSTEM_ANSWER, answer_ctx, temp=0.1, max_tokens=256)
        if a == '[Not Directly Stated]':
            a_retry = generate_with_system(SYSTEM_ANSWER, answer_ctx, temp=0.2, max_tokens=256)
            if a_retry != '[Not Directly Stated]':
                a = a_retry
        
        # Generate reasoning
        reasoning_ctx = f"Text: {context_for_generation}\nQuestion: {q}\nAnswer: {a}"
        r = generate_with_system(SYSTEM_REASONING, reasoning_ctx, temp=0.1, max_tokens=512)
        
        triplets.append({
            "page_number": page_num,
            "page_tokens": page_tokens,
            "questions_generated": questions_count,
            "cumulative_context_tokens": context_tokens,
            #"context": context_for_generation, # not recommended to save as it is a growing context
            "question": q,
            "answer": a,
            "reasoning": r
        })
        
        time.sleep(1.0)
    
    return triplets


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print(f"Using NVIDIA API with model: {MODEL_NAME}")
    print(f"Maximum context tokens: {MAX_CONTEXT_TOKENS}")
    print(f"Questions per {TOKENS_PER_QUESTION} tokens on each page")
    print(f"Processing PDFs from: {PDF_DIR}")
    
    for pdf_path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        
        pdf_name = slugify(os.path.splitext(os.path.basename(pdf_path))[0])
        out_dir = os.path.join(OUTPUT_ROOT, pdf_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "qar_pages.csv")

        # OCR and extract text
        print("Running OCR...")
        pages = ocr_pages_to_dict(pdf_path)
        all_triples = []
        cumulative_text = []

        for page_num in sorted(pages):          # dictionary keys unchanged
            print(f"\nProcessing page {page_num}/{len(pages)}")
        
            current_page_text = pages[page_num]             # already a single string
            cumulative_text.append(current_page_text)       # list of page strings
            full_cumulative_context = "\n\n".join(cumulative_text)
            
            # Show context growth
            context_tokens = count_tokens(full_cumulative_context)
            page_tokens = count_tokens(current_page_text)
            print(f"Current page has {page_tokens} tokens")
            print(f"Cumulative context now has {context_tokens} tokens")
            
            try:
                # Pass both cumulative context and current page text
                page_triples = generate_qar_for_page(
                    cumulative_context=full_cumulative_context,
                    page_text=current_page_text,
                    page_num=page_num,
                    total_pages=len(pages)
                )
                
                # Save to CSV
                df = pd.DataFrame(page_triples)
                df.to_csv(
                    csv_path, 
                    mode="a",
                    header=(page_num == 1),
                    index=False, 
                    encoding="utf-8"
                )
                
                all_triples.extend(page_triples)
                print(f"  Generated {len(page_triples)} questions based on {page_tokens} page tokens")
                
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue
                
        print(f"\nCompleted {pdf_name}: {len(all_triples)} total questions")
        
        # Generate summary report
        summary_path = os.path.join(out_dir, "context_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"PDF: {pdf_name}\n")
            f.write(f"Total Pages: {len(pages)}\n")
            f.write(f"Total Questions: {len(all_triples)}\n")
            f.write(f"Model Used: {MODEL_NAME}\n")
            f.write(f"Max Context Tokens: {MAX_CONTEXT_TOKENS}\n")
            f.write(f"Tokens per Question: {TOKENS_PER_QUESTION}\n")
            f.write(f"Final Cumulative Context Size: {count_tokens(full_cumulative_context)} tokens\n")
            
            # Add page-wise breakdown
            f.write(f"\nPage-wise Breakdown:\n")
            for page_num in sorted(pages.keys()):
                page_text = "\n\n".join(pages[page_num])
                page_tokens = count_tokens(page_text)
                questions_count = max(1, page_tokens // TOKENS_PER_QUESTION)
                f.write(f"Page {page_num}: {page_tokens} tokens -> {questions_count} questions\n")

if __name__ == "__main__":
    main()
