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
PDF_DIR = os.path.expanduser("/home/research2/DigitalTwin/dataset_PDFs/PDFs_A6000")
PARA_TXT_ROOT = os.path.expanduser("/home/research2/DigitalTwin/page_tests")
OUTPUT_ROOT = "qar_dataset"
TOKENS_PER_QUESTION = 100  # 1 question per 100 tokens on the page
NVIDIA_API_KEY = "nvapi-KEY"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"

MIN_WORDS = 50
MAX_CONTEXT_TOKENS = 990000  # Reserve some tokens for system prompts and output
# =====================

# System Prompts

SYSTEM_ANSWER = """
You are a Senior Financial Analyst conducting institutional-grade analysis. Provide a comprehensive response that demonstrates the analytical rigor expected in professional financial research.

ANALYTICAL APPROACH:
1. Extract and synthesize relevant financial data from the text
2. Apply appropriate financial analysis frameworks where evident
3. Connect individual data points to broader financial themes
4.Identify potential risks or opportunities suggested by the information

RESPONSE STRUCTURE (180-250 words):
1. Executive Summary: Lead with your primary finding or conclusion
2. Supporting Analysis: Present evidence from the text with proper financial context
3. Quantitative Details: Include relevant numbers, ratios, or percentages with clear interpretation
4. Professional Assessment: Conclude with implications for stakeholders or decision-makers

QUALITY STANDARDS:
1. Maintain objectivity while providing insightful interpretation
2. Use precise financial language appropriate for senior management audiences
4. Clearly distinguish between explicit text information and reasonable inferences
5. When data is incomplete, explicitly state limitations: "[Analysis limited by available data: text shows X but does not specify Y]"

For completely unstated information, respond: "[Not Directly Stated]"

Deliver your analysis with the confidence and precision expected from a senior financial professional.
"""


SYSTEM_REASONING = """
You are a CFA charterholder providing comprehensive financial analysis. Write your reasoning as cohesive, flowing paragraphs that seamlessly integrate quantitative analysis with finance theory.

Begin with "Based solely on the text:" then develop your analysis in 2-4 well-structured paragraphs:

Paragraph 1: Present primary numerical findings with immediate contextual interpretation
Paragraph 2: Connect these findings to relevant finance principles and industry benchmarks  
Paragraph 3 (if needed): Discuss implications for stakeholders or strategic decision-making
Final sentence: "Takeaway:" followed by a synthesized insight that captures the essence of your analysis

Embed all calculations naturally within sentences, showing both the computation and its significance. Demonstrate how multiple financial metrics interact to create a comprehensive picture of performance or risk. Your analysis should reflect the sophisticated reasoning expected in professional investment research.

Use only the provided text and numbers. Output only the reasoning paragraphs in approximately 400 words.
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
    1. No paragraph / word-count filtering
    2. Optionally saves one .txt file per page (remove if not needed)
    """
    images = pdf_to_images(pdf_path)
    pages = {}
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    dump_dir = os.path.join(PARA_TXT_ROOT, base_name)          # keep existing root
    os.makedirs(dump_dir, exist_ok=True)

    for i, img in enumerate(images, start=1):
        print(f"Processing page {i} / { len(images) } ...")
        page_lines = reader.readtext(np.array(img), detail=0)
        page_text = " ".join(page_lines).strip()

        pages[i] = page_text                           

        # --- optional: persist full page text ---
        #txt_name = os.path.join(dump_dir, f"{base_name}_page{i}.txt")
        #with open(txt_name, "w", encoding="utf-8") as f:
        #    f.write(page_text)

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

def generate_qar_for_page(
        cumulative_context: str,
        page_text: str,
        page_num: int,
        total_pages: int
) -> list[dict]:
    """
    Generate QAR triplets.
    1 question per 100 tokens on CURRENT page (minimum 1 if page > 50 tokens).
    """
    # --- determine how many questions to ask ------------------------------
    page_tokens = count_tokens(page_text)
    if page_tokens <= 50:
        return []                 # skip short pages

    questions_count = max(1, page_tokens // TOKENS_PER_QUESTION)
    print(f"Page {page_num}/{total_pages}: {page_tokens} tokens -> "
          f"{questions_count} questions")

    # --- trim cumulative context if needed --------------------------------
    context_tokens = count_tokens(cumulative_context)
    context_for_generation = (
        truncate_to_limit(cumulative_context, MAX_CONTEXT_TOKENS)
        if context_tokens > MAX_CONTEXT_TOKENS else cumulative_context
    )

    # --- build the multi-question prompt ----------------------------------
    SYSTEM_QUESTION = f"""
    You are a senior financial analyst.
    From the text below, generate EXACTLY {questions_count} distinct questions
    (max 50 words each) covering valuation, risk analysis, market behavior, or
    business strategy.
    DO NOT START WITH TEXTS LIKE: "Here is a list of questions:"
    Each question should start with "<" and end with ">"
    Return ONLY that block in the exact format given below:
    <1. Question one >
    <2. Question two >
    ...
    <{questions_count}. Question {questions_count}>
    DO NOT include any other text before, between, or after these lines.
    """.strip()


    # --- ask for all questions in one call --------------------------------
    raw_q_block = generate_with_system(
        system_prompt=SYSTEM_QUESTION,
        user_content=page_text,
        temp=0.9,
        max_tokens=512
    )

    # --- parse the block into a clean list --------------------------------
    # remove any whitespace/newlines at ends
    raw_q_block = raw_q_block.strip()

    # ensure every question starts with "<" and ends with ">"
    q_items = []
    for chunk in raw_q_block.split("<"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # chunk now looks like "1. This is the question>"
        # drop the final ">" and any trailing spaces / newlines
        chunk = chunk.rstrip(">").strip()
        # remove the leading enumeration ("1." etc.)
        parts = chunk.split(".", 1)
        q_items.append(parts[1].strip() if len(parts) == 2 else chunk)

    # sanity-check length
    if len(q_items) != questions_count:
        print(f"[WARN] Model returned {len(q_items)} questions, expected "
              f"{questions_count}. Proceeding with the shorter list.")
        questions_count = len(q_items)

    # --- generate A & R per question --------------------------------------
    triplets = []
    for i, q in enumerate(q_items, start=1):
        # Answer
        answer_ctx = f"Text: {context_for_generation}\nQuestion: {q}"
        a = generate_with_system(SYSTEM_ANSWER, answer_ctx, temp=0.1,
                                 max_tokens=256)

        if a == "[Not Directly Stated]":
            a_retry = generate_with_system(SYSTEM_ANSWER, answer_ctx,
                                           temp=0.2, max_tokens=256)
            if a_retry != "[Not Directly Stated]":
                a = a_retry

        # Reasoning
        reasoning_ctx = f"Text: {context_for_generation}\nQuestion: {q}\nAnswer: {a}"
        r = generate_with_system(SYSTEM_REASONING, reasoning_ctx,
                                 temp=0.1, max_tokens=512)

        triplets.append({
            "page_number": page_num,
            "page_tokens": page_tokens,
            "questions_generated": questions_count,
            "cumulative_context_tokens": context_tokens,
            "question": q,
            "answer": a,
            "reasoning": r
        })

        time.sleep(1.0)   # courtesy pause

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
        fName = f"qar_pages_{slugify(os.path.basename(pdf_path))}.csv"
        csv_path = os.path.join(out_dir, fName)

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
