## Digital-Persona

## Part 1: Creating the dataset

### Financial Document QAR Generator (OCR → Question-Answer-Reasoning)

This part automates the extraction of financial insights from PDF documents by performing the following steps:

1. **OCR on PDFs** using EasyOCR (page-level)
2. **Contextual analysis** across all pages (up to 120K tokens)
3. **Question generation** per page based on financial reasoning
4. **Answer and reasoning generation** using large-context LLMs via NVIDIA's API
5. **Exports QAR triplets** (Question-Answer-Reasoning) to CSV per document

## Part 2: Finetuning

This part uses the generated dataset, and then finetunes the LLM
