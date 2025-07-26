# Digital-Persona

A sophisticated financial document analysis and LLM fine-tuning system that automatically extracts financial insights from PDF documents and creates Question-Answer-Reasoning (QAR) datasets for training specialized financial AI models.

## Overview

Digital-Persona is designed to bridge the gap between unstructured financial documents and structured AI training data. The system performs OCR on financial PDFs, generates contextual questions based on financial reasoning, provides comprehensive answers, and creates detailed reasoning chains - all optimized for fine-tuning language models to become domain-specific financial experts.

## Features

### Part 1: Dataset Generation

- **Advanced OCR Processing**: Uses EasyOCR for page-level text extraction from financial PDFs[^1]
- **Large Context Analysis**: Handles up to 120K tokens for comprehensive document understanding[^1]
- **Smart Question Generation**: Creates financial reasoning-based questions per page using configurable token thresholds[^1]
- **Professional Answer Generation**: Leverages large-context LLMs via NVIDIA API or local GPU inference[^1]
- **Detailed Reasoning Chains**: Generates CFA-level analytical reasoning for each question-answer pair[^1]
- **Multiple Processing Options**: Support for both NVIDIA API and local GPU processing[^1]


### Part 2: Model Fine-tuning

- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) for memory-efficient training[^1]
- **4-bit Quantization**: Optimized for resource-constrained environments[^1]
- **Flexible Model Support**: Compatible with various instruction-tuned models[^1]
- **Performance Comparison**: Built-in inference comparison between original and fine-tuned models[^1]


## Project Structure

```
Digital-Persona/
├── generate_QAR_triplets_from_pdf.py    # NVIDIA API-based QAR generation
├── generate_QAR_local_GPU.py            # Local GPU-based QAR generation  
├── qarNvidiaAPI.py                      # Enhanced NVIDIA API implementation
├── finetune.py                          # Finetune a model with the help of generated QAR Triplets
├── inference.py                         # Model comparison and inference
└── README.md                            # Project documentation
```


## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- NVIDIA API key (for API-based processing)
- Hugging Face account and token


### Dependencies

```bash
pip install torch transformers
pip install easyocr pdf2image
pip install pandas numpy
pip install peft bitsandbytes
pip install tiktoken requests
pip install huggingface_hub datasets
```


## Configuration

### API-Based Processing

1. Set your NVIDIA API key in the configuration:
```python
NVIDIA_API_KEY = "nvapi-your-key-here"
```

2. Configure paths and parameters:
```python
PDF_DIR = "/path/to/your/pdfs"
OUTPUT_ROOT = "qar_dataset"
TOKENS_PER_QUESTION = 100  # Adjust question density
```


### Local GPU Processing

1. Set your Hugging Face token:
```python
HF_TOKEN = "hf_your-token-here"
```

2. Configure GPU memory allocation:
```python
max_mem = {
    0: "48GiB",  # Adjust based on your GPU
}
```


## Usage

### Step 1: Generate QAR Dataset

#### Using NVIDIA API (Recommended for High Quality)

```bash
python generate_QAR_triplets_from_pdf.py
```

- Uses advanced models like Llama-3.1-70B for superior reasoning quality[^1]
- Handles large context windows (up to 128K tokens)[^1]
- Automatic context truncation and management[^1]


#### Using Local GPU

```bash
python generate_QAR_local_GPU.py
```

- Processes documents entirely on local hardware[^1]
- Uses quantized models for memory efficiency[^1]
- Suitable for sensitive documents requiring on-premise processing[^1]


#### Using Enhanced NVIDIA API

```bash
python qarNvidiaAPI.py
```

- Professional-grade system prompts for institutional analysis[^1]
- Enhanced reasoning with CFA-level financial expertise[^1]
- Batch question generation for improved efficiency[^1]


### Step 2: Fine-tune the Model

```bash
python finetune.py
```

Key features:

- **LoRA Configuration**: Efficient fine-tuning with minimal resource requirements[^1]
- **4-bit Quantization**: Reduces memory usage while maintaining performance[^1]
- **Flexible Training**: Configurable batch sizes and learning rates[^1]


### Step 3: Compare Performance

```bash
python inference.py
```

- Side-by-side comparison of original vs fine-tuned models[^1]
- Interactive questioning interface[^1]
- Performance evaluation for financial domain expertise[^1]


## System Prompts and Quality

The system employs professionally crafted prompts designed for institutional-grade financial analysis:

### Question Generation

- Focuses on valuation, risk analysis, market behavior, and business strategy[^1]
- Limited to 50 words for clarity and precision[^1]
- Ensures financial domain relevance[^1]


### Answer Generation

- Senior financial analyst perspective with 200-word responses[^1]
- Explicit information extraction with clear limitation statements[^1]
- Professional objectivity with insightful interpretation[^1]


### Reasoning Generation

- CFA charterholder-level analytical rigor[^1]
- Step-by-step reasoning connecting numbers to finance principles[^1]
- Comprehensive 400-word analytical reasoning chains[^1]


## Output Format

The system generates structured CSV files containing:

- **page_number**: Source page reference
- **page_tokens**: Token count for context sizing
- **questions_generated**: Number of questions per page
- **cumulative_context_tokens**: Growing context size
- **question**: Generated financial question
- **answer**: Comprehensive analytical response
- **reasoning**: Detailed step-by-step analysis


## Technical Specifications

### Context Management

- **Maximum Context**: 120K tokens with intelligent truncation[^1]
- **Token Counting**: Uses tiktoken for accurate token management[^1]
- **Memory Optimization**: Automatic garbage collection and CUDA cache clearing[^1]


### Model Support

- **NVIDIA API Models**: Llama-3.1-70B, Nemotron-70B[^1]
- **Local Models**: Qwen2.5-32B-Instruct-GPTQ, Llama-3.1-8B[^1]
- **Quantization**: 4-bit and 8-bit support for resource efficiency[^1]


### Quality Controls

- **Temperature Settings**: Low temperature (0.1) for consistent answers[^1]
- **Retry Logic**: Automatic retry for "[Not Directly Stated]" responses[^1]
- **Professional Standards**: Institutional-grade analytical rigor[^1]


## Performance Considerations

- **GPU Memory**: Requires 24GB+ VRAM for optimal local processing[^1]
- **Processing Speed**: 1 question per 100 tokens (configurable)[^1]
- **Quality vs Speed**: API processing offers higher quality, local processing offers privacy[^1]


## Applications

- **Investment Research**: Automated analysis of financial reports and documents
- **Risk Assessment**: Systematic extraction of risk factors and mitigation strategies
- **Financial Education**: Training datasets for financial AI applications
- **Document Digitization**: Converting legacy financial documents into structured data
- **Compliance Analysis**: Automated review of regulatory and compliance documents


## Contributing

This project is designed for financial professionals, AI researchers, and developers working on domain-specific language models. Contributions focusing on enhanced financial reasoning, improved OCR accuracy, or model optimization are welcome.
