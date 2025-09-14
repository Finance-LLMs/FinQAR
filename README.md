# FinQAR

A dataset for this project is available at the following Google Drive link:
[Dataset Download](https://drive.google.com/drive/folders/1KjHJuadd6VclbZd-yAIUBjlw0VWJi96S?usp=sharing)

A sophisticated financial document analysis and LLM fine-tuning system that automatically extracts financial insights from PDF documents and creates Question-Answer-Reasoning (QAR) datasets for training specialized financial AI models.

## Overview

FinQAR - Finance Question Answer Reasoning Triplets is designed to bridge the gap between unstructured financial documents and structured AI training data. The system performs OCR on financial PDFs, generates contextual questions based on financial reasoning, provides comprehensive answers, and creates detailed reasoning chains - all optimized for fine-tuning language models to become domain-specific financial experts.

## Features

### Part 1: Dataset Generation

- **Advanced OCR Processing**: Uses EasyOCR for page-level text extraction from financial PDFs 
- **Large Context Analysis**: Handles up to 120K tokens for comprehensive document understanding 
- **Smart Question Generation**: Creates financial reasoning-based questions per page using configurable token thresholds 
- **Professional Answer Generation**: Leverages large-context LLMs via NVIDIA API or local GPU inference 
- **Detailed Reasoning Chains**: Generates CFA-level analytical reasoning for each question-answer pair 
- **Multiple Processing Options**: Support for both NVIDIA API and local GPU processing 


### Part 2: Model Fine-tuning

- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) for memory-efficient training 
- **4-bit Quantization**: Optimized for resource-constrained environments 
- **Flexible Model Support**: Compatible with various instruction-tuned models 
- **Performance Comparison**: Built-in inference comparison between original and fine-tuned models 



## Project Structure

```
FinQAR/
├── data/
│   └── README.md                # Dataset documentation and download link
├── QAR-Generation/
│   ├── generate_QAR_local_GPU.py
│   ├── generate_QAR_triplets_from_pdf.py
│   └── qarNvidiaAPI.py
├── scripts/
│   ├── finetune.py
│   └── inference.py
└── README.md                    # Project documentation
```

### Folder Overview

- **data/**: Contains dataset documentation and download instructions.
- **QAR-Generation/**: Scripts for generating Question-Answer-Reasoning (QAR) triplets from financial PDFs using local GPU or NVIDIA API.
- **scripts/**: Scripts for model fine-tuning and inference/comparison.
- **README.md**: Main project documentation and instructions.


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

- Uses advanced models like Llama-3.1-70B for superior reasoning quality 
- Handles large context windows (up to 128K tokens) 
- Automatic context truncation and management 


#### Using Local GPU

```bash
python generate_QAR_local_GPU.py
```

- Processes documents entirely on local hardware 
- Uses quantized models for memory efficiency 
- Suitable for sensitive documents requiring on-premise processing 


#### Using Enhanced NVIDIA API

```bash
python qarNvidiaAPI.py
```

- Professional-grade system prompts for institutional analysis 
- Enhanced reasoning with CFA-level financial expertise 
- Batch question generation for improved efficiency 


### Step 2: Fine-tune the Model

```bash
python finetune.py
```

Key features:

- **LoRA Configuration**: Efficient fine-tuning with minimal resource requirements 
- **4-bit Quantization**: Reduces memory usage while maintaining performance 
- **Flexible Training**: Configurable batch sizes and learning rates 


### Step 3: Compare Performance

```bash
python inference.py
```

<img width="1600" height="780" alt="image" src="https://github.com/user-attachments/assets/11df9049-486a-4973-b520-d3c811718bb1" />


- Side-by-side comparison of original vs fine-tuned models 
- Interactive questioning interface 
- Performance evaluation for financial domain expertise 


## System Prompts and Quality

The system employs professionally crafted prompts designed for institutional-grade financial analysis:

### Question Generation

- Focuses on valuation, risk analysis, market behavior, and business strategy 
- Limited to 50 words for clarity and precision 
- Ensures financial domain relevance 


### Answer Generation

- Senior financial analyst perspective with 200-word responses 
- Explicit information extraction with clear limitation statements 
- Professional objectivity with insightful interpretation 


### Reasoning Generation

- CFA charterholder-level analytical rigor 
- Step-by-step reasoning connecting numbers to finance principles 
- Comprehensive 400-word analytical reasoning chains 


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

- **Maximum Context**: 120K tokens with intelligent truncation 
- **Token Counting**: Uses tiktoken for accurate token management 
- **Memory Optimization**: Automatic garbage collection and CUDA cache clearing 


### Model Support

- **NVIDIA API Models**: Llama-3.1-70B, Nemotron-70B, Llama-4-scout, Llama-4-Maverick 
- **Local Models**: Qwen2.5-32B-Instruct-GPTQ, Llama-3.1-8B 
- **Quantization**: 4-bit and 8-bit support for resource efficiency 


### Quality Controls

- **Temperature Settings**: Low temperature (0.1) for consistent answers 
- **Retry Logic**: Automatic retry for "[Not Directly Stated]" responses 
- **Professional Standards**: Institutional-grade analytical rigor 


## Performance Considerations

- **GPU Memory**: Requires 24GB+ VRAM for optimal local processing 
- **Processing Speed**: 1 question per 100 tokens (configurable) 
- **Quality vs Speed**: API processing offers higher quality, local processing offers privacy 


## Applications

- **Investment Research**: Automated analysis of financial reports and documents
- **Risk Assessment**: Systematic extraction of risk factors and mitigation strategies
- **Financial Education**: Training datasets for financial AI applications
- **Document Digitization**: Converting legacy financial documents into structured data
- **Compliance Analysis**: Automated review of regulatory and compliance documents


## Contributing

This project is designed for financial professionals, AI researchers, and developers working on domain-specific language models. Contributions focusing on enhanced financial reasoning, improved OCR accuracy, or model optimization are welcome.

## License

This project is licensed under [MIT License](LICENSE).
