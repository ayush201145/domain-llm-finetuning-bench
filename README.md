# Domain-Specific LLM Fine-Tuning & Evaluation Benchmark

Parameter-efficient fine-tuning (PEFT/QLoRA) pipeline, synthetic dataset curation, and automated vLLM evaluation harness scoring domain accuracy (SQL generation / specialized vertical) using DeepEval / Ragas metrics.

## Key Features
- **QLoRA 4-bit Quantized Fine-Tuning**: Target adapter configuration (`q_proj`, `v_proj`) with low rank ($r=16$) matrix decomposition.
- **Dataset Curation Tooling**: Automated generation of task-specific instruction tuning pairs.
- **vLLM Inference Harness**: High-throughput token inference benchmark measuring tokens/sec and latency percentiles.
- **DeepEval Automated Benchmark**: Evaluation suite scoring Exact Match Accuracy, Execution Accuracy, and Faithfulness metrics.

## Quick Start

### 1. Prepare Dataset
```bash
python dataset_prep.py
```

### 2. Execute QLoRA Fine-Tuning
```bash
python finetune_qlora.py
```

### 3. Run Benchmark Suite
```bash
python eval_harness.py
```

## Tech Stack
- **Deep Learning**: PyTorch, Hugging Face Transformers, PEFT, bitsandbytes
- **Inference Engine**: vLLM High-Throughput Engine
- **Evaluation**: DeepEval / Ragas Metric Suite
