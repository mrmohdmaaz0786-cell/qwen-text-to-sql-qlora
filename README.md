# Qwen2.5-Coder-7B Text-to-SQL QLoRA

Fine-tuned `Qwen/Qwen2.5-Coder-7B-Instruct` for Text-to-SQL generation using QLoRA on the Spider dataset.

The model takes a database schema and a natural-language question and generates the corresponding SQL query.

## Model

- Base Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Fine-tuning: QLoRA
- Quantization: 4-bit NF4
- LoRA Rank: 16
- LoRA Alpha: 32
- LoRA Dropout: 0.05
- Trainable Parameters: ~40.4M (~0.53%)

## Dataset

Spider Text-to-SQL dataset.

- Training examples after preprocessing: 6,918
- Validation examples: 1,034

The input contains:

- Database schema
- Table and column names
- Foreign-key relationships
- Natural-language question

The target is the SQL query.

## Results

| Metric | Score |
|---|---:|
| Normalized Exact Match | 46.62% |
| Custom Execution Accuracy | 75.53% |
| Execution Errors | 47 / 1,034 |

> The execution accuracy is based on a custom SQLite evaluator and is not the official Spider benchmark score.

## Failure Analysis

Out of 47 execution errors:

- Invalid column: 41
- Invalid table: 4
- Syntax error: 1
- Other: 1

The main failure mode was schema grounding, especially incorrect or hallucinated column names.

## Hugging Face Model

Trained QLoRA adapter:

https://huggingface.co/mohd-maaz/qwen2.5-coder-7b-text-to-sql-qlora

## Notebook

The complete training, evaluation, and failure-analysis pipeline is available in:

`qlora_sql_assistant.ipynb`

## Tech Stack

- Python
- PyTorch
- Transformers
- PEFT
- TRL
- BitsAndBytes
- Hugging Face Datasets
- SQLite
- Kaggle GPU
