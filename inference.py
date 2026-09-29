import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_MODEL = "mohd-maaz/qwen2.5-coder-7b-text-to-sql-qlora"

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. "
    "Generate only a valid SQL query using the provided schema. "
    "Do not invent tables or columns."
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load base model in 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    dtype=torch.float16,
    device_map="auto",
)


# Attach trained LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL,
)

model.eval()


def generate_sql(schema, question):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Question:\n{question}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    sql = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return sql


if __name__ == "__main__":
    schema = """
    singer(
        Singer_ID,
        Name,
        Country,
        Song_Name,
        Song_release_year,
        Age
    )
    """

    question = "How many singers are there?"

    sql = generate_sql(schema, question)

    print("Generated SQL:")
    print(sql)
