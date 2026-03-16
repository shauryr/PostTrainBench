#!/usr/bin/env python3
"""
SFT training script for Qwen3-1.7B-Base on math reasoning.
Uses MetaMathQA (GSM types) + GSM8K train data.
Formats output to match the GSM8K evaluation prompt (ANSWER: X format).
"""

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer, SFTConfig

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = "./sft_output"
FINAL_MODEL_DIR = "./final_model"
MAX_SEQ_LENGTH = 1536
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.05

SYSTEM_MSG = "You are a helpful assistant that solves math problems step by step."

PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def format_chat(system: str, user: str, assistant: str) -> str:
    """Format as Qwen3 chat template."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def extract_metamath_answer(response: str) -> tuple[str, str]:
    """Extract reasoning and final answer from MetaMathQA response."""
    # MetaMathQA ends with "The answer is: X"
    match = re.search(r'The answer is:\s*(.+?)\.?\s*$', response, re.MULTILINE)
    if match:
        final_answer = match.group(1).strip()
        reasoning = response[:match.start()].strip()
        return reasoning, final_answer

    # Fallback: try to find boxed answer
    match = re.search(r'\\boxed\{(.+?)\}', response)
    if match:
        final_answer = match.group(1).strip()
        reasoning = response.strip()
        return reasoning, final_answer

    # Last resort
    return response.strip(), response.strip().split('\n')[-1].strip()


def extract_gsm8k_answer(answer: str) -> tuple[str, str]:
    """Extract reasoning and final answer from GSM8K format."""
    # GSM8K uses #### as delimiter
    parts = answer.split("####")
    if len(parts) >= 2:
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
        # Remove calculator annotations <<...>>
        reasoning = re.sub(r'<<.*?>>', '', reasoning)
        return reasoning, final_answer
    return answer.strip(), ""


def process_metamath(examples):
    """Process MetaMathQA examples."""
    texts = []
    for query, response in zip(examples['query'], examples['response']):
        reasoning, final_answer = extract_metamath_answer(response)
        user_msg = PROMPT_TEMPLATE.format(question=query)
        assistant_msg = f"{reasoning}\n\nANSWER: {final_answer}"
        text = format_chat(SYSTEM_MSG, user_msg, assistant_msg)
        texts.append(text)
    return {"text": texts}


def process_gsm8k(examples):
    """Process GSM8K train examples."""
    texts = []
    for question, answer in zip(examples['question'], examples['answer']):
        reasoning, final_answer = extract_gsm8k_answer(answer)
        if not final_answer:
            continue
        user_msg = PROMPT_TEMPLATE.format(question=question)
        assistant_msg = f"{reasoning}\n\nANSWER: {final_answer}"
        text = format_chat(SYSTEM_MSG, user_msg, assistant_msg)
        texts.append(text)
    return {"text": texts}


def main():
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # Load MetaMathQA - all GSM types + some MATH types
    metamath = load_dataset("meta-math/MetaMathQA", split="train")

    # All GSM types (240K)
    gsm_types = ['GSM_AnsAug', 'GSM_Rephrased', 'GSM_FOBAR', 'GSM_SV']
    gsm_metamath = metamath.filter(lambda x: x['type'] in gsm_types, num_proc=1)
    print(f"MetaMathQA GSM types: {len(gsm_metamath)}")

    # Some MATH types for broader reasoning (50K)
    math_metamath = metamath.filter(lambda x: x['type'] not in gsm_types, num_proc=1)
    math_metamath = math_metamath.shuffle(seed=42).select(range(min(50000, len(math_metamath))))
    print(f"MetaMathQA MATH types (sampled): {len(math_metamath)}")

    # Load GSM8K train set
    gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
    print(f"GSM8K train: {len(gsm8k_train)}")

    # Process datasets
    print("Processing MetaMathQA GSM...")
    gsm_metamath_processed = gsm_metamath.map(
        process_metamath, batched=True, num_proc=1,
        remove_columns=gsm_metamath.column_names
    )

    print("Processing MetaMathQA MATH...")
    math_metamath_processed = math_metamath.map(
        process_metamath, batched=True, num_proc=1,
        remove_columns=math_metamath.column_names
    )

    print("Processing GSM8K train...")
    gsm8k_processed = gsm8k_train.map(
        process_gsm8k, batched=True, num_proc=1,
        remove_columns=gsm8k_train.column_names
    )

    # Combine and shuffle
    train_data = concatenate_datasets([
        gsm_metamath_processed,
        math_metamath_processed,
        gsm8k_processed,
    ]).shuffle(seed=42)

    # Filter empty
    train_data = train_data.filter(lambda x: len(x['text']) > 100, num_proc=1)
    print(f"\nTotal training examples: {len(train_data)}")
    print(f"Sample:\n{train_data[0]['text'][:500]}...")

    # ── Load model and tokenizer ──
    print("\n" + "=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # ── Training ──
    print("\n" + "=" * 60)
    print("Starting SFT training...")
    print("=" * 60)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        dataloader_num_workers=4,
        gradient_checkpointing=False,  # Not needed with H200s
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    trainer.train()

    # ── Save final model ──
    print("\n" + "=" * 60)
    print(f"Saving model to {FINAL_MODEL_DIR}...")
    print("=" * 60)

    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    # Verify
    print("Verifying saved model...")
    from transformers import AutoModelForCausalLM as AM, AutoTokenizer as AT
    m = AM.from_pretrained(FINAL_MODEL_DIR)
    t = AT.from_pretrained(FINAL_MODEL_DIR)
    print(f"Model loaded OK: {type(m).__name__}")
    print(f"Tokenizer loaded OK: vocab_size={t.vocab_size}")
    print("DONE!")


if __name__ == "__main__":
    main()
