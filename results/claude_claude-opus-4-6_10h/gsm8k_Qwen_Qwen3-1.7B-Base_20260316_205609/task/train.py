#!/usr/bin/env python3
"""
SFT training script for Qwen3-1.7B-Base on GSM8K math reasoning.
Uses GSM8K train split + MetaMathQA GSM-related examples.
"""

import os
import re
import json
import argparse
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ─── Prompt template (matches evaluate.py) ───
MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="final_model")
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--use_metamath", action="store_true", default=True)
    parser.add_argument("--metamath_gsm_only", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    return parser.parse_args()


def extract_gsm8k_answer(answer_text):
    """Extract numeric answer from GSM8K format: ... #### {number}"""
    match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def clean_gsm8k_reasoning(answer_text):
    """Clean GSM8K reasoning: remove #### and <<calculations>>"""
    # Remove the #### answer line
    reasoning = re.sub(r'####.*$', '', answer_text, flags=re.MULTILINE).strip()
    # Remove <<calculation>> annotations
    reasoning = re.sub(r'<<[^>]*>>', '', reasoning)
    return reasoning.strip()


def extract_metamath_answer(response_text):
    """Extract answer from MetaMathQA format: ... The answer is: {number}"""
    # Try "The answer is: X" format
    match = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', response_text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Try #### format
    match = re.search(r'####\s*([\-\d,\.]+)', response_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def clean_metamath_reasoning(response_text):
    """Clean MetaMathQA reasoning: remove final answer line"""
    # Remove "The answer is: X" at the end
    reasoning = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', response_text).strip()
    # Remove #### line if present
    reasoning = re.sub(r'####\s*[\-\d,\.]+\s*$', '', reasoning, flags=re.MULTILINE).strip()
    # Remove <<calculation>> annotations
    reasoning = re.sub(r'<<[^>]*>>', '', reasoning)
    return reasoning.strip()


def format_as_chat(question, reasoning, answer):
    """Format as chat messages matching the eval prompt format."""
    user_content = MATH_PROMPT_TEMPLATE.format(question=question)
    assistant_content = f"{reasoning}\n\nANSWER: {answer}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def prepare_gsm8k_data():
    """Load and format GSM8K training data."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    formatted = []
    for example in ds:
        answer = extract_gsm8k_answer(example["answer"])
        if answer is None:
            continue
        reasoning = clean_gsm8k_reasoning(example["answer"])
        messages = format_as_chat(example["question"], reasoning, answer)
        formatted.append({"messages": messages})
    print(f"GSM8K train: {len(formatted)} examples")
    return Dataset.from_list(formatted)


def prepare_metamath_data(gsm_only=True):
    """Load and format MetaMathQA data."""
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    if gsm_only:
        # Filter for GSM-related types only
        gsm_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}
        ds = ds.filter(lambda x: x["type"] in gsm_types, num_proc=1)
        print(f"MetaMathQA GSM-only: {len(ds)} examples")

    formatted = []
    skipped = 0
    for example in ds:
        answer = extract_metamath_answer(example["response"])
        if answer is None:
            skipped += 1
            continue
        reasoning = clean_metamath_reasoning(example["response"])
        if not reasoning:
            skipped += 1
            continue
        messages = format_as_chat(example["query"], reasoning, answer)
        formatted.append({"messages": messages})

    print(f"MetaMathQA formatted: {len(formatted)} examples (skipped {skipped})")
    return Dataset.from_list(formatted)


def main():
    args = parse_args()

    print("=" * 60)
    print("Training Qwen3-1.7B-Base for GSM8K")
    print("=" * 60)

    # ─── Load tokenizer ───
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Prepare data ───
    print("\nPreparing training data...")
    gsm8k_data = prepare_gsm8k_data()

    if args.use_metamath:
        metamath_data = prepare_metamath_data(gsm_only=args.metamath_gsm_only)
        train_dataset = concatenate_datasets([gsm8k_data, metamath_data])
    else:
        train_dataset = gsm8k_data

    # Shuffle
    train_dataset = train_dataset.shuffle(seed=args.seed)
    print(f"\nTotal training examples: {len(train_dataset)}")

    # Print a sample
    print("\n--- Sample training example ---")
    sample = train_dataset[0]["messages"]
    for msg in sample:
        print(f"[{msg['role']}]: {msg['content'][:200]}...")
    print("--- End sample ---\n")

    # ─── Load model ───
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # ─── Training config ───
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=True,
    )

    # ─── Trainer ───
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # ─── Train ───
    print(f"\nStarting training: {args.num_train_epochs} epochs, {len(train_dataset)} examples")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    trainer.train()

    # ─── Save ───
    print(f"\nSaving model to {args.final_model_dir}...")
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)

    # Verify
    print("Verifying saved model...")
    from transformers import AutoModelForCausalLM as AM, AutoTokenizer as AT
    m = AM.from_pretrained(args.final_model_dir)
    t = AT.from_pretrained(args.final_model_dir)
    print(f"Model loaded OK: {type(m).__name__}")
    print(f"Tokenizer loaded OK: vocab_size={t.vocab_size}")
    del m, t

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
