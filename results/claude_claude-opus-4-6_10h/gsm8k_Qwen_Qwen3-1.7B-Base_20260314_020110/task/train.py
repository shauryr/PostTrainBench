#!/usr/bin/env python3
"""SFT training script for Qwen3-1.7B-Base on GSM8K math reasoning."""

import os
import re
import json
import torch
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig


MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def clean_gsm8k_answer(answer_text):
    """Convert GSM8K answer format to clean step-by-step + ANSWER format."""
    # Split on ####
    parts = answer_text.split("####")
    if len(parts) != 2:
        return None

    reasoning = parts[0].strip()
    final_answer = parts[1].strip()

    # Remove calculator annotations like <<48/2=24>>
    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)

    return f"{reasoning}\n\nANSWER: {final_answer}"


def clean_metamath_answer(response_text):
    """Convert MetaMathQA response format to ANSWER format."""
    # MetaMathQA responses end with "The answer is: X"
    match = re.search(r'The answer is:\s*(.+?)\.?\s*$', response_text)
    if not match:
        return None

    final_answer = match.group(1).strip()

    # Remove the "The answer is:" line and replace with ANSWER format
    reasoning = re.sub(r'\s*The answer is:\s*.+?\.?\s*$', '', response_text).strip()

    # Clean up any \boxed{} commands
    # Extract value from \boxed{...}
    def extract_boxed(m):
        return m.group(1)
    reasoning = re.sub(r'\\boxed\{([^}]+)\}', extract_boxed, reasoning)

    # Clean final_answer too
    final_answer = re.sub(r'\\boxed\{([^}]+)\}', extract_boxed, final_answer)

    # Try to extract numeric value for GSM-type questions
    # Remove dollar signs, commas, percent signs for numeric answers
    clean_answer = final_answer.replace('$', '').replace(',', '').replace('%', '').strip()

    return f"{reasoning}\n\nANSWER: {final_answer}"


def format_as_chat(question, answer):
    """Format as a chat conversation for SFT."""
    user_content = MATH_PROMPT_TEMPLATE.format(question=question)
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }


def prepare_gsm8k_data():
    """Load and format GSM8K train data."""
    ds = load_dataset("gsm8k", "main", split="train")

    formatted = []
    for ex in ds:
        answer = clean_gsm8k_answer(ex["answer"])
        if answer:
            formatted.append(format_as_chat(ex["question"], answer))

    print(f"GSM8K train: {len(formatted)} examples")
    return formatted


def prepare_metamath_data(max_per_type=None):
    """Load and format MetaMathQA GSM-related data."""
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    # Filter to GSM-related types only
    gsm_types = ["GSM_AnsAug", "GSM_Rephrased", "GSM_SV", "GSM_FOBAR"]

    formatted = []
    type_counts = {}

    for ex in ds:
        if ex["type"] not in gsm_types:
            continue

        if max_per_type and type_counts.get(ex["type"], 0) >= max_per_type:
            continue

        answer = clean_metamath_answer(ex["response"])
        if answer:
            formatted.append(format_as_chat(ex["query"], answer))
            type_counts[ex["type"]] = type_counts.get(ex["type"], 0) + 1

    print(f"MetaMathQA GSM: {len(formatted)} examples")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--final-model-dir", default="./final_model")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--metamath-max-per-type", type=int, default=20000)
    parser.add_argument("--use-metamath", action="store_true", default=True)
    parser.add_argument("--no-metamath", action="store_true", default=False)
    args = parser.parse_args()

    use_metamath = args.use_metamath and not args.no_metamath

    print("=" * 60)
    print("Preparing training data...")
    print("=" * 60)

    # Prepare data
    all_data = prepare_gsm8k_data()

    if use_metamath:
        metamath_data = prepare_metamath_data(max_per_type=args.metamath_max_per_type)
        all_data.extend(metamath_data)

    print(f"\nTotal training examples: {len(all_data)}")

    # Create HF dataset
    train_dataset = Dataset.from_list(all_data)

    # Shuffle
    train_dataset = train_dataset.shuffle(seed=42)

    print(f"\nSample training example:")
    print(json.dumps(train_dataset[0], indent=2)[:500])

    print("\n" + "=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print(f"Saving model to {args.final_model_dir}...")
    print("=" * 60)

    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
