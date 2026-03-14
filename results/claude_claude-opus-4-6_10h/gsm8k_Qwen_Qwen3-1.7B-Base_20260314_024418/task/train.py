#!/usr/bin/env python3
"""SFT training script for Qwen3-1.7B-Base on GSM8K-style math reasoning."""

import os
import re
import json
import argparse
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def extract_answer_from_gsm8k(answer_text: str) -> tuple[str, str]:
    """Extract reasoning and final answer from GSM8K format."""
    DELIM = "####"
    parts = answer_text.split(DELIM)
    final_answer = parts[-1].strip()
    reasoning = DELIM.join(parts[:-1]).strip()
    # Clean up the calculator annotations like <<48/2=24>>
    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)
    return reasoning, final_answer


def extract_answer_from_metamath(response: str) -> tuple[str, str]:
    """Extract reasoning and final answer from MetaMathQA format."""
    # MetaMathQA ends with "The answer is: X" or "#### X"
    # Try "The answer is:" first
    match = re.search(r'The answer is:\s*(.+?)$', response, re.MULTILINE)
    if match:
        final_answer = match.group(1).strip()
        reasoning = response[:match.start()].strip()
    else:
        # Try #### format
        parts = response.split("####")
        if len(parts) > 1:
            final_answer = parts[-1].strip()
            reasoning = "####".join(parts[:-1]).strip()
        else:
            # Fallback: last line is the answer
            lines = response.strip().split('\n')
            final_answer = lines[-1].strip()
            reasoning = '\n'.join(lines[:-1]).strip()

    # Clean up calculator annotations
    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)
    # Clean up \boxed{} commands
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', final_answer)
    if boxed_match:
        final_answer = boxed_match.group(1)
    return reasoning, final_answer


def format_example_as_chat(question: str, reasoning: str, answer: str) -> list[dict]:
    """Format a single example as a chat conversation."""
    user_content = MATH_PROMPT_TEMPLATE.format(prompt=question)
    assistant_content = f"{reasoning}\n\nANSWER: {answer}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def prepare_gsm8k_data():
    """Load and format GSM8K training data."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for ex in ds:
        reasoning, answer = extract_answer_from_gsm8k(ex["answer"])
        chat = format_example_as_chat(ex["question"], reasoning, answer)
        examples.append({"messages": chat})
    return Dataset.from_list(examples)


def prepare_metamath_gsm_data():
    """Load and format MetaMathQA GSM-related data."""
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    # Filter to GSM-related types only
    gsm_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}
    examples = []
    for ex in ds:
        if ex["type"] not in gsm_types:
            continue
        reasoning, answer = extract_answer_from_metamath(ex["response"])
        # Skip if we can't extract a clean numeric answer
        clean_answer = answer.replace(',', '').replace('$', '').replace('%', '').strip()
        try:
            float(clean_answer)
        except ValueError:
            continue
        chat = format_example_as_chat(ex["query"], reasoning, answer)
        examples.append({"messages": chat})
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output-dir", default=os.path.join(WORK_DIR, "final_model"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--use-metamath", action="store_true", default=True)
    parser.add_argument("--no-metamath", action="store_true")
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()

    if args.no_metamath:
        args.use_metamath = False

    print("=" * 60)
    print("Loading tokenizer and model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the chat template from the jinja file
    template_path = os.path.join(WORK_DIR, "templates", "qwen3.jinja")
    with open(template_path) as f:
        chat_template = f.read()
    tokenizer.chat_template = chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    print("=" * 60)
    print("Preparing training data...")
    print("=" * 60)

    # Prepare GSM8K data
    gsm8k_data = prepare_gsm8k_data()
    print(f"GSM8K train examples: {len(gsm8k_data)}")

    if args.use_metamath:
        metamath_data = prepare_metamath_gsm_data()
        print(f"MetaMathQA GSM examples: {len(metamath_data)}")
        train_dataset = concatenate_datasets([gsm8k_data, metamath_data])
    else:
        train_dataset = gsm8k_data

    # Shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    print(f"Total training examples: {len(train_dataset)}")

    # Print a sample
    sample = train_dataset[0]
    formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    print("\n--- Sample formatted example ---")
    print(formatted[:500])
    print("...")
    print("--- End sample ---\n")

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    checkpoint_dir = os.path.join(WORK_DIR, "checkpoints")

    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print("=" * 60)
    print(f"Saving model to {args.output_dir}")
    print("=" * 60)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
