#!/usr/bin/env python3
"""SFT training script for Qwen3-1.7B-Base on GSM8K-style math reasoning.
Optimized for single H200 GPU."""

import os
import re
import argparse
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def extract_answer_from_gsm8k(answer_text: str) -> tuple:
    DELIM = "####"
    parts = answer_text.split(DELIM)
    final_answer = parts[-1].strip()
    reasoning = DELIM.join(parts[:-1]).strip()
    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)
    return reasoning, final_answer


def extract_answer_from_metamath(response: str) -> tuple:
    match = re.search(r'The answer is:\s*(.+?)$', response, re.MULTILINE)
    if match:
        final_answer = match.group(1).strip()
        reasoning = response[:match.start()].strip()
    else:
        parts = response.split("####")
        if len(parts) > 1:
            final_answer = parts[-1].strip()
            reasoning = "####".join(parts[:-1]).strip()
        else:
            lines = response.strip().split('\n')
            final_answer = lines[-1].strip()
            reasoning = '\n'.join(lines[:-1]).strip()

    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', final_answer)
    if boxed_match:
        final_answer = boxed_match.group(1)
    return reasoning, final_answer


def format_example_as_chat(question: str, reasoning: str, answer: str) -> list:
    user_content = MATH_PROMPT_TEMPLATE.format(prompt=question)
    assistant_content = f"{reasoning}\n\nANSWER: {answer}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def prepare_gsm8k_data():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for ex in ds:
        reasoning, answer = extract_answer_from_gsm8k(ex["answer"])
        chat = format_example_as_chat(ex["question"], reasoning, answer)
        examples.append({"messages": chat})
    return Dataset.from_list(examples)


def prepare_metamath_gsm_data(max_per_type=None):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}
    type_counts = {t: 0 for t in gsm_types}
    examples = []
    for ex in ds:
        if ex["type"] not in gsm_types:
            continue
        if max_per_type and type_counts[ex["type"]] >= max_per_type:
            continue
        reasoning, answer = extract_answer_from_metamath(ex["response"])
        clean_answer = answer.replace(',', '').replace('$', '').replace('%', '').strip()
        try:
            float(clean_answer)
        except ValueError:
            continue
        chat = format_example_as_chat(ex["query"], reasoning, answer)
        examples.append({"messages": chat})
        type_counts[ex["type"]] += 1
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output-dir", default=os.path.join(WORK_DIR, "final_model"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--no-metamath", action="store_true")
    parser.add_argument("--metamath-per-type", type=int, default=20000,
                        help="Max examples per MetaMathQA type")
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("Loading tokenizer and model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    gsm8k_data = prepare_gsm8k_data()
    print(f"GSM8K train examples: {len(gsm8k_data)}")

    if not args.no_metamath:
        metamath_data = prepare_metamath_gsm_data(max_per_type=args.metamath_per_type)
        print(f"MetaMathQA GSM examples: {len(metamath_data)}")
        train_dataset = concatenate_datasets([gsm8k_data, metamath_data])
    else:
        train_dataset = gsm8k_data

    train_dataset = train_dataset.shuffle(seed=42)
    print(f"Total training examples: {len(train_dataset)}")

    # Check token lengths to make sure our max_length is reasonable
    sample = train_dataset[0]
    formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    tokens = tokenizer.encode(formatted)
    print(f"Sample token length: {len(tokens)}")
    print(f"Max sequence length: {args.max_seq_length}")
    print("\n--- Sample formatted example ---")
    print(formatted[:500])
    print("...\n--- End sample ---\n")

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    checkpoint_dir = os.path.join(WORK_DIR, "checkpoints")
    effective_batch = args.batch_size * args.grad_accum
    total_steps = int(len(train_dataset) * args.epochs / effective_batch)
    print(f"Effective batch size: {effective_batch}")
    print(f"Estimated total steps: {total_steps}")
    print(f"Estimated time (at ~2.5s/step): {total_steps * 2.5 / 3600:.1f} hours")

    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        max_length=args.max_seq_length,
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
