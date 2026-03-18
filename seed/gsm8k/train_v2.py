#!/usr/bin/env python3
"""
SFT training script v2 for Qwen3-1.7B-Base on GSM8K math reasoning.
Key improvements:
- More epochs (4)
- System messages with few-shot examples in training data
- Better save handling
- Higher learning rate
"""

import os
import re
import random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ─── Prompt template (matches evaluate.py EXACTLY) ───
MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def extract_answer(text):
    """Extract numeric answer from various formats."""
    # Try #### format
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    # Try "The answer is: X"
    m = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def clean_reasoning(text):
    """Clean reasoning text."""
    # Remove #### line
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    # Remove "The answer is: X"
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', text).strip()
    # Remove <<calculations>>
    text = re.sub(r'<<[^>]*>>', '', text)
    return text.strip()


def format_fewshot_example(question, reasoning, answer):
    """Format as a few-shot example (matches eval format)."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def prepare_data():
    """Prepare training data with different formats."""
    random.seed(42)

    # Load GSM8K train
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    # Parse GSM8K examples
    gsm8k_parsed = []
    for ex in gsm8k:
        answer = extract_answer(ex["answer"])
        if answer is None:
            continue
        reasoning = clean_reasoning(ex["answer"])
        gsm8k_parsed.append({
            "question": ex["question"],
            "reasoning": reasoning,
            "answer": answer,
        })
    print(f"GSM8K parsed: {len(gsm8k_parsed)} examples")

    # Load MetaMathQA (GSM types only)
    metamath = load_dataset("meta-math/MetaMathQA", split="train")
    metamath = metamath.filter(
        lambda x: x["type"] in {"GSM_AnsAug", "GSM_Rephrased"},
        num_proc=1,
    )
    print(f"MetaMathQA filtered: {len(metamath)} examples")

    metamath_parsed = []
    for ex in metamath:
        answer = extract_answer(ex["response"])
        if answer is None:
            continue
        reasoning = clean_reasoning(ex["response"])
        if not reasoning or len(reasoning) < 20:
            continue
        metamath_parsed.append({
            "question": ex["query"],
            "reasoning": reasoning,
            "answer": answer,
        })
    print(f"MetaMathQA parsed: {len(metamath_parsed)} examples")

    # Sample MetaMathQA to keep training manageable
    if len(metamath_parsed) > 40000:
        metamath_parsed = random.sample(metamath_parsed, 40000)
        print(f"MetaMathQA sampled: {len(metamath_parsed)} examples")

    all_examples = gsm8k_parsed + metamath_parsed
    random.shuffle(all_examples)

    # Build training dataset with multiple formats:
    # 1. Simple format (no system message) - 60%
    # 2. With system message containing few-shot examples - 40%
    formatted = []

    for i, ex in enumerate(all_examples):
        question = ex["question"]
        reasoning = ex["reasoning"]
        answer = ex["answer"]

        user_content = MATH_PROMPT_TEMPLATE.format(question=question)
        assistant_content = f"{reasoning}\n\nANSWER: {answer}"

        if i % 5 < 3:
            # Format 1: Simple user-assistant (60%)
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            # Format 2: With system message containing few-shot examples (40%)
            # Pick 2-5 random few-shot examples (not the current one)
            n_fewshot = random.randint(2, 5)
            fewshot_pool = [e for j, e in enumerate(gsm8k_parsed) if j != i % len(gsm8k_parsed)]
            fewshot_examples = random.sample(fewshot_pool, min(n_fewshot, len(fewshot_pool)))

            system_content = "\n\n".join([
                format_fewshot_example(fe["question"], fe["reasoning"], fe["answer"])
                for fe in fewshot_examples
            ])

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

        formatted.append({"messages": messages})

    print(f"\nTotal training examples: {len(formatted)}")
    return Dataset.from_list(formatted)


def main():
    print("=" * 60)
    print("Training Qwen3-1.7B-Base for GSM8K (v2)")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    print("\nPreparing training data...")
    train_dataset = prepare_data()

    # Shuffle
    train_dataset = train_dataset.shuffle(seed=42)

    # Print samples
    for idx in [0, len(train_dataset) // 2]:
        print(f"\n--- Sample {idx} ---")
        sample = train_dataset[idx]["messages"]
        for msg in sample:
            role = msg["role"]
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            print(f"[{role}]: {content}")
        print("--- End sample ---")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Training config
    training_args = SFTConfig(
        output_dir="checkpoints_v2",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        max_length=1536,
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    n_gpus = torch.cuda.device_count()
    eff_bs = 4 * 4 * n_gpus
    print(f"\nStarting training: 4 epochs, {len(train_dataset)} examples")
    print(f"GPUs: {n_gpus}, Effective batch size: {eff_bs}")
    print(f"Steps per epoch: ~{len(train_dataset) // eff_bs}")
    trainer.train()

    # Save model (only on rank 0 in distributed training)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model to final_model...")
        trainer.save_model("final_model")
        tokenizer.save_pretrained("final_model")

        # Verify
        print("Verifying saved model...")
        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Model loaded OK: {type(m).__name__}")
        print(f"Tokenizer loaded OK: vocab_size={t.vocab_size}")
        del m, t

    print("\n" + "=" * 60)
    print("Training v2 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
