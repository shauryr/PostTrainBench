#!/usr/bin/env python3
"""
SFT training script v3 for Qwen3-1.7B-Base on GSM8K math reasoning.
Key improvements over v2:
- More data: ALL GSM types from MetaMathQA + orca-math
- More epochs (6)
- Better data formatting with diverse prompts
- Proper EOS token in generation config
"""

import os
import re
import json
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
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def clean_reasoning(text):
    """Clean reasoning text."""
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', text).strip()
    text = re.sub(r'<<[^>]*>>', '', text)
    return text.strip()


def format_fewshot_example(question, reasoning, answer):
    """Format as a few-shot example."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def prepare_gsm8k_data():
    """Load and parse GSM8K training data."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    parsed = []
    for ex in ds:
        answer = extract_answer(ex["answer"])
        if answer is None:
            continue
        reasoning = clean_reasoning(ex["answer"])
        parsed.append({"question": ex["question"], "reasoning": reasoning, "answer": answer})
    print(f"GSM8K parsed: {len(parsed)} examples")
    return parsed


def prepare_metamath_data():
    """Load ALL GSM types from MetaMathQA."""
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}
    ds = ds.filter(lambda x: x["type"] in gsm_types, num_proc=1)
    print(f"MetaMathQA GSM filtered: {len(ds)} examples")

    parsed = []
    for ex in ds:
        answer = extract_answer(ex["response"])
        if answer is None:
            continue
        reasoning = clean_reasoning(ex["response"])
        if not reasoning or len(reasoning) < 20:
            continue
        parsed.append({"question": ex["query"], "reasoning": reasoning, "answer": answer})
    print(f"MetaMathQA parsed: {len(parsed)} examples")
    return parsed


def prepare_orca_math():
    """Load orca-math word problems."""
    try:
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        print(f"Orca-math loaded: {len(ds)} examples")
    except Exception as e:
        print(f"Could not load orca-math: {e}")
        return []

    parsed = []
    for ex in ds:
        question = ex.get("question", "")
        answer_text = ex.get("answer", "")
        if not question or not answer_text:
            continue

        # Try to extract a numeric final answer
        # Look for patterns like "the answer is X" or "= X" at the end
        answer_num = None
        m = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', answer_text)
        if m:
            answer_num = m.group(1).replace(',', '').strip()
        else:
            # Try last number in the text
            numbers = re.findall(r'[\-]?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
            if numbers:
                answer_num = numbers[-1].replace(',', '')

        if answer_num is None:
            continue

        reasoning = answer_text
        # Clean up
        reasoning = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*\.?\s*$', '', reasoning).strip()

        if len(reasoning) < 20:
            continue

        parsed.append({"question": question, "reasoning": reasoning, "answer": answer_num})

    print(f"Orca-math parsed: {len(parsed)} examples")
    # Sample to keep balanced
    if len(parsed) > 30000:
        parsed = random.sample(parsed, 30000)
        print(f"Orca-math sampled: {len(parsed)}")
    return parsed


def build_training_dataset(gsm8k_data, metamath_data, orca_data):
    """Build training dataset with various formats."""
    random.seed(42)

    all_data = gsm8k_data + metamath_data + orca_data
    random.shuffle(all_data)

    formatted = []
    for i, ex in enumerate(all_data):
        question = ex["question"]
        reasoning = ex["reasoning"]
        answer = ex["answer"]

        user_content = MATH_PROMPT_TEMPLATE.format(question=question)
        assistant_content = f"{reasoning}\n\nANSWER: {answer}"

        # 70% simple format, 30% with system message (few-shot)
        if i % 10 < 7:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            n_fewshot = random.randint(2, 6)
            fewshot_pool = gsm8k_data  # Use GSM8K for few-shot examples
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

    return Dataset.from_list(formatted)


def main():
    print("=" * 60)
    print("Training Qwen3-1.7B-Base for GSM8K (v3)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    gsm8k_data = prepare_gsm8k_data()
    metamath_data = prepare_metamath_data()
    orca_data = prepare_orca_math()

    train_dataset = build_training_dataset(gsm8k_data, metamath_data, orca_data)
    train_dataset = train_dataset.shuffle(seed=42)
    print(f"\nTotal training examples: {len(train_dataset)}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    n_gpus = torch.cuda.device_count()
    total_examples = len(train_dataset)
    eff_bs = 4 * 4 * n_gpus
    steps_per_epoch = total_examples // eff_bs
    total_steps = steps_per_epoch * 6

    print(f"GPUs: {n_gpus}, Effective batch size: {eff_bs}")
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    training_args = SFTConfig(
        output_dir="checkpoints_v3",
        num_train_epochs=6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        max_length=1536,
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting training v3: 6 epochs, {total_examples} examples")
    trainer.train()

    # Save (rank 0 only)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model to final_model...")
        trainer.save_model("final_model")
        tokenizer.save_pretrained("final_model")

        # Fix generation config to include <|im_end|> as EOS
        gen_config_path = "final_model/generation_config.json"
        with open(gen_config_path, 'r') as f:
            gen_config = json.load(f)

        # Add <|im_end|> token (151645) as EOS
        eos_ids = gen_config.get("eos_token_id", [])
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if 151645 not in eos_ids:
            eos_ids.append(151645)
        gen_config["eos_token_id"] = eos_ids

        with open(gen_config_path, 'w') as f:
            json.dump(gen_config, f, indent=2)

        print("Fixed generation_config with <|im_end|> EOS token")

        # Verify
        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Model OK: {type(m).__name__}, Tokenizer OK: vocab_size={t.vocab_size}")
        del m, t

    print("\nTraining v3 complete!")


if __name__ == "__main__":
    main()
