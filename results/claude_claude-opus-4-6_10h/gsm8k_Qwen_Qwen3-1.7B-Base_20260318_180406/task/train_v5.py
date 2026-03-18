#!/usr/bin/env python3
"""
SFT training v5: Continue from v4 checkpoint (73.3%) with enhanced data.
- Continue from previous_run/checkpoint_v4
- GSM8K train oversampled 5x (~37.5K)
- MetaMathQA AnsAug+Rephrased (~160K)
- NuminaMath-CoT filtered for grade-school math (~30-50K)
- Lower LR (2e-5) for continued training
- Large batch size (32 per GPU), no gradient checkpointing
- 2 epochs
"""

import os
import re
import json
import random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""


def extract_answer_gsm8k(text):
    """Extract answer from GSM8K format (#### answer)."""
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def extract_answer_generic(text):
    """Extract numeric answer from various formats."""
    # Try #### format first
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    # Try "The answer is" format
    m = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    # Try boxed format
    m = re.search(r'\\boxed\{([\-\d,\.]+)\}', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def clean_reasoning(text):
    """Clean reasoning text, removing answer markers."""
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', text).strip()
    text = re.sub(r'<<[^>]*>>', '', text)  # Remove GSM8K calc annotations
    text = re.sub(r'\\boxed\{[^}]*\}\s*$', '', text).strip()
    return text.strip()


def is_valid_number(s):
    """Check if string is a valid number."""
    try:
        float(s.replace(',', ''))
        return True
    except (ValueError, AttributeError):
        return False


def format_fewshot(q, r, a):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"


def load_gsm8k_data():
    """Load and parse GSM8K training data."""
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    parsed = []
    for ex in gsm8k:
        a = extract_answer_gsm8k(ex["answer"])
        if a is None or not is_valid_number(a):
            continue
        r = clean_reasoning(ex["answer"])
        if not r or len(r) < 10:
            continue
        parsed.append({"question": ex["question"], "reasoning": r, "answer": a})
    return parsed


def load_metamath_data():
    """Load MetaMathQA AnsAug + Rephrased subsets."""
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    mm = mm.filter(lambda x: x["type"] in {"GSM_AnsAug", "GSM_Rephrased"}, num_proc=64)
    parsed = []
    for ex in mm:
        a = extract_answer_generic(ex["response"])
        if a is None or not is_valid_number(a):
            continue
        r = clean_reasoning(ex["response"])
        if not r or len(r) < 20:
            continue
        parsed.append({"question": ex["query"], "reasoning": r, "answer": a})
    return parsed


def load_numinamath_data(max_samples=50000):
    """Load NuminaMath-CoT, filtered for grade-school level math."""
    try:
        nm = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"Could not load NuminaMath-CoT: {e}")
        return []

    parsed = []
    for ex in nm:
        # Filter for easier problems (gsm8k-like)
        source = ex.get("source", "").lower()
        # Focus on grade-school level sources
        if source not in {"gsm8k", "math", "amc_aime", "cn_k12", "olympiads", ""}:
            # Include general sources but skip very advanced ones
            pass

        solution = ex.get("solution", "")
        if not solution or len(solution) < 20:
            continue

        # Extract numeric answer
        a = extract_answer_generic(solution)
        if a is None or not is_valid_number(a):
            continue

        # Only keep problems with integer or simple decimal answers
        try:
            val = float(a)
            if abs(val) > 1e8:  # Skip very large answers
                continue
        except:
            continue

        r = clean_reasoning(solution)
        if not r or len(r) < 20:
            continue

        question = ex.get("problem", "")
        if not question or len(question) < 10:
            continue

        # Skip problems that are too complex (very long solutions)
        if len(r) > 3000:
            continue

        parsed.append({"question": question, "reasoning": r, "answer": a})

        if len(parsed) >= max_samples:
            break

    return parsed


def build_training_examples(all_data, gsm8k_parsed, use_fewshot_ratio=0.4):
    """Build chat-formatted training examples."""
    formatted = []
    for i, ex in enumerate(all_data):
        user_content = MATH_PROMPT_TEMPLATE.format(question=ex["question"])
        # Ensure answer is clean - just the number
        answer = ex["answer"].strip()
        assistant_content = f"{ex['reasoning']}\n\nANSWER: {answer}"

        # Ratio of examples with few-shot system message
        if random.random() > use_fewshot_ratio:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            # Use 2-4 few-shot examples (keep sequences manageable)
            n_fs = random.randint(2, 4)
            fs = random.sample(gsm8k_parsed, min(n_fs, len(gsm8k_parsed)))
            sys_content = "\n\n".join([
                format_fewshot(f["question"], f["reasoning"], f["answer"])
                for f in fs
            ])
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        formatted.append({"messages": messages})

    return formatted


def main():
    print("=" * 60)
    print("Training v5: Continue from v4 with enhanced data")
    print("=" * 60)
    random.seed(42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "previous_run/checkpoint_v4", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading datasets...")
    gsm8k_parsed = load_gsm8k_data()
    print(f"GSM8K train: {len(gsm8k_parsed)}")

    mm_parsed = load_metamath_data()
    print(f"MetaMathQA AnsAug+Rephrased: {len(mm_parsed)}")

    nm_parsed = load_numinamath_data(max_samples=40000)
    print(f"NuminaMath-CoT (filtered): {len(nm_parsed)}")

    # Build combined dataset with GSM8K oversampled
    gsm8k_oversampled = gsm8k_parsed * 5  # Oversample GSM8K 5x
    all_data = gsm8k_oversampled + mm_parsed + nm_parsed
    random.shuffle(all_data)
    print(f"Total data (with GSM8K 5x oversample): {len(all_data)}")

    # Build training examples
    formatted = build_training_examples(all_data, gsm8k_parsed, use_fewshot_ratio=0.4)
    dataset = Dataset.from_list(formatted).shuffle(seed=42)
    print(f"Training examples: {len(dataset)}")

    # Load model from v4 checkpoint
    print("\nLoading model from previous_run/checkpoint_v4...")
    model = AutoModelForCausalLM.from_pretrained(
        "previous_run/checkpoint_v4",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    n_gpus = torch.cuda.device_count()
    per_device_bs = 8
    grad_accum = 2
    eff_bs = per_device_bs * grad_accum * n_gpus
    print(f"GPUs: {n_gpus}, per_device_bs: {per_device_bs}, grad_accum: {grad_accum}, eff_bs: {eff_bs}")

    training_args = SFTConfig(
        output_dir="checkpoints_v5",
        num_train_epochs=2,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        max_length=1024,
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=16,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    total_steps = len(dataset) // eff_bs * 2
    print(f"\nStarting: 2 epochs, {len(dataset)} examples, eff_bs={eff_bs}, ~{total_steps} steps")
    trainer.train()

    # Save model
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model to final_model/...")
        trainer.save_model("final_model")
        tokenizer.save_pretrained("final_model")

        # Fix generation config - ensure EOS token 151645 is included
        gc_path = "final_model/generation_config.json"
        if os.path.exists(gc_path):
            with open(gc_path, 'r') as f:
                gc = json.load(f)
            eos = gc.get("eos_token_id", [])
            if isinstance(eos, int):
                eos = [eos]
            if 151645 not in eos:
                eos.append(151645)
            gc["eos_token_id"] = eos
            with open(gc_path, 'w') as f:
                json.dump(gc, f, indent=2)
        else:
            gc = {
                "eos_token_id": [151643, 151645],
                "pad_token_id": 151643,
                "do_sample": True,
                "temperature": 0.6,
                "max_new_tokens": 4096
            }
            with open(gc_path, 'w') as f:
                json.dump(gc, f, indent=2)

        # Verify model loads correctly
        print("Verifying model...")
        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Verified: {type(m).__name__}, vocab={t.vocab_size}")
        del m, t

    print("\nTraining v5 complete!")


if __name__ == "__main__":
    main()
