#!/usr/bin/env python3
"""
SFT training v6: Fresh training from base with proven data + optimized hyperparams.
Key changes from v4:
- 7 epochs instead of 5 (more training on same good data)
- Oversample GSM8K 3x within the dataset (more weight on actual task)
- Larger effective batch size
- max_length=1536 (same as v4)
- Stricter answer formatting (always end with ANSWER: {integer})
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


def extract_answer(text):
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def clean_reasoning(text):
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', text).strip()
    text = re.sub(r'<<[^>]*>>', '', text)
    return text.strip()


def normalize_answer(a):
    """Normalize answer to clean integer/decimal string."""
    a = a.replace(',', '').strip()
    try:
        val = float(a)
        if val == int(val):
            return str(int(val))
        return a
    except:
        return a


def format_fewshot(q, r, a):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"


def main():
    print("=" * 60)
    print("Training v6: Fresh from base, proven data, optimized hyperparams")
    print("=" * 60)
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parse GSM8K
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_parsed = []
    for ex in gsm8k:
        a = extract_answer(ex["answer"])
        if a is None:
            continue
        a = normalize_answer(a)
        r = clean_reasoning(ex["answer"])
        if not r or len(r) < 10:
            continue
        gsm8k_parsed.append({"question": ex["question"], "reasoning": r, "answer": a})
    print(f"GSM8K: {len(gsm8k_parsed)}")

    # Parse MetaMathQA (AnsAug + Rephrased only)
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    mm = mm.filter(lambda x: x["type"] in {"GSM_AnsAug", "GSM_Rephrased"}, num_proc=64)
    mm_parsed = []
    for ex in mm:
        a = extract_answer(ex["response"])
        if a is None:
            continue
        a = normalize_answer(a)
        r = clean_reasoning(ex["response"])
        if not r or len(r) < 20:
            continue
        mm_parsed.append({"question": ex["query"], "reasoning": r, "answer": a})
    print(f"MetaMathQA AnsAug+Rephrased: {len(mm_parsed)}")

    # Oversample GSM8K 3x within the dataset
    all_data = gsm8k_parsed * 3 + mm_parsed
    random.shuffle(all_data)
    print(f"Total data (GSM8K 3x + MetaMathQA): {len(all_data)}")

    # Build training examples
    formatted = []
    for i, ex in enumerate(all_data):
        user_content = MATH_PROMPT_TEMPLATE.format(question=ex["question"])
        assistant_content = f"{ex['reasoning']}\n\nANSWER: {ex['answer']}"

        # 60% simple, 40% with few-shot system message
        if i % 5 < 3:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            n_fs = random.randint(3, 6)
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

    dataset = Dataset.from_list(formatted).shuffle(seed=42)
    print(f"Training examples: {len(dataset)}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    n_gpus = torch.cuda.device_count()
    per_device_bs = 8
    grad_accum = 2
    eff_bs = per_device_bs * grad_accum * n_gpus

    training_args = SFTConfig(
        output_dir="checkpoints_v6",
        num_train_epochs=7,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        max_length=1536,
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

    print(f"\nStarting: 7 epochs, {len(dataset)} examples, {n_gpus} GPUs, eff_bs={eff_bs}")
    trainer.train()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model...")
        trainer.save_model("final_model")
        tokenizer.save_pretrained("final_model")

        # Fix generation config
        gc_path = "final_model/generation_config.json"
        if os.path.exists(gc_path):
            with open(gc_path, 'r') as f:
                gc = json.load(f)
        else:
            gc = {}
        eos = gc.get("eos_token_id", [])
        if isinstance(eos, int):
            eos = [eos]
        if 151645 not in eos:
            eos.append(151645)
        gc["eos_token_id"] = eos
        with open(gc_path, 'w') as f:
            json.dump(gc, f, indent=2)

        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Verified: {type(m).__name__}, vocab={t.vocab_size}")
        del m, t

    print("\nTraining v6 complete!")


if __name__ == "__main__":
    main()
