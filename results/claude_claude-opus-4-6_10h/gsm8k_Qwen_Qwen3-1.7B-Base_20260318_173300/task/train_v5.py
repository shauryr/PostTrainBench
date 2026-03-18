#!/usr/bin/env python3
"""
SFT training v5: Improved GSM8K training.

Changes from v4:
1. Reasoning inside <think> blocks (proper Qwen3 thinking format)
2. GSM8K upsampled 5x (increase ratio from 4.5% to ~19%)
3. Larger batch size (16 per GPU), no gradient checkpointing
4. max_length 2048 (from 1536)
5. 50% few-shot with 5-10 examples (from 40% with 3-8)
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
    """Extract numeric answer from various formats."""
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'[Tt]he answer is:?\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'\\boxed\{([\-\d,\.]+)\}', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def clean_reasoning(text):
    """Clean reasoning text by removing answer markers and annotations."""
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*\.?\s*$', '', text).strip()
    text = re.sub(r'\\boxed\{[\-\d,\.]+\}\.?\s*$', '', text).strip()
    text = re.sub(r'<<[^>]*>>', '', text)
    return text.strip()


def format_fewshot(q, r, a):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"


def main():
    print("=" * 60)
    print("Training v5: Think blocks + upsampled GSM8K")
    print("=" * 60)
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Load GSM8K ===
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_parsed = []
    for ex in gsm8k:
        a = extract_answer(ex["answer"])
        if a is None:
            continue
        r = clean_reasoning(ex["answer"])
        if len(r) < 20:
            continue
        gsm8k_parsed.append({"question": ex["question"], "reasoning": r, "answer": a})
    print(f"GSM8K: {len(gsm8k_parsed)}")

    # === Load MetaMathQA ===
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    mm = mm.filter(lambda x: x["type"] in {"GSM_AnsAug", "GSM_Rephrased"}, num_proc=1)
    mm_parsed = []
    for ex in mm:
        a = extract_answer(ex["response"])
        if a is None:
            continue
        r = clean_reasoning(ex["response"])
        if not r or len(r) < 20:
            continue
        mm_parsed.append({"question": ex["query"], "reasoning": r, "answer": a})
    print(f"MetaMathQA AnsAug+Rephrased: {len(mm_parsed)}")

    # === Upsample GSM8K 5x ===
    gsm8k_upsampled = gsm8k_parsed * 5
    print(f"GSM8K upsampled 5x: {len(gsm8k_upsampled)}")

    all_data = gsm8k_upsampled + mm_parsed
    random.shuffle(all_data)
    print(f"Total data: {len(all_data)} (GSM8K ratio: {len(gsm8k_upsampled)/len(all_data)*100:.1f}%)")

    # === Format training examples ===
    formatted = []
    for i, ex in enumerate(all_data):
        user_content = MATH_PROMPT_TEMPLATE.format(question=ex["question"])
        # Reasoning inside <think> blocks
        assistant_content = f"<think>\n{ex['reasoning']}\n</think>\n\nANSWER: {ex['answer']}"

        # 50% with few-shot system message
        if i % 2 == 0:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            n_fs = random.randint(5, 10)
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

    # === Load model ===
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    n_gpus = torch.cuda.device_count()

    training_args = SFTConfig(
        output_dir="checkpoints_v5",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        max_length=2048,
        seed=42,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    eff_bs = 16 * n_gpus
    print(f"\nStarting: 5 epochs, {len(dataset)} examples, {n_gpus} GPUs, eff_bs={eff_bs}")
    trainer.train()

    # === Save model ===
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model...")
        trainer.save_model("final_model")
        tokenizer.save_pretrained("final_model")

        # Fix generation config - ensure EOS token 151645
        gc_path = "final_model/generation_config.json"
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

        # Verify model loads correctly
        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Verified: {type(m).__name__}, vocab={t.vocab_size}")
        del m, t

    print("\nTraining v5 complete!")


if __name__ == "__main__":
    main()
