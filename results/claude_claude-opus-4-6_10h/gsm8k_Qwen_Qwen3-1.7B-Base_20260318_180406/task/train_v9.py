#!/usr/bin/env python3
"""
SFT training v9: Same as v7 recipe but 3 epochs instead of 4.
v7 (4 epochs) = 74.0%. Fewer epochs = less overfitting.
Also try slightly larger effective batch size for smoother gradients.
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


def extract_answer_math(text):
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
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', text).strip()
    text = re.sub(r'<<[^>]*>>', '', text)
    text = re.sub(r'\\boxed\{[^}]*\}\s*$', '', text).strip()
    return text.strip()


def format_fewshot(q, r, a):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"


def main():
    print("=" * 60)
    print("Training v9: v7 recipe with 3 epochs (less overfitting)")
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
        r = clean_reasoning(ex["answer"])
        gsm8k_parsed.append({"question": ex["question"], "reasoning": r, "answer": a})
    print(f"GSM8K: {len(gsm8k_parsed)}")

    # Parse MetaMathQA - GSM subsets
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    mm_gsm = mm.filter(lambda x: x["type"] in {"GSM_AnsAug", "GSM_Rephrased"}, num_proc=1)
    mm_gsm_parsed = []
    for ex in mm_gsm:
        a = extract_answer(ex["response"])
        if a is None:
            continue
        r = clean_reasoning(ex["response"])
        if not r or len(r) < 20:
            continue
        mm_gsm_parsed.append({"question": ex["query"], "reasoning": r, "answer": a})
    print(f"MetaMathQA GSM subsets: {len(mm_gsm_parsed)}")

    # Parse MetaMathQA - MATH_Rephrased (same as v7)
    mm_math = mm.filter(lambda x: x["type"] == "MATH_Rephrased", num_proc=1)
    mm_math_parsed = []
    for ex in mm_math:
        a = extract_answer_math(ex["response"])
        if a is None:
            continue
        try:
            float(a.replace(',', ''))
        except ValueError:
            continue
        r = clean_reasoning(ex["response"])
        if not r or len(r) < 20 or len(r) > 2000:
            continue
        mm_math_parsed.append({"question": ex["query"], "reasoning": r, "answer": a})
    random.shuffle(mm_math_parsed)
    mm_math_parsed = mm_math_parsed[:30000]
    print(f"MetaMathQA MATH_Rephrased (filtered): {len(mm_math_parsed)}")

    all_data = gsm8k_parsed + mm_gsm_parsed + mm_math_parsed
    random.shuffle(all_data)
    print(f"Total data: {len(all_data)}")

    # Build training examples (same format as v7)
    formatted = []
    for i, ex in enumerate(all_data):
        user_content = MATH_PROMPT_TEMPLATE.format(question=ex["question"])
        assistant_content = f"{ex['reasoning']}\n\nANSWER: {ex['answer']}"

        if i % 5 < 3:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            n_fs = random.randint(3, 8)
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
    per_device_bs = 4
    grad_accum = 4
    eff_bs = per_device_bs * grad_accum * n_gpus

    training_args = SFTConfig(
        output_dir="checkpoints_v9",
        num_train_epochs=3,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=5000,
        save_total_limit=1,
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
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting: 3 epochs, {len(dataset)} examples, {n_gpus} GPUs, eff_bs={eff_bs}")
    trainer.train()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model...")
        trainer.save_model("final_model_v9")
        tokenizer.save_pretrained("final_model_v9")

        # Fix generation config
        gc_path = "final_model_v9/generation_config.json"
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

        # Remove stale index
        idx_path = "final_model_v9/model.safetensors.index.json"
        if os.path.exists(idx_path):
            os.remove(idx_path)

        m = AutoModelForCausalLM.from_pretrained("final_model_v9")
        t = AutoTokenizer.from_pretrained("final_model_v9")
        print(f"Verified: {type(m).__name__}, vocab={t.vocab_size}")
        del m, t

    print("\nTraining v9 complete!")


if __name__ == "__main__":
    main()
