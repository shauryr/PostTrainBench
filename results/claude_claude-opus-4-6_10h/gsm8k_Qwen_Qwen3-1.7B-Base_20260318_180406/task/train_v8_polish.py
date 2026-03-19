#!/usr/bin/env python3
"""
SFT training v8: Quick polish from v7 checkpoint.
- Continue from v7 (74.0% accuracy)
- Only GSM8K train data (7.5K examples) - focus on exact task format
- 2 epochs at very low LR (5e-6) - gentle fine-tuning
- This is a "task adaptation" step
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
    return None


def clean_reasoning(text):
    text = re.sub(r'####.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'[Tt]he answer is:?\s*[\-\d,\.]+\s*$', '', text).strip()
    text = re.sub(r'<<[^>]*>>', '', text)
    return text.strip()


def format_fewshot(q, r, a):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"


def main():
    print("=" * 60)
    print("Training v8: Polish from v7 with GSM8K focus")
    print("=" * 60)
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained("final_model_v7_backup", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parse GSM8K only
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_parsed = []
    for ex in gsm8k:
        a = extract_answer(ex["answer"])
        if a is None:
            continue
        r = clean_reasoning(ex["answer"])
        gsm8k_parsed.append({"question": ex["question"], "reasoning": r, "answer": a})
    print(f"GSM8K: {len(gsm8k_parsed)}")

    # Build training examples - ALL with few-shot context (to match eval)
    formatted = []
    for i, ex in enumerate(gsm8k_parsed):
        user_content = MATH_PROMPT_TEMPLATE.format(question=ex["question"])
        assistant_content = f"{ex['reasoning']}\n\nANSWER: {ex['answer']}"

        # 50% with few-shot, 50% plain
        if i % 2 == 0:
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

    # Load model from v7
    model = AutoModelForCausalLM.from_pretrained(
        "final_model_v7_backup",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    n_gpus = torch.cuda.device_count()

    training_args = SFTConfig(
        output_dir="checkpoints_v8",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=500,
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

    eff_bs = 4 * 4 * n_gpus
    print(f"\nStarting: 2 epochs, {len(dataset)} examples, {n_gpus} GPUs, eff_bs={eff_bs}, LR=5e-6")
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

        # Remove stale index
        idx_path = "final_model/model.safetensors.index.json"
        if os.path.exists(idx_path):
            os.remove(idx_path)

        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Verified: {type(m).__name__}, vocab={t.vocab_size}")
        del m, t

    print("\nTraining v8 complete!")


if __name__ == "__main__":
    main()
