#!/usr/bin/env python3
"""
SFT training v4: Focused, high-quality training for GSM8K.
- GSM8K train (7.5K) + MetaMathQA AnsAug+Rephrased (160K)
- 5 epochs with cosine schedule
- More system message examples (40%) to match eval context
- Fix EOS in generation config
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


def format_fewshot(q, r, a):
    return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"


def main():
    print("=" * 60)
    print("Training v4: Focused GSM8K training")
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
        if a is None: continue
        r = clean_reasoning(ex["answer"])
        gsm8k_parsed.append({"question": ex["question"], "reasoning": r, "answer": a})
    print(f"GSM8K: {len(gsm8k_parsed)}")

    # Parse MetaMathQA (AnsAug + Rephrased only - highest quality)
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    mm = mm.filter(lambda x: x["type"] in {"GSM_AnsAug", "GSM_Rephrased"}, num_proc=1)
    mm_parsed = []
    for ex in mm:
        a = extract_answer(ex["response"])
        if a is None: continue
        r = clean_reasoning(ex["response"])
        if not r or len(r) < 20: continue
        mm_parsed.append({"question": ex["query"], "reasoning": r, "answer": a})
    print(f"MetaMathQA AnsAug+Rephrased: {len(mm_parsed)}")

    all_data = gsm8k_parsed + mm_parsed
    random.shuffle(all_data)
    print(f"Total data: {len(all_data)}")

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
            # Use 3-8 few-shot examples to match eval's 10
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
    eff_bs = 4 * 4 * n_gpus

    training_args = SFTConfig(
        output_dir="checkpoints_v4",
        num_train_epochs=5,
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
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting: 5 epochs, {len(dataset)} examples, {n_gpus} GPUs, eff_bs={eff_bs}")
    trainer.train()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nSaving model...")
        trainer.save_model("final_model")
        tokenizer.save_pretrained("final_model")

        # Fix generation config
        gc_path = "final_model/generation_config.json"
        with open(gc_path, 'r') as f:
            gc = json.load(f)
        eos = gc.get("eos_token_id", [])
        if isinstance(eos, int): eos = [eos]
        if 151645 not in eos: eos.append(151645)
        gc["eos_token_id"] = eos
        with open(gc_path, 'w') as f:
            json.dump(gc, f, indent=2)

        m = AutoModelForCausalLM.from_pretrained("final_model")
        t = AutoTokenizer.from_pretrained("final_model")
        print(f"Verified: {type(m).__name__}, vocab={t.vocab_size}")
        del m, t

    print("\nTraining v4 complete!")


if __name__ == "__main__":
    main()
