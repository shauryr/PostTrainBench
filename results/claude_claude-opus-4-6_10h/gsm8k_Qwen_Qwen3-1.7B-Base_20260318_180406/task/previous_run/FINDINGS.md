# GSM8K Training Findings (Previous Run: 73.3% accuracy)

## Best Model: v4 (73.3% on 150 samples)
- Checkpoint at `checkpoint_v4/` — a complete fine-tuned Qwen3-1.7B-Base model
- Training script at `train_v4.py` — the exact code that produced this result

## What Works
1. **Data**: GSM8K train (7.5K) + MetaMathQA AnsAug+Rephrased subsets (~160K). Total ~167K examples.
2. **Format**: Qwen3 chat format with `<think>` blocks. 60% plain user/assistant, 40% with few-shot system message.
3. **Output format**: Model must produce `ANSWER: {number}` — the eval uses `match(numeric=True)` to extract it.
4. **Hyperparams**: 5 epochs, LR 5e-5, cosine schedule, warmup 0.03, weight_decay 0.01, bf16.
5. **EOS token**: Must include token ID 151645 in generation_config.json `eos_token_id` list. Without this, model generates endlessly and accuracy drops to 0%.
6. **Multi-GPU**: Use `accelerate launch` with the provided `accelerate_config.yaml` (8 GPUs, DDP).

## What Doesn't Work
1. **Orca-math data**: Adding microsoft/orca-math-word-problems HURT accuracy (72% → 69.3%). Stick to GSM8K + MetaMathQA.
2. **Too many epochs on diverse data**: 6 epochs with extra data sources performed worse than 5 focused epochs.
3. **Incomplete model saves**: Multi-GPU training with SFTTrainer only saves weights on rank 0. Always save BOTH model and tokenizer, and verify with `AutoModelForCausalLM.from_pretrained()`.

## What To Try Next
1. **Larger batch size**: Previous run used batch_size=4 per GPU with gradient_checkpointing. Each H200 has 140GB VRAM, model is only 3.4GB. Try batch_size=32+ per GPU WITHOUT gradient_checkpointing.
2. **More SFT data**: NuminaMath-CoT, MathInstruct, TIGER-Lab/MathInstruct, or other high-quality math datasets.
3. **Continue training from checkpoint_v4**: Instead of training from base model, load `checkpoint_v4/` and do additional epochs with new/different data.
4. **Curriculum learning**: Start with easy problems, gradually increase difficulty.
5. **Data filtering**: Filter training data by answer correctness or reasoning quality.
6. **Higher `--gpu-memory-utilization`**: Use 0.8 instead of default 0.3 when running evaluate.py for faster inference.

## Hardware
- 8x NVIDIA H200 GPUs, 140GB VRAM each (1.12TB total GPU memory)
- 128 CPU cores, 1.5TB system RAM
- The model (1.7B params, bf16) uses ~3.4GB per GPU in DDP
- Use `num_proc=64` for datasets.map() tokenization (not 1)
- Use `dataloader_num_workers=16` for training

## Evaluation
- `python3 evaluate.py --model-path final_model --limit 150 --max-connections 2 --gpu-memory-utilization 0.3`
- Uses vLLM backend via inspect-ai
- 10 few-shot examples are prepended by the evaluation framework
- Chat template: `templates/qwen3.jinja` (adds `<think>` block)
