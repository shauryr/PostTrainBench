# GSM8K Training Findings (2 runs completed)

## Best Model: v7 (74.0% on 150 samples)
- Checkpoint at `checkpoint_v7/` — complete fine-tuned Qwen3-1.7B-Base model (74.0%)
- Also available: `checkpoint_v4/` — previous best (73.3%)
- Best training script: `train_v7.py`

## Run History
| Run | Version | Accuracy | Key Change |
|-----|---------|----------|------------|
| 1 | v1 | 10% | Incomplete save, garbage output |
| 1 | v2 | 72% | EOS token fix = breakthrough |
| 1 | v3 | 69.3% | Added orca-math — hurt |
| 1 | v4 | 73.3% | Refined v2 data mix, 5 epochs |
| 2 | v5 | 64.7% | Continue from v4 + NuminaMath — hurt |
| 2 | v6 | 69.3% | Fresh from base, 7 epochs, GSM8K 3x oversample |
| 2 | **v7** | **74.0%** | **4 epochs + MATH_Rephrased diversity** |
| 2 | v8 | 62.0% | GSM8K-only polish — catastrophic forgetting |

## What Works
1. **Data mix (proven)**: GSM8K train (7.5K) + MetaMathQA GSM_AnsAug + GSM_Rephrased (~160K) + **MetaMathQA MATH_Rephrased (30K, filtered to numeric answers only)**. Total ~197K examples.
2. **Format**: Qwen3 chat format. 60% plain user/assistant, 40% with few-shot system message (3-8 examples).
3. **Output format**: Model must produce `ANSWER: {number}` — the eval uses `match(numeric=True)`.
4. **Hyperparams**: 4 epochs (not 5 — slight overfitting), LR 5e-5, cosine schedule, warmup 0.03, weight_decay 0.01, bf16.
5. **EOS token**: MUST include token ID 151645 in generation_config.json. Without this → 0% accuracy.
6. **Training from base model**: Fresh training from Qwen3-1.7B-Base outperforms continuing from checkpoint.

## What DOES NOT Work (confirmed across 2 runs)
1. **NuminaMath data**: Hurts in both runs. Don't use it.
2. **Orca-math data**: Hurts. Don't use it.
3. **External math datasets** outside MetaMathQA family: Generally hurt.
4. **Catastrophic forgetting**: Fine-tuning a trained model on narrow data (e.g., GSM8K-only) destroys performance (74% → 62%).
5. **Continuing training from checkpoint with new data**: v5 continued from v4 + new data → 64.7%. Worse than training fresh.
6. **Too many epochs**: 7 epochs overfit (v6: 69.3%). 5 epochs slightly overfit (v4: 73.3%). 4 epochs is optimal.
7. **GSM8K oversampling (3x)**: Didn't help (v6: 69.3%).

## What To Try Next (SFT is hitting ~74% ceiling)
1. **Data quality filtering**: Use the v7 model to score training examples, keep only those it gets wrong (hard example mining).
2. **Rejection sampling**: Generate multiple solutions with v7, keep only correct ones as training data.
3. **DPO/ORPO**: Train on preference pairs (correct vs incorrect solutions).
4. **Ensemble checkpoints**: Average weights from v4 and v7 checkpoints.
5. **Different LR schedules**: WSD (warmup-stable-decay) instead of cosine.
6. **Longer max_length**: Current 1536 may truncate some reasoning chains.

## Hardware
- 8x NVIDIA H200 GPUs, 140GB VRAM each (1.12TB total GPU memory)
- 128 CPU cores, 1.5TB system RAM
- Model (1.7B params, bf16) uses ~3.4GB per GPU in DDP
- Use `num_proc=64` for datasets.map() tokenization
- Use `dataloader_num_workers=16` for training
- Few-shot system messages create long sequences → batch_size=4 with gradient_checkpointing is safe. batch_size=16 causes OOM.

## Evaluation
- `python3 evaluate.py --model-path final_model --limit 150 --max-connections 2 --gpu-memory-utilization 0.3`
- Uses vLLM backend via inspect-ai
- 10 few-shot examples are prepended by the evaluation framework
- Chat template: `templates/qwen3.jinja` (adds `<think>` block)
