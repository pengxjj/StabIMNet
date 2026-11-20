#!/usr/bin/env bash
# CEDAR (Stage-3) contrastive finetune, paper protocol: 5 seeds {42..46}, each
# warm-started from the matching Phase-2 checkpoint opt_phase2_seed{seed}_best.pt,
# stratified 80/10/10 split per seed, bce 2.0 / mse 0.5 / kld 0.5 / lambda_CW 0.01,
# 40 epochs (10 warmup + 30 cosine), wd 1e-6, bs 128, lr 1e-4, decoder_layers=3.
# Reports BOTH the paper's binarized "bug" PPVn and the corrected continuous PPVn.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for seed in 42 43 44 45 46; do
  gpu=$(( (seed - 42) % 4 ))
  save_dir="results/opt_phase3_paper"
  log="results/opt_phase3_paper_seed${seed}.log"
  echo "launch phase3 seed=${seed} gpu=cuda:${gpu} -> ${log}"
  setsid nohup python -u train_opt_phase3_paper.py \
    --seeds "${seed}" \
    --egnn-type opt \
    --decoder-layers 3 \
    --focal-gamma 1.0 \
    --bce-weight 2.0 --mse-weight 0.5 --kld-weight 0.5 \
    --coeff-contrastive 0.01 \
    --epochs 40 --warmup-epochs 10 \
    --lr 1e-4 --weight-decay 1e-6 \
    --batch-size 128 \
    --eval-mc 100 \
    --early-stop 20 \
    --save-dir "${save_dir}" \
    --device "cuda:${gpu}" \
    > "${log}" 2>&1 < /dev/null &
done

echo "launched 5 phase3 seeds (42-46)"
