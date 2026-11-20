#!/usr/bin/env bash
# CEDAR (Stage-3) contrastive finetune — cc=1.0, 150 ep, split 75/10/15.
# Same dec3 backbone + loss weights as the cc=1.0 80/10/10 run, but the stratified
# holdout is 75% train / 10% val / 15% test (larger test set: n~407 vs 271).
# Distinct save-dir/logs; launched AFTER the cc=1.0 80/10/10 run finishes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for seed in 42 43 44 45 46; do
  gpu=$(( (seed - 42) % 4 ))
  save_dir="results/opt_phase3_paper_cc1_751015"
  log="results/opt_phase3_paper_cc1_751015_seed${seed}.log"
  echo "launch phase3 cc=1.0 75/10/15 seed=${seed} gpu=cuda:${gpu} -> ${log}"
  setsid nohup python -u train_opt_phase3_paper.py \
    --seeds "${seed}" \
    --egnn-type opt \
    --decoder-layers 3 \
    --focal-gamma 1.0 \
    --bce-weight 2.0 --mse-weight 0.5 --kld-weight 0.5 \
    --coeff-contrastive 1.0 \
    --epochs 150 --warmup-epochs 10 \
    --lr 1e-4 --weight-decay 1e-6 \
    --batch-size 128 \
    --split 0.75 0.10 0.15 \
    --eval-mc 100 \
    --early-stop 20 \
    --save-dir "${save_dir}" \
    --device "cuda:${gpu}" \
    > "${log}" 2>&1 < /dev/null &
done

echo "launched 5 phase3 cc=1.0 75/10/15 seeds (42-46)"
