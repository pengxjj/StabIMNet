#!/usr/bin/env bash
# CEDAR (Stage-3) contrastive finetune — cc=1.0 COMPARISON (paper protocol, 150 ep).
# Same dec3 backbone + loss weights as the cc=0.01 run, only coeff-contrastive raised
# to 1.0 and epochs extended to 150 to test whether a stronger contrastive pull helps.
# Distinct save-dir/logs so it does NOT clobber the cc=0.01 run. GPU offset by +1
# relative to cc=0.01 (which used gpu=(seed-42)%4) to balance load across 4 GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for seed in 42 43 44 45 46; do
  gpu=$(( (seed - 42 + 1) % 4 ))
  save_dir="results/opt_phase3_paper_cc1"
  log="results/opt_phase3_paper_cc1_seed${seed}.log"
  echo "launch phase3 cc=1.0 seed=${seed} gpu=cuda:${gpu} -> ${log}"
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
    --eval-mc 100 \
    --early-stop 20 \
    --save-dir "${save_dir}" \
    --device "cuda:${gpu}" \
    > "${log}" 2>&1 < /dev/null &
done

echo "launched 5 phase3 cc=1.0 seeds (42-46)"
