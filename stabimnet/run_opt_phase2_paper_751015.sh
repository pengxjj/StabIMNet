#!/usr/bin/env bash
# dec3 (HybridModelOpt) IEDB Phase-2 — stratified 75/10/15, loss 2.0/0.5/0.5.
# Seeds 42-46, one process/seed across 4 GPUs. This is the IEDB half of the
# "original IEDB vs dec3 IEDB" comparison (original = run_orig_iedb_751015.sh).
# Distinct save-dir so the CEDAR phase3 backbone (results/opt_phase2_paper) is
# left untouched.
set -euo pipefail

ESP="/public/home/lifex/pengxj/miniconda3/envs/esm/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results

for seed in 42 43 44 45 46; do
  gpu=$(( (seed - 42) % 4 ))
  save_dir="results/opt_phase2_paper_751015"
  log="results/opt_phase2_paper_751015_seed${seed}.log"
  echo "launch dec3-IEDB 75/10/15 seed=${seed} gpu=cuda:${gpu} -> ${log}"
  setsid nohup "$ESP" -u train_opt_phase2_paper.py \
    --seeds "${seed}" \
    --egnn-type opt \
    --decoder-layers 3 \
    --focal-gamma 1.0 \
    --bce-weight 2.0 --mse-weight 0.5 --kld-weight 0.5 \
    --epochs 200 --early-stop 20 \
    --lr 1e-4 --lr-vae 3e-5 \
    --weight-decay 1e-6 \
    --batch-size 128 \
    --split 0.75 0.10 0.15 \
    --save-dir "${save_dir}" \
    --device "cuda:${gpu}" \
    > "${log}" 2>&1 < /dev/null &
done

echo "launched 5 dec3-IEDB 75/10/15 seeds (42-46)"
