#!/usr/bin/env bash
# Phase-2 (IEDB) tower drop-one ablation — HybridModelOpt_Ablate.
# 2 variants x 5 seeds (42-46), stratified 75/10/15, loss 2.0/0.5/0.5, 200 ep,
# early-stop 20, lr 1e-4 / VAE 3e-5, phase1 warm-start (same protocol as
# run_opt_phase2_paper_751015.sh). Full reference = existing
# results/opt_phase2_paper_751015 (ROC 0.882±0.005), not retrained here.
#
# Variants run SEQUENTIALLY (wait between variants) to keep GPU memory bounded;
# within a variant the 5 seeds spread across 4 GPUs via (seed-42)%4.
#
#   seq prop     No-struct  (cls_in=80)
#   struct prop  No-seq     (cls_in=80)
set -euo pipefail

ESP="/public/home/lifex/pengxj/miniconda3/envs/esm/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results

variants=(
  "seq prop"
  "struct prop"
)

for fuse in "${variants[@]}"; do
  fuse_tag=$(echo "$fuse" | tr ' ' '-')
  echo "=== [$(date +%H:%M:%S)] launching variant: ${fuse_tag} ==="
  for seed in 42 43 44 45 46; do
    gpu=$(( (seed - 42) % 4 ))
    log="results/opt_phase2_ablate_${fuse_tag}_seed${seed}.log"
    setsid nohup "$ESP" -u train_opt_phase2_ablate.py \
      --fuse $fuse \
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
      --save-dir "results/opt_phase2_ablate" \
      --device "cuda:${gpu}" \
      > "${log}" 2>&1 < /dev/null &
  done
  wait
  echo "=== [$(date +%H:%M:%S)] variant ${fuse_tag} done ==="
done

echo "ALL Phase-2 ablation variants done."
