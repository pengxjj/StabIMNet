#!/usr/bin/env bash
# Phase-3 (CEDAR) delta-head drop ablation — HybridModelOpt_Comparative_Ablate.
# 2 variants x 5 seeds (42-46), stratified 75/10/15, cc=0.0, 150 ep, warmup 10,
# early-stop 20, lr 1e-4, phase2 warm-start (results/opt_phase2_paper). Full
# reference = existing results/opt_phase3_paper_cc0_751015 (ROC 0.741±0.017),
# not retrained here.
#
#   concat  No-delta     [emb_c ; emb_w]           (cls_in=256)
#   delta   Delta-only   [delta ; ||delta||]       (cls_in=129)
#
# Variants run SEQUENTIALLY (wait between variants) to keep GPU memory bounded;
# within a variant the 5 seeds spread across 4 GPUs via (seed-42)%4.
set -euo pipefail

ESP="/public/home/lifex/pengxj/miniconda3/envs/esm/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results

variants=(
  "concat"
  "delta"
)

for fusion in "${variants[@]}"; do
  echo "=== [$(date +%H:%M:%S)] launching variant: ${fusion} ==="
  for seed in 42 43 44 45 46; do
    gpu=$(( (seed - 42) % 4 ))
    log="results/opt_phase3_ablate_delta_${fusion}_seed${seed}.log"
    setsid nohup "$ESP" -u train_opt_phase3_ablate.py \
      --fusion "${fusion}" \
      --seeds "${seed}" \
      --egnn-type opt \
      --decoder-layers 3 \
      --focal-gamma 1.0 \
      --bce-weight 2.0 --mse-weight 0.5 --kld-weight 0.5 \
      --coeff-contrastive 0.0 \
      --epochs 150 --warmup-epochs 10 \
      --lr 1e-4 --weight-decay 1e-6 \
      --batch-size 128 \
      --split 0.75 0.10 0.15 \
      --eval-mc 100 \
      --early-stop 20 \
      --save-dir "results/opt_phase3_ablate_delta" \
      --device "cuda:${gpu}" \
      > "${log}" 2>&1 < /dev/null &
  done
  wait
  echo "=== [$(date +%H:%M:%S)] variant ${fusion} done ==="
done

echo "ALL Phase-3 delta-head ablation variants done."
