#!/usr/bin/env bash
# ============================================================================
# Smoke test for the StabIMNet release package.
#
# Runs the core StabIMNet stage-2 (IEDB) training path end-to-end on the small
# data_smoke/ subset, warm-starting the Phase-1 VAE from the provided seed-42
# weights. This exercises: data loading (graphs + CSV + HLA), the Phase-1 VAE
# weight transfer (input_dim 283*21 = 5943), the HybridModelOpt forward/backward,
# and the full paper metric protocol — with a 2-epoch / single-seed budget so it
# finishes in a few minutes.
#
# The full three-stage reproduction (all 5 seeds, CEDAR, clinical, HLA holdout,
# TESLA/NEPdb, TCR) is driven by reproduce_all.sh in the sibling
# ImmunoStruct-main/ repository; this script only proves the packaged code runs.
#
# Usage:
#   bash run_smoke.sh              # auto-detect device (cuda:0 if present, else cpu)
#   ESP=/path/to/python bash run_smoke.sh
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESP="${ESP:-/public/home/lifex/pengxj/miniconda3/envs/esm/bin/python}"

# Prefer a GPU if one is visible; otherwise fall back to CPU.
if "$ESP" -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null | grep -q '^1$'; then
  DEVICE="cuda:0"
else
  DEVICE="cpu"
fi

cd "$ROOT/stabimnet"

echo "== StabIMNet stage-2 smoke train (device=$DEVICE) =="
echo "   data_smoke: $ROOT/data_smoke"
echo "   stage1 VAE: $ROOT/weights/stage1_vae_presentation.pt"

"$ESP" -u train_opt_phase2_paper.py \
  --graph-dir "$ROOT/data_smoke/graph_pyg_IEDB_esmfold2/" \
  --property-path "$ROOT/data_smoke/ImmunoStruct_IEDB_data.csv" \
  --hla-path "$ROOT/data_smoke/HLA_allele_sequences.csv" \
  --phase1-ckpt "$ROOT/weights/stage1_vae_presentation.pt" \
  --save-dir "$ROOT/results_smoke/opt_phase2" \
  --seeds 42 \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 0 \
  --device "$DEVICE"

echo ""
echo "smoke test complete. Checkpoint written to $ROOT/results_smoke/opt_phase2/"
