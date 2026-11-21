#!/bin/bash
# =============================================================================
# HybridModelv7 (full-MHC-sequence cross-attention) cross-locus generalization
# run + full-data run for ITSNdb eval. Self-gating: a single "canary" seed runs
# first; only if its first epoch completes without NaN does the pipeline fan out.
#
#   0. canary gate  : seed 42, holdout-locus B (A->B) — wait for "Ep 1" in log
#   1. A->B         : 4 seeds (holdout-locus B)
#   2. B->A         : 4 seeds (holdout-locus A)
#   3. aggregate    : per-direction mean +/- sd
#   4. full-data    : 4 seeds (holdout none) -> checkpoints for ITSNdb eval
# =============================================================================
cd /public/home/lifex/pengxj/bio_models/ImmunoStruct-main/StructIM-hla
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm

SEEDS="42 123 456 789"
DIR_B=results/v7_hlaholdout_locusB
DIR_A=results/v7_hlaholdout_locusA
DIR_FULL=results/v7_full
MASTER_LOG=results/v7_master.log

mkdir -p "$DIR_A" "$DIR_B" "$DIR_FULL"

log() { echo "[v7 $(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

log "==================== V7 FULL-MHC RUN START ===================="

# ── Phase 0: canary gate (seed 42, A->B) ──────────────────────────────
log "Phase 0: canary seed 42 (holdout-locus B) on GPU 1"
nohup python train_v7_hlaholdout.py --holdout-locus B --seeds 42 --gpu 1 \
    --save-dir results/v7_hlaholdout \
    > "${DIR_B}/seed_42.stdout" 2>&1 &
CANARY_PID=$!
log "  canary pid=$CANARY_PID -> ${DIR_B}/seed_42.log"

log "Phase 0: waiting for first epoch (up to 70 min; import ~8 min + load ~20 min) ..."
GATE=0
for i in $(seq 1 70); do
  if grep -q "Ep   1/" "${DIR_B}/seed_42.log" 2>/dev/null; then GATE=1; break; fi
  if grep -qE "missing (pep|HLA) perres|SystemExit|Error|Traceback" "${DIR_B}/seed_42.log" 2>/dev/null; then
    log "Phase 0: ERROR in canary — aborting. tail of log:"
    tail -20 "${DIR_B}/seed_42.log" | tee -a "$MASTER_LOG"
    exit 1
  fi
  if ! kill -0 "$CANARY_PID" 2>/dev/null && [ ! -f "${DIR_B}/seed_42_metrics.json" ]; then
    log "Phase 0: canary process died before metrics — aborting. tail of log:"
    tail -20 "${DIR_B}/seed_42.log" | tee -a "$MASTER_LOG"
    exit 1
  fi
  sleep 60
done

if [ "$GATE" != "1" ]; then
  log "Phase 0: TIMEOUT waiting for canary epoch 1 — aborting."
  tail -30 "${DIR_B}/seed_42.log" | tee -a "$MASTER_LOG"
  exit 1
fi
log "Phase 0 PASSED — canary reached epoch 1 without NaN. Fanning out."

# ── Phase 1: A->B (holdout-locus B), remaining 3 seeds ────────────────
log "Phase 1: A->B (holdout-locus B) seeds 123 456 789"
GPUS=(0 2 3)
i=0
for s in 123 456 789; do
  gpu=${GPUS[$i]}
  nohup python train_v7_hlaholdout.py --holdout-locus B --seeds "$s" --gpu "$gpu" \
      --save-dir results/v7_hlaholdout \
      > "${DIR_B}/seed_${s}.stdout" 2>&1 &
  log "  locusB seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

# ── Phase 2: B->A (holdout-locus A), 4 seeds ──────────────────────────
log "Phase 2: B->A (holdout-locus A) seeds 42 123 456 789"
GPUS=(1 0 2 3)
i=0
for s in $SEEDS; do
  gpu=${GPUS[$i]}
  nohup python train_v7_hlaholdout.py --holdout-locus A --seeds "$s" --gpu "$gpu" \
      --save-dir results/v7_hlaholdout \
      > "${DIR_A}/seed_${s}.stdout" 2>&1 &
  log "  locusA seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

log "Phase 1+2: waiting for all cross-locus metrics ..."
while true; do
  missing=""
  for s in $SEEDS; do
    [ -f "$DIR_B/seed_${s}_metrics.json" ] || missing="$missing B:$s"
    [ -f "$DIR_A/seed_${s}_metrics.json" ] || missing="$missing A:$s"
  done
  [ -z "$missing" ] && break
  sleep 120
done
log "Cross-locus training done."

# ── Phase 3: aggregate ────────────────────────────────────────────────
log "Phase 3: aggregate cross-locus"
python aggregate_seeds.py --save-dir "$DIR_B" --seeds 42 123 456 789 | tee -a "$MASTER_LOG"
python aggregate_seeds.py --save-dir "$DIR_A" --seeds 42 123 456 789 | tee -a "$MASTER_LOG"

# ── Phase 4: full-data (holdout none) 4 seeds, for ITSNdb eval ────────
log "Phase 4: full-data 4 seeds (holdout none)"
i=0
for s in $SEEDS; do
  gpu=${GPUS[$i]}
  nohup python train_v7_hlaholdout.py --holdout-locus none --seeds "$s" --gpu "$gpu" \
      --save-dir results/v7_full \
      > "${DIR_FULL}/seed_${s}.stdout" 2>&1 &
  log "  full seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

log "Phase 4: waiting for full-data metrics ..."
while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_FULL/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 120
done
log "Full-data training done."

log "==================== V7 RUN DONE $(date '+%F %T') ===================="
log "Next: python eval_v7_itsndb.py --seeds $SEEDS --dataset both --gpu 0 --save-dir results/v7_full"
