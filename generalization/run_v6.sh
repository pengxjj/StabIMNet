#!/bin/bash
# =============================================================================
# HybridModelv6 (sequence-primary cross-attention) cross-locus generalization run.
# Per-residue peptide↔HLA-pseudo-seq cross-attention + EGNN structure auxiliary.
# Both directions (A->B and B->A), 4 seeds each, then aggregate.
#   0. dry-run gate (locus A)
#   1. full 4-seed A->B (holdout-locus B)
#   2. full 4-seed B->A (holdout-locus A)
#   3. aggregate
# =============================================================================
cd /public/home/lifex/pengxj/bio_models/ImmunoStruct-main/StructIM-hla
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm

SEEDS="42 123 456 789"
DIR_B=results/v6_hlaholdout_locusB
DIR_A=results/v6_hlaholdout_locusA
DRYRUN_DIR=results/v6_dryrun_locusA
MASTER_LOG=results/v6_master.log

mkdir -p "$DIR_A" "$DIR_B"

log() { echo "[v6 $(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

log "==================== V6 CROSS-ATTN RUN START ===================="

# ── Phase 0: dry-run gate (SKIPPED — already verified manually) ──────
log "Phase 0: dry-run gate skipped (verified: Val ROC 0.82 @1ep, no NaN)"

# ── Phase 1: A->B (holdout locus B) ───────────────────────────────────
log "Phase 1: A->B (holdout locus B) 4 seeds"
GPUS=(0 1 2 3)
i=0
for s in $SEEDS; do
  gpu=${GPUS[$i]}
  nohup python train_v6_hlaholdout.py --holdout-locus B --seeds "$s" --gpu "$gpu" \
      --save-dir results/v6_hlaholdout \
      > "${DIR_B}/seed_${s}.stdout" 2>&1 &
  log "  locusB seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_B/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 60
done
log "Phase 1 done (locus B) $(date '+%F %T')"

# ── Phase 2: B->A (holdout locus A) ───────────────────────────────────
log "Phase 2: B->A (holdout locus A) 4 seeds"
i=0
for s in $SEEDS; do
  gpu=${GPUS[$i]}
  nohup python train_v6_hlaholdout.py --holdout-locus A --seeds "$s" --gpu "$gpu" \
      --save-dir results/v6_hlaholdout \
      > "${DIR_A}/seed_${s}.stdout" 2>&1 &
  log "  locusA seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_A/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 60
done
log "Phase 2 done (locus A) $(date '+%F %T')"

# ── Phase 3: aggregate ────────────────────────────────────────────────
log "Phase 3: aggregate"
python aggregate_seeds.py --save-dir "$DIR_B" --seeds 42 123 456 789 | tee -a "$MASTER_LOG"
python aggregate_seeds.py --save-dir "$DIR_A" --seeds 42 123 456 789 | tee -a "$MASTER_LOG"

log "==================== V6 CROSS-ATTN RUN DONE $(date '+%F %T') ===================="
