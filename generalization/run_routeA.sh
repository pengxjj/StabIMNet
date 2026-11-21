#!/bin/bash
# =============================================================================
# Route A — pan-allelic EL pretrain -> IEDB immunogenicity finetune.
# One finetune process PER SEED across 4 GPUs (parallel), mirroring run_v7.sh.
#
#   0. wait for stage-1 seqtower (train_routeA_pretrain.py, launched separately)
#   1. A->B  : 4 seeds (holdout-locus B) on GPUs 0,1,2,3 (one per GPU)
#   2. B->A  : 4 seeds (holdout-locus A) on GPUs 0,1,2,3
#   3. aggregate per-direction mean +/- sd
#   4. full-data : 4 seeds (holdout none) -> ITSNdb eval
# =============================================================================
cd /public/home/lifex/pengxj/bio_models/ImmunoStruct-main/StructIM-hla
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm

SEEDS="42 123 456 789"
SEQTOWER="results/routeA_pretrain/el_pretrain_seed42_seqtower.pt"
DIR_B=results/routeA_hlaholdout_locusB
DIR_A=results/routeA_hlaholdout_locusA
DIR_FULL=results/routeA_full
MASTER_LOG=results/routeA_master.log

mkdir -p "$DIR_A" "$DIR_B" "$DIR_FULL"
log() { echo "[routeA $(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

log "==================== ROUTE A RUN START ===================="

# ── Phase 0: wait for stage-1 seqtower ─────────────────────────────────
log "Phase 0: waiting for seqtower $SEQTOWER"
while [ ! -f "$SEQTOWER" ]; do sleep 60; done
log "Phase 0: seqtower present."

# ── Phase 1: A->B (holdout-locus B), 4 seeds, one process per GPU ──────
log "Phase 1: A->B (holdout-locus B) 4 seeds on GPUs 0 1 2 3"
i=0
for s in $SEEDS; do
  gpu=$i
  nohup python train_routeA_finetune.py --holdout-locus B --seeds "$s" --gpu "$gpu" \
      --pretrained-seq-path "$SEQTOWER" --save-dir results/routeA_hlaholdout \
      > "${DIR_B}/seed_${s}.stdout" 2>&1 &
  log "  locusB seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

log "Phase 1: waiting for A->B metrics ..."
while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_B/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 300
done
log "A->B finetune done."

# ── Phase 2: B->A (holdout-locus A), 4 seeds, one process per GPU ──────
log "Phase 2: B->A (holdout-locus A) 4 seeds on GPUs 0 1 2 3"
i=0
for s in $SEEDS; do
  gpu=$i
  nohup python train_routeA_finetune.py --holdout-locus A --seeds "$s" --gpu "$gpu" \
      --pretrained-seq-path "$SEQTOWER" --save-dir results/routeA_hlaholdout \
      > "${DIR_A}/seed_${s}.stdout" 2>&1 &
  log "  locusA seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

log "Phase 2: waiting for B->A metrics ..."
while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_A/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 300
done
log "B->A finetune done."

# ── Phase 3: aggregate ────────────────────────────────────────────────
log "Phase 3: aggregate cross-locus"
python aggregate_seeds.py --save-dir "$DIR_B" --seeds $SEEDS | tee -a "$MASTER_LOG"
python aggregate_seeds.py --save-dir "$DIR_A" --seeds $SEEDS | tee -a "$MASTER_LOG"

# ── Phase 4: full-data 4 seeds -> ITSNdb eval ─────────────────────────
log "Phase 4: full-data 4 seeds (holdout none) on GPUs 0 1 2 3"
i=0
for s in $SEEDS; do
  gpu=$i
  nohup python train_routeA_finetune.py --holdout-locus none --seeds "$s" --gpu "$gpu" \
      --pretrained-seq-path "$SEQTOWER" --save-dir results/routeA_full \
      > "${DIR_FULL}/seed_${s}.stdout" 2>&1 &
  log "  full seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

log "Phase 4: waiting for full-data metrics ..."
while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_FULL/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 300
done
log "Full-data finetune done."

log "Phase 4b: ITSNdb eval (v6 evaluator, routeA checkpoints) on GPU 0"
python eval_v6_itsndb.py --dataset both --seeds $SEEDS \
    --save-dir results/routeA_full --ckpt-name routeA_phase2_best.pt --gpu 0 2>&1 | tee -a "$MASTER_LOG"

log "==================== ROUTE A RUN DONE $(date '+%F %T') ===================="
