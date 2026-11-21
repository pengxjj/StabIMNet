#!/bin/bash
# Launch the full 8-seed HybridModelv5 cross-locus HLA-holdout in 2 GPU waves.
# Wave 1: B->A (holdout-locus A, test HLA-A) seeds 42 123 456 789 on GPU 0-3
# Wave 2: A->B (holdout-locus B, test HLA-B) seeds 42 123 456 789 on GPU 0-3
cd /public/home/lifex/pengxj/bio_models/ImmunoStruct-main/StructIM-hla
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm

SEEDS="42 123 456 789"
DIR_A=results/v5_hlaholdout_locusA
DIR_B=results/v5_hlaholdout_locusB

echo "[v5] wave 1 (locus A) start $(date)"
GPUS=(0 1 2 3)
i=0
for s in $SEEDS; do
  gpu=${GPUS[$i]}
  nohup python train_v5_hlaholdout.py --holdout-locus A --seeds "$s" --gpu "$gpu" \
      > "${DIR_A}/seed_${s}.stdout" 2>&1 &
  echo "  locusA seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

# wait for wave 1
echo "[v5] waiting for wave 1 ..."
while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_A/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 60
done
echo "[v5] wave 1 done $(date)"

echo "[v5] wave 2 (locus B) start $(date)"
i=0
for s in $SEEDS; do
  gpu=${GPUS[$i]}
  nohup python train_v5_hlaholdout.py --holdout-locus B --seeds "$s" --gpu "$gpu" \
      > "${DIR_B}/seed_${s}.stdout" 2>&1 &
  echo "  locusB seed=$s gpu=$gpu pid=$!"
  i=$((i+1))
done

echo "[v5] waiting for wave 2 ..."
while true; do
  missing=""
  for s in $SEEDS; do [ -f "$DIR_B/seed_${s}_metrics.json" ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  sleep 60
done
echo "[v5] wave 2 done $(date)"

echo ""
echo "################ B->A  (train HLA-B, test HLA-A) ################"
python aggregate_seeds.py --save-dir "$DIR_A" --seeds 42 123 456 789

echo ""
echo "################ A->B  (train HLA-A, test HLA-B) ################"
python aggregate_seeds.py --save-dir "$DIR_B" --seeds 42 123 456 789

echo ""
echo "[v5] ALL DONE $(date)"
