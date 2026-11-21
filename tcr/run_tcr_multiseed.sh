#!/usr/bin/env bash
# 5-seed (42-46) swap-negative TCR-pMHC training, parallelized across 2 free GPUs.
set -u
cd /public/home/lifex/pengxj/bio_models/ImmunoStruct-main/StructIM-hla
source /public/home/lifex/pengxj/miniconda3/etc/profile.d/conda.sh
conda activate esm

SEEDS=(42 43 44 45 46)
GPUS=(cuda:1 cuda:3)

mkdir -p logs_tcr

for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}
  gpu=${GPUS[$((i % 2))]}
  out="data/tcr_pmhc_s${s}"
  (
    python -u preprocess_vdjdb_tcr.py --vdjdb-dir /tmp/vdjdb_cache/vdjdb-db-master \
        --out "$out" --negative swap --seed "$s" \
      && python -u train_tcr_pmhc.py --data "$out" --epochs 15 --device "$gpu" --seed "$s" \
      && python -u eval_tcr_pmhc.py --data "$out" --device "$gpu"
  ) > "logs_tcr/seed${s}.log" 2>&1 &
done

wait
echo "ALL SEEDS DONE"
