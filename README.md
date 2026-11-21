# StabIMNet — Reproducibility Release Package

This directory contains the **code** needed to reproduce the three core result
groups of the StabIMNet paper (*"StabIMNet improves reproducibility and clinical
stratification in pMHC immunogenicity prediction"*), plus **one seed-42
three-stage weight set** and a **small smoke dataset** to prove the pipeline runs
end-to-end. The **full datasets and all remaining weights are NOT included** —
see [`DATA_AND_WEIGHTS.md`](../ImmunoStruct-main/DATA_AND_WEIGHTS.md) in the
sibling repo for the complete manifest and hosting plan.

The paper's core contributions are organized here into three experiment classes:

1. **Four-level generalization evaluation** → [`generalization/`](generalization/)
2. **StabIMNet model architecture + benchmark results** → [`stabimnet/`](stabimnet/)
3. **TCR-pMHC binding evaluation** → [`tcr/`](tcr/)

---

## 1. Quick start (smoke test)

```bash
bash run_smoke.sh
```

This runs the StabIMNet **stage-2 (IEDB) training** for 2 epochs on the
`data_smoke/` subset, warm-started from the provided stage-1 VAE weights. It
exercises the full code path — graph loading, the Phase-1 VAE transfer
(`input_dim = 283 × 21 = 5943`), the `HybridModelOpt` forward/backward, and the
paper's 14-metric evaluation — and should finish in a few minutes on CPU or GPU.

Requirements: the `esm` conda environment (torch 2.5.1+cu124, DGL 2.1.0 CPU-only,
PyG 2.8.0, esm 3.3.0). Override the interpreter with
`ESP=/path/to/python bash run_smoke.sh`.

**Expected output** — the smoke run is a *plumbing* test, not a results test. With
2 epochs on 400 samples you will see `IEDB n=400 input_dim=5943` (the VAE
dimension matching the stage-1 weights), a `[phase1] transferred 14 VAE tensors`
line, two training epochs, all 14 metrics printed, and a checkpoint under
`results_smoke/opt_phase2/`. The metrics will be near chance (ROC ≈ 0.5) and the
single-seed s.d. prints `nan` — both expected. Paper numbers require the full
data + 5 seeds via the sibling repo.

---

## 2. Directory layout

```
ImmunoStruct_release/
├── build_smoke_data.py          # rebuilds data_smoke/ from the full IEDB tree
├── run_smoke.sh                 # end-to-end stage-2 smoke run
├── weights/                     # ONE seed-42 three-stage weight set
│   ├── stage1_vae_presentation.pt   # Phase-1 VAE (presentation, vae_presentation_vae.pt)
│   ├── stage2_iedb_seed42.pt        # Phase-2 IEDB (opt_phase2_seed42_best.pt)
│   └── stage3_cedar_seed42.pt       # Phase-3 CEDAR (opt_phase3_seed42_best.pt)
├── data_smoke/                  # ~400-row stratified IEDB subset + 400 graphs
├── stabimnet/                   # 3-stage StabIMNet model + IEDB/CEDAR/clinical scripts
│   └── base/                    #   (tumor base: optimized_models, egnn_opt, comparative)
├── generalization/              # 4-level generalization scripts
│   └── base/                    #   (hla base: hybrid_models, CrossModalAttentionBlock)
└── tcr/                         # TCR-pMHC binding scripts
    └── base/                    #   (hla base, same copy as generalization/)
```

`base/` is the shared foundation library each script group builds on. There are
**two distinct** `base` packages, and they must NOT be merged:

- **tumor** base (`stabimnet/base/`) — `optimized_models.py`, `egnn_opt.py`,
  `comparative_models.py`, `utils/optimized_loss.py`, … used by the 3-stage
  StabIMNet scripts.
- **hla** base (`generalization/base/` and `tcr/base/`) — `hybrid_models.py`
  (v5/v6/v7/v8 cross-attention), `layers.py` (`CrossModalAttentionBlock`), … used by
  the generalization and TCR scripts. The two hla copies are identical real
  directories (no symlinks, so a plain ZIP download works).

The two bases share several filenames (`hybrid_models.py`, `layers.py`,
`mapping.py`, `preprocess.py`, `metric.py`, …) whose **contents differ**, which is
why they cannot be collapsed into one directory.

Each script group is self-contained: its `base/` subpackage is added to `sys.path`
by the scripts themselves (`_PROJ_ROOT/base`), so run them from their own directory
with no extra `PYTHONPATH`.

---

## 3. The StabIMNet three-stage model (`stabimnet/`)

| Stage | Script | Model | Data | Provided weight |
|---|---|---|---|---|
| 1 | `train_opt_phase1.py` | Presentation VAE (`HybridModelOpt.vae`) | IEDB | `weights/stage1_vae_presentation.pt` |
| 2 | `train_opt_phase2_paper.py` | `HybridModelOpt` (IEDB) | IEDB | `weights/stage2_iedb_seed42.pt` |
| 3 | `train_opt_phase3_paper.py` | `HybridModelOpt_Comparative` (CEDAR) | CEDAR | `weights/stage3_cedar_seed42.pt` |
| 3 (clinical) | `train_opt_phase3_clinical.py` | clinical fine-tune | 70-patient cohort | — |

Evaluation / aggregation:

- `eval_opt_phase2_test.py`, `eval_opt_phase2_multi.py` — held-out IEDB metrics.
- `eval_original_paper.py`, `eval_original_cedar.py`, `eval_dec3_paper3500.py` —
  Original (`HybridModelv2`) vs StabIMNet (`HybridModelOpt`) comparison on the
  fixed 3500-sample test set (Table 1 / Figure 2).
- `infer_clinical_dec3.py` — per-patient survival loads (Figure 5).
- `aggregate_seeds.py` — mean ± s.d. across seeds.

> The faithful "Original" (`HybridModelv2`) baseline reproduction lives in the
> sibling `ImmunoStruct-orig/ImmunoStruct-main/` repo, not here — this package
> carries the **StabIMNet** (`HybridModelOpt`) lineage.

---

## 4. Four-level generalization (`generalization/`)

The paper evaluates generalization at four escalating levels; the scripts map as:

| Level | Description | Scripts |
|---|---|---|
| L1 | Random IEDB split | `stabimnet/train_opt_phase2_paper.py` |
| L2 | Cross-locus HLA holdout (A→B / B→A) | `train_v{2,5,6,7,8}_hlaholdout.py` (Table S5) |
| L3 | External ITSNdb | `eval_v{6,7}_itsndb.py`, `analyze_itsndb_allele_ood.py` |
| L4 | TESLA / NEPdb T-cell-validated | `eval_v{2,6,7}_tesla.py`, `validate_nepdb_denovo.py`, `build_allele_balanced_benchmark.py`, `analyze_allele_seen_ood.py` |

Model variants (`models/hybrid_models.py`): `v2`=Original, `v5`=FiLM, `v6`=XAttn,
`v7`=FullSeq, `v8`=StructAttn, `RouteA`=EL-pretrained.

De-novo feature pipeline (sequence-only, no structure):

- `compute_denovo_features.py` → `extract_tesla_features.py` → `eval_features_tesla.py`
- `extract_hla_pseudosequence*.py`, `precompute_esmc_*.py` — ESMC per-residue
  embedding precomputation (LARGE outputs, not bundled — see the data manifest).

---

## 5. TCR-pMHC binding (`tcr/`)

| Script | Role |
|---|---|
| `preprocess_vdjdb_tcr.py` | Build paired TCR-pMHC splits from VDJdb |
| `tcr_model.py` / `tcr_model_fullseq.py` / `tcr_model_esm*.py` / `tcr_model_contrastive.py` | TCR binding encoders |
| `train_tcr_pmhc.py` / `train_tcr_pmhc_sup.py` / `train_tcr_pmhc_contrastive.py` / `train_tcr_pmhc_esm*.py` | Training variants |
| `eval_tcr_pmhc.py`, `predict_tcr_pmhc*.py`, `inventory_tcr_pmhc.py` | Evaluation / prediction |
| `run_tcr_multiseed.sh` | 5-seed launcher |

The swap-decoy evaluation (honest negative control) is the paper's headline
TCR result; see `tcr/tcr_pmhc_results.md` in the sibling repo for the numbers.

---

## 6. Rebuilding the smoke data

```bash
python build_smoke_data.py
```

Reads the full IEDB table + graph dir (paths at the top of the script) and writes
a ~400-row stratified subset that preserves the ~22% positive rate, the 9/10/11
peptide-length mix (**including ≥1 11-mer so the VAE `input_dim` stays 5943** and
the stage-1 weights load), and ≥5 HLA alleles.

---

## 7. Full reproduction

The complete paper reproduction (all 5 seeds, all tables/figures, all sections)
is orchestrated by [`reproduce_all.sh`](../ImmunoStruct-main/reproduce_all.sh) and
documented in [`REPRODUCIBILITY.md`](../ImmunoStruct-main/REPRODUCIBILITY.md) in
the sibling `ImmunoStruct-main/` repo. That path requires the full datasets and
weights enumerated in `DATA_AND_WEIGHTS.md`.
