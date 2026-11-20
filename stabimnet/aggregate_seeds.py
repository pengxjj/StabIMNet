#!/usr/bin/env python3
"""Aggregate per-seed metrics from parallel single-seed runs into paper-style summary.

Reads seed_{seed}_metrics.json produced by train_v2_multiseed.py (single-seed mode)
and reports mean +/- s.d. for AUROC/AUPRC and both PPVn variants (continuous
BigMHC definition, and precision@0.5 which matches the paper's reported value).
"""
import os
import json
import argparse

import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-seed metrics")
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/five_seeds_aggregate")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    results, missing = [], []
    for seed in args.seeds:
        path = os.path.join(args.save_dir, f"seed_{seed}_metrics.json")
        if not os.path.exists(path):
            missing.append(seed)
            continue
        with open(path) as f:
            results.append(json.load(f))

    if missing:
        print(f"[WARN] missing (not finished?): {missing}")

    if not results:
        print("No seed metrics found.")
        return

    results.sort(key=lambda r: r["seed"])
    n = len(results)
    rocs = np.array([r["test_roc"] for r in results])
    prs = np.array([r["test_pr"] for r in results])
    ppvn_conts = np.array([r["test_ppvn_cont"] for r in results])
    ppvn_precs = np.array([r["test_ppvn_prec"] for r in results])

    def mean_sd(x):
        return float(x.mean()), (float(x.std(ddof=1)) if n > 1 else 0.0)

    mean_roc, sd_roc = mean_sd(rocs)
    mean_pr, sd_pr = mean_sd(prs)
    mean_ppvn_cont, sd_ppvn_cont = mean_sd(ppvn_conts)
    mean_ppvn_prec, sd_ppvn_prec = mean_sd(ppvn_precs)

    print(f"\n{'=' * 78}\n  MULTI-SEED SUMMARY (n={n})\n{'=' * 78}")
    print(f"{'Seed':>6} | {'Test ROC':>9} | {'Test PR':>8} | {'PPVn_cont':>9} | "
          f"{'PPVn_prec':>10} | {'Best Val ROC':>12} | {'Best Ep':>7}")
    print("-" * 78)
    for r in results:
        print(f"{r['seed']:>6} | {r['test_roc']:>9.4f} | {r['test_pr']:>8.4f} | "
              f"{r['test_ppvn_cont']:>9.4f} | {r['test_ppvn_prec']:>10.4f} | "
              f"{r['best_val_roc']:>12.4f} | {r['best_epoch']:>7}")
    print("-" * 78)
    print(f"{'mean':>6} | {mean_roc:>9.4f} | {mean_pr:>8.4f} | {mean_ppvn_cont:>9.4f} | {mean_ppvn_prec:>10.4f}")
    print(f"{'s.d.':>6} | {sd_roc:>9.4f} | {sd_pr:>8.4f} | {sd_ppvn_cont:>9.4f} | {sd_ppvn_prec:>10.4f}")
    print(f"\n  AUROC: {mean_roc:.4f} +/- {sd_roc:.4f} (sample s.d.)")
    print(f"  AUPRC: {mean_pr:.4f} +/- {sd_pr:.4f} (sample s.d.)")
    print(f"  PPVn continuous:   {mean_ppvn_cont:.4f} +/- {sd_ppvn_cont:.4f} (sample s.d.)")
    print(f"  PPVn precision@0.5:{mean_ppvn_prec:.4f} +/- {sd_ppvn_prec:.4f} (sample s.d.)")

    csv_path = os.path.join(args.save_dir, "multiseed_summary.csv")
    with open(csv_path, "w") as f:
        f.write("seed,test_roc,test_pr,test_ppvn_cont,test_ppvn_prec,best_val_roc,best_epoch,n_test\n")
        for r in results:
            f.write(f"{r['seed']},{r['test_roc']:.6f},{r['test_pr']:.6f},"
                    f"{r['test_ppvn_cont']:.6f},{r['test_ppvn_prec']:.6f},"
                    f"{r['best_val_roc']:.6f},{r['best_epoch']},{r['n_test']}\n")

    summary = {
        "seeds": [r["seed"] for r in results],
        "per_seed": results,
        "auroc_mean": mean_roc, "auroc_sd": sd_roc,
        "auprc_mean": mean_pr, "auprc_sd": sd_pr,
        "ppvn_cont_mean": mean_ppvn_cont, "ppvn_cont_sd": sd_ppvn_cont,
        "ppvn_prec_mean": mean_ppvn_prec, "ppvn_prec_sd": sd_ppvn_prec,
    }
    json_path = os.path.join(args.save_dir, "multiseed_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary CSV:  {csv_path}")
    print(f"  Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
