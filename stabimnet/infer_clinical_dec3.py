"""Clinical survival validation for the dec3 single-branch model (clinical_dec3).

Runs the trained ``HybridModelOpt_ComparativeSingle`` (dec3 backbone + single
branch immunogenicity head, see train_opt_phase3_clinical.py) over the 29,216
clinical pMHC graphs, aggregates into a per-patient "immunogenicity load", median
splits (35/35), and reports log-rank OS/PFS plus a Cox hazard ratio — the same
computation as the original paper Fig. 5c,d.

Unlike the original-model repro (which used the PDB-derived graph_pyg_clinical),
this uses the ESMFold2 clinical graphs (``graph_pyg_clinical_esmfold2``) so the
dec3 backbone (ESMFold2-trained) stays in-domain.

Reuses ``procedures/clinical_validation.py:clinical_pvalues`` semantics verbatim
(convert_patient_code mUC->BC / MM->Neye / RH->RH, groupby-sum, percentile-50
split, log-rank OS/PFS) plus a Cox PH for the HR/CI shown in the figure.

Run from inside StructIM-tumor/, e.g.:
    python -u infer_clinical_dec3.py --seeds 42 43 44 45 46 --device cuda:0
"""

import os
import argparse
import json
import types

import torch

# ---- environment shims (no change to inference logic) ----
_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
import sys
sys.modules["dgl.graphbolt"] = _dgl_gb

_wandb = types.ModuleType("wandb")
_wandb.init = lambda *a, **k: None
_wandb.log = lambda *a, **k: None
sys.modules["wandb"] = _wandb

_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from dgl.dataloading import GraphDataLoader

import dgl as _dgl
_dgl_orig_graph_to = _dgl.DGLGraph.to
def _dgl_graph_to_cpu(self, device=None, *args, **kwargs):
    return self
_dgl.DGLGraph.to = _dgl_graph_to_cpu

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from data_loading import ClinicalDataset, SplitDataset, collate
from models.optimized_models_clinical import HybridModelOpt_ComparativeSingle
from utils import seed_everything


def convert_patient_code(patient_code):
    mapping = {'mUC': 'BC', 'MM': 'Neye', 'RH': 'RH'}
    prefix, number = patient_code.split('-')
    if prefix in mapping:
        return f"{mapping[prefix]}-{number}"
    return patient_code


def survival_pvalues(predicted_probs_clinical, clin_seq_path, clin_survival_path, save_per_patient=None):
    """Identical computation to clinical_validation.py:clinical_pvalues, plus a Cox
    hazard ratio so the output matches the paper figure (HR + 95% CI). Also returns the
    per-patient immunogenicity load (indexed by patient_ID) and optionally saves the
    per-patient table for Kaplan-Meier plotting."""
    seq_df = pd.read_csv(clin_seq_path)
    survival_df = pd.read_csv(clin_survival_path)

    seq_df['patient_ID'] = seq_df['patient'].apply(convert_patient_code)
    seq_df['ImmunoStruct_predicted'] = predicted_probs_clinical
    seq_df = seq_df.dropna(subset=['ImmunoStruct_predicted'])

    immunostruct_load_df = seq_df.groupby('patient_ID')['ImmunoStruct_predicted'].sum().reset_index()
    immunostruct_load_df = immunostruct_load_df.sort_values('patient_ID')

    survival_df['patient_ID'] = survival_df['Patient']
    survival_df = survival_df.sort_values('patient_ID')
    survival_df['ImmunoStruct_predicted_load'] = immunostruct_load_df['ImmunoStruct_predicted'].tolist()

    med = np.percentile(survival_df['ImmunoStruct_predicted_load'], 50)
    low = survival_df[survival_df['ImmunoStruct_predicted_load'] <= med]
    high = survival_df[survival_df['ImmunoStruct_predicted_load'] >= med]

    load_series = survival_df.set_index('patient_ID')['ImmunoStruct_predicted_load']
    if save_per_patient is not None:
        out = survival_df[['Patient', 'RECIST', 'mut_load', 'OS.Time', 'OS.Event',
                           'PFS.Time', 'PFS.Event', 'ImmunoStruct_predicted_load']].copy()
        out['group'] = np.where(out['ImmunoStruct_predicted_load'] >= med, 'High', 'Low')
        os.makedirs(os.path.dirname(save_per_patient) or '.', exist_ok=True)
        out.to_csv(save_per_patient, index=False)

    res = {"n_total": int(len(survival_df)), "n_low": int(len(low)), "n_high": int(len(high)),
           "median_load": float(med)}

    os_lr = logrank_test(low['OS.Time'], high['OS.Time'],
                         event_observed_A=low['OS.Event'], event_observed_B=high['OS.Event'])
    pfs_lr = logrank_test(low['PFS.Time'], high['PFS.Time'],
                          event_observed_A=low['PFS.Event'], event_observed_B=high['PFS.Event'])
    res["os_p_logrank"] = float(os_lr.p_value)
    res["pfs_p_logrank"] = float(pfs_lr.p_value)

    cox_df = survival_df[['OS.Time', 'OS.Event', 'PFS.Time', 'PFS.Event',
                          'ImmunoStruct_predicted_load']].copy()
    cox_df['high'] = (cox_df['ImmunoStruct_predicted_load'] >= med).astype(int)
    for endpoint, tcol, ecol in [("os", "OS.Time", "OS.Event"), ("pfs", "PFS.Time", "PFS.Event")]:
        d = cox_df[[tcol, ecol, 'high']].rename(columns={tcol: 'T', ecol: 'E'})
        cph = CoxPHFitter().fit(d, duration_col='T', event_col='E')
        hr = float(np.exp(cph.params_['high']))
        ci = np.exp(cph.confidence_intervals_.loc['high'].values)
        res[f"{endpoint}_hr"] = hr
        res[f"{endpoint}_hr_ci_low"] = float(ci[0])
        res[f"{endpoint}_hr_ci_high"] = float(ci[1])
        res[f"{endpoint}_p_cox"] = float(cph.summary.loc['high', 'p'])
    return res, load_series


def run_clinical_inference(model, device, clinical_loader):
    model.eval()
    predicted_probs_clinical = []
    with torch.no_grad():
        for graph_data, sequence_data, target, peptide_property in clinical_loader:
            graph_data = graph_data.to(device)
            sequence_data, peptide_property = sequence_data.to(device), peptide_property.to(device)

            _, _, _, final_output = model(graph_data, sequence_data, peptide_property)

            probs = torch.sigmoid(final_output).squeeze()
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
            probs = probs.detach().cpu().numpy()
            peptide_property = peptide_property.detach().cpu().numpy()

            nan_indices = np.asarray(np.where(np.isnan(peptide_property[:, 0]))).reshape(-1)
            probs[nan_indices] = np.nan
            predicted_probs_clinical.extend(probs.tolist())
    return np.array(predicted_probs_clinical)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clinical_dec3 survival (5-seed).")
    parser.add_argument("--graph-dir-clinical", default=f"{_PROJ_ROOT}/data/graph_pyg_clinical_esmfold2/", type=str)
    parser.add_argument("--clinical-seq-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_clinical_data.csv", type=str)
    parser.add_argument("--clinical-survival-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_clinical_data_survival.csv", type=str)
    parser.add_argument("--ckpt-dir", default=f"{_PROJ_ROOT}/results/opt_phase3_clinical_751015/", type=str,
                        help="dir containing opt_phase3_clinical_seed{seed}_best.pt checkpoints")
    parser.add_argument("--out-dir", default=f"{_PROJ_ROOT}/results/opt_phase3_clinical_751015/", type=str)
    parser.add_argument("--vae-hidden-dim", default=768, type=int)
    parser.add_argument("--vae-latent-dim", default=64, type=int)
    parser.add_argument("--gat-hidden", default=64, type=int)
    parser.add_argument("--gnn-layers", default=5, type=int)
    parser.add_argument("--property-emb-dim", default=16, type=int)
    parser.add_argument("--egnn-type", default="opt", type=str, choices=["opt", "torch"])
    parser.add_argument("--decoder-layers", default=3, type=int)
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="cuda:0", type=str)
    config = parser.parse_args()

    device = torch.device(config.device)

    print(f"[clinical dec3] building clinical dataset from {config.graph_dir_clinical}", flush=True)
    clinical_dataset = ClinicalDataset(config,
                                       graph_directory=config.graph_dir_clinical,
                                       seq_path=config.clinical_seq_path)
    vae_input_dim = clinical_dataset.encoded_full_seq.shape[1] * clinical_dataset.encoded_full_seq.shape[2]
    print(f"[clinical dec3] vae_input_dim = {vae_input_dim}", flush=True)

    clinical_dataset = SplitDataset(clinical_dataset, "inference", binary=True,
                                    full=True, comparative=True,
                                    return_amino_acid=False)
    clinical_loader = GraphDataLoader(clinical_dataset, batch_size=config.batch_size,
                                      collate_fn=collate, shuffle=False,
                                      num_workers=config.num_workers)
    print(f"[clinical dec3] clinical samples = {len(clinical_dataset)}", flush=True)

    all_res = {}
    all_loads = {}
    for seed in config.seeds:
        ckpt = os.path.join(config.ckpt_dir, f"opt_phase3_clinical_seed{seed}_best.pt")
        seed_everything(seed)
        model = HybridModelOpt_ComparativeSingle(
            vae_input_dim, device,
            gcn_layers=config.gnn_layers,
            vae_hidden_dim=config.vae_hidden_dim,
            vae_latent_dim=config.vae_latent_dim,
            gat_hidden_channels=config.gat_hidden,
            property_embedding_dim=config.property_emb_dim,
            egnn_type=config.egnn_type,
            decoder_layers=config.decoder_layers).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[clinical dec3] seed {seed} loaded {ckpt}", flush=True)

        probs = run_clinical_inference(model, device, clinical_loader)
        res, load = survival_pvalues(probs, config.clinical_seq_path, config.clinical_survival_path)
        res["seed"] = seed
        all_res[str(seed)] = res
        all_loads[seed] = load
        print(f"REPRO_RESULTS_CLINICAL_DEC3 {json.dumps(res)}", flush=True)
        del model

    load_df = pd.DataFrame(all_loads).sort_index()
    load_df.index.name = 'patient_ID'
    load_csv = os.path.join(config.out_dir, "clinical_per_patient_stabnet.csv")
    load_df.to_csv(load_csv)
    print(f"[saved] {load_csv} ({len(load_df)} patients x {len(all_loads)} seeds)")

    os_p = [all_res[str(s)]["os_p_logrank"] for s in config.seeds]
    pfs_p = [all_res[str(s)]["pfs_p_logrank"] for s in config.seeds]
    os_hr = [all_res[str(s)]["os_hr"] for s in config.seeds]
    pfs_hr = [all_res[str(s)]["pfs_hr"] for s in config.seeds]
    agg = {
        "os_p_logrank": {f"seed{s}": round(all_res[str(s)]["os_p_logrank"], 5) for s in config.seeds},
        "os_p_logrank_mean": float(np.mean(os_p)),
        "pfs_p_logrank": {f"seed{s}": round(all_res[str(s)]["pfs_p_logrank"], 5) for s in config.seeds},
        "pfs_p_logrank_mean": float(np.mean(pfs_p)),
        "os_hr_mean": float(np.mean(os_hr)),
        "pfs_hr_mean": float(np.mean(pfs_hr)),
    }
    print(f"REPRO_RESULTS_CLINICAL_DEC3_SUMMARY {json.dumps(agg)}", flush=True)

    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "clinical_dec3_results.json"), "w") as f:
        json.dump({"per_seed": all_res, "summary": agg}, f, indent=2)
    print("DONE")
