#!/usr/bin/env python3
"""
Phase 2 HLA-HOLDOUT (cross-locus) generalization probe.

Same training loop as train_v2_multiseed_meta2.py but the test set is an ENTIRE
HLA locus held out of training, so the model is evaluated on alleles it never
saw during finetuning. This measures new-HLA generalization (the key question
for personalised-neoantigen use).

Usage:
    # Train on HLA-A only, test on all of HLA-B:
    python train_v2_hlaholdout.py --holdout-locus B --seeds 42 123 456 789

    # Train on HLA-B only, test on all of HLA-A:
    python train_v2_hlaholdout.py --holdout-locus A --seeds 42 123 456 789

    # (--holdout-locus none reproduces the original random stratified holdout)
    python train_v2_hlaholdout.py --holdout-locus none --seeds 42 123 456 789

The allele is recovered from dataset.key_list (key = "{allele}_{peptide}").
"""

import os
import sys
import argparse
import types
import logging
import json
import gc

import numpy as np

# DGL compatibility patch
_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
sys.modules["dgl.graphbolt"] = _dgl_gb

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dgl.dataloading import GraphDataLoader
from models.hybrid_models import HybridModelv2
from data_loading import ImmunoPredDataset, collate, SplitDataset
from utils import seed_everything, update_paths
from procedures.metric import mean_PPVn

PHASE1_MAX_LEN = 290
PHASE1_N_AA = 21
PHASE1_INPUT_DIM = PHASE1_MAX_LEN * PHASE1_N_AA  # 6090
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def align_vae_weights(ckpt, ckpt_input_dim, target_input_dim):
    """Trim Phase 1 VAE weights to match IEDB input dimension."""
    if ckpt_input_dim == target_input_dim:
        return ckpt
    ckpt["vae_fc1.weight"] = ckpt["vae_fc1.weight"][:, :target_input_dim]
    ckpt["vae_fc4.weight"] = ckpt["vae_fc4.weight"][:target_input_dim, :]
    ckpt["vae_fc4.bias"] = ckpt["vae_fc4.bias"][:target_input_dim]
    return ckpt


def stratified_holdout(dataset, n_test, seed):
    """Fixed stratified holdout of n_test samples preserving the class ratio.

    Drawn ONCE with a dedicated RNG (independent of training seeds) so every
    seed evaluates on the identical test set; only train/val vary per seed.
    """
    labels = dataset.immunogenicity.numpy()
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    pos_rate = len(pos_idx) / len(labels)
    n_test_pos = int(round(n_test * pos_rate))
    n_test_neg = n_test - n_test_pos
    rng = np.random.default_rng(seed)
    test_pos = rng.choice(pos_idx, size=n_test_pos, replace=False)
    test_neg = rng.choice(neg_idx, size=n_test_neg, replace=False)
    test_idx = np.sort(np.concatenate([test_pos, test_neg]))
    return test_idx.astype(np.int64)


def locus_holdout(dataset, locus):
    """Hold out every sample whose HLA allele belongs to `locus` ('A' or 'B').

    The allele is recovered from ``dataset.key_list`` where each key is
    ``"{allele}_{peptide}"`` (e.g. ``"HLA-A*11:01_LSNSGKDVPK"``).
    Returns indices of all samples in that locus.
    """
    assert locus in ("A", "B"), f"locus must be 'A' or 'B', got {locus!r}"
    prefix = f"HLA-{locus}"
    test_idx = [i for i, k in enumerate(dataset.key_list) if k.startswith(prefix)]
    return np.asarray(test_idx, dtype=np.int64)


def get_separate_param_groups(model, lr_vae, lr_other, weight_decay):
    vae_keys = {"vae_fc1", "vae_fc21", "vae_fc22", "vae_fc3", "vae_fc4",
                "property_embedding"}
    vae_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(k) for k in vae_keys):
            vae_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": vae_params, "lr": lr_vae, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr_other, "weight_decay": weight_decay},
    ]


def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss, n, nan_skipped = 0.0, 0, 0

    for batch in loader:
        graph_data, sequence_data, target, peptide_property = batch[:4]
        sequence_data = sequence_data.to(device)
        target = target.to(device)
        peptide_property = peptide_property.to(device)

        optimizer.zero_grad()
        recon, mu, logvar, output = model(graph_data, sequence_data, peptide_property)
        output = output.squeeze(-1)

        bce = F.binary_cross_entropy_with_logits(output, target,
                                                 pos_weight=pos_weight.to(device),
                                                 reduction="mean")
        mse = F.mse_loss(recon, sequence_data.view(sequence_data.shape[0], -1),
                         reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = 5.0 * bce + 0.1 * mse + 0.1 * kld

        if torch.isnan(loss) or torch.isinf(loss):
            nan_skipped += 1
            if nan_skipped <= 6:
                def _dbg(name, t):
                    tt = t.detach()
                    return (f"{name}: nan={torch.isnan(tt).any().item()} "
                            f"inf={torch.isinf(tt).any().item()} "
                            f"max={tt.abs().max().item():.4g}")
                print(f"  [NaN#{nan_skipped}] loss={loss.item():.4g} "
                      f"bce={bce.item():.4g} mse={mse.item():.4g} kld={kld.item():.4g}")
                print(f"    {_dbg('output', output)}")
                print(f"    {_dbg('recon', recon)}")
                print(f"    {_dbg('mu', mu)}")
                print(f"    {_dbg('logvar', logvar)}")
                print(f"    {_dbg('graph_x', graph_data.ndata['x'])}")
                w_nan = any(torch.isnan(p).any().item() for p in model.parameters())
                w_inf = any(torch.isinf(p).any().item() for p in model.parameters())
                print(f"    weight_nan={w_nan} weight_inf={w_inf}")
                cb = getattr(model, '_combined', None)
                if cb is not None:
                    print(f"    {_dbg('x_gat_node', cb[:, :64])}")
                    print(f"    {_dbg('z_vae     ', cb[:, 64:])}")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = len(target)
        total_loss += loss.item() * bs
        n += bs

    if nan_skipped > 0:
        print(f"  [WARNING] train_epoch: skipped {nan_skipped} batch(es) with NaN/Inf loss")
    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def validate_epoch(model, loader, device, pos_weight):
    model.eval()
    total_loss, n = 0.0, 0
    nan_skipped_val = 0
    all_preds, all_labels = [], []

    for batch in loader:
        graph_data, sequence_data, target, peptide_property = batch[:4]
        sequence_data = sequence_data.to(device)
        target = target.to(device)
        peptide_property = peptide_property.to(device)

        recon, mu, logvar, output = model(graph_data, sequence_data, peptide_property)
        output = output.squeeze(-1)

        bce = F.binary_cross_entropy_with_logits(output, target,
                                                 pos_weight=pos_weight.to(device),
                                                 reduction="mean")
        mse = F.mse_loss(recon, sequence_data.view(sequence_data.shape[0], -1),
                         reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = 5.0 * bce + 0.1 * mse + 0.1 * kld

        bs = len(target)
        if torch.isnan(loss) or torch.isinf(loss):
            nan_skipped_val += 1
            if nan_skipped_val <= 6:
                def _dbgv(name, t):
                    tt = t.detach()
                    return (f"{name}: nan={torch.isnan(tt).any().item()} "
                            f"inf={torch.isinf(tt).any().item()} "
                            f"max={tt.abs().max().item():.4g}")
                print(f"  [VAL-NaN#{nan_skipped_val}] loss={loss.item():.4g} "
                      f"bce={bce.item():.4g} mse={mse.item():.4g} kld={kld.item():.4g}")
                print(f"    {_dbgv('output', output)}")
                print(f"    {_dbgv('recon', recon)}")
                print(f"    {_dbgv('mu', mu)}")
                print(f"    {_dbgv('logvar', logvar)}")
                print(f"    {_dbgv('graph_x', graph_data.ndata['x'])}")
                w_nan = any(torch.isnan(p).any().item() for p in model.parameters())
                w_inf = any(torch.isinf(p).any().item() for p in model.parameters())
                print(f"    weight_nan={w_nan} weight_inf={w_inf}")
                cb = getattr(model, '_combined', None)
                if cb is not None:
                    print(f"    {_dbgv('x_gat_node', cb[:, :64])}")
                    print(f"    {_dbgv('z_vae     ', cb[:, 64:])}")
            continue

        total_loss += loss.item() * bs
        n += bs
        probs = torch.sigmoid(output)
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(target.cpu().numpy())

    if n == 0:
        return 0.0, 0.5, 0.0, 0.0, 0.0

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)

    roc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    pr = average_precision_score(labels, preds)
    ppvn_cont = mean_PPVn(labels, preds)                                   # BigMHC continuous def
    ppvn_prec = precision_score(labels, preds >= 0.5, zero_division=0)     # paper ~0.514
    return total_loss / n, roc, pr, ppvn_cont, ppvn_prec


@torch.no_grad()
def get_predictions(model, loader, device):
    """Return (preds, labels) for final test scoring (NaN-safe)."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        graph_data, sequence_data, target, peptide_property = batch[:4]
        sequence_data = sequence_data.to(device)
        target = target.to(device)
        peptide_property = peptide_property.to(device)
        recon, mu, logvar, output = model(graph_data, sequence_data, peptide_property)
        output = output.squeeze(-1)
        probs = torch.sigmoid(output)
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(target.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
    return preds, labels


def run_seed(seed, config, dataset, phase1_ckpt, input_dim, device, logger, test_idx):
    """Train once under `seed`, return a dict of test metrics.

    `test_idx` is the FIXED stratified holdout shared by all seeds; only the
    train/val split (drawn from the remaining data) varies with `seed`.
    """
    seed_everything(seed)
    logger.info(f"\n{'=' * 60}\n  SEED {seed}\n{'=' * 60}")

    # Split: test is fixed; train/val drawn from the non-test pool under `seed`.
    # n_train is a fraction of the POOL (not the whole dataset) so an entire
    # held-out locus (22% of the data) still leaves a valid train/val split.
    n_total = len(dataset)
    test_mask = np.ones(n_total, dtype=bool)
    test_mask[test_idx] = False
    pool_idx = np.where(test_mask)[0]
    n_train = int(0.8 * len(pool_idx))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pool_idx), generator=g).tolist()
    train_idx = pool_idx[perm[:n_train]].tolist()
    val_idx = pool_idx[perm[n_train:]].tolist()
    n_val = len(val_idx)
    n_test = len(test_idx)

    # Model
    model = HybridModelv2(vae_input_dim=input_dim, device=device,
                          gcn_layers=config.gnn_layers,
                          vae_hidden_dim=config.vae_hidden_dim,
                          vae_latent_dim=config.vae_latent_dim,
                          gat_hidden_channels=config.gnn_dim,
                          property_embedding_dim=config.property_emb_dim)
    ckpt = {k: v for k, v in phase1_ckpt.items()}
    ckpt_input_dim = ckpt["vae_fc1.weight"].shape[1]
    ckpt = align_vae_weights(ckpt, ckpt_input_dim, input_dim)
    model.load_state_dict(ckpt, strict=False)
    model.to(device)

    # Diagnostic: capture classifier input [x_gat_node(64), z_vae(80)] to localise NaN
    model._combined = None

    def _capture_combined(module, args, output):
        model._combined = args[0].detach()

    model.classifier[1].register_forward_hook(_capture_combined)

    # Loaders
    train_loader = GraphDataLoader(
        SplitDataset(Subset(dataset, train_idx), "train", binary=True,
                     full=config.full_sequence, comparative=False, return_amino_acid=False),
        batch_size=config.batch_size, collate_fn=collate,
        shuffle=True, num_workers=config.num_workers, drop_last=True)
    val_loader = GraphDataLoader(
        SplitDataset(Subset(dataset, val_idx), "val", binary=True,
                     full=config.full_sequence, comparative=False, return_amino_acid=False),
        batch_size=config.batch_size, collate_fn=collate,
        shuffle=False, num_workers=config.num_workers)
    test_loader = GraphDataLoader(
        SplitDataset(Subset(dataset, test_idx), "test", binary=True,
                     full=config.full_sequence, comparative=False, return_amino_acid=False),
        batch_size=config.batch_size, collate_fn=collate,
        shuffle=False, num_workers=config.num_workers)

    # Class weight
    pos_count = sum(1 for i in train_idx if dataset.immunogenicity[i].item() > 0.5)
    neg_count = n_train - pos_count
    pos_weight = torch.tensor(neg_count / max(pos_count, 1))
    logger.info(f"Seed {seed}: Train {n_train} | Val {n_val} | Test {n_test} | "
                f"pos_weight {pos_weight:.4f} (pos={pos_count}, neg={neg_count})")

    # Optimizer + scheduler (CosineAnnealingLR stepped per epoch, T_max=epochs)
    param_groups = get_separate_param_groups(model, config.lr_vae, config.lr, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    seed_dir = os.path.join(config.save_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    best_path = os.path.join(seed_dir, "v2_phase2_best.pt")

    best_val_roc, best_epoch, no_improve = 0.0, 0, 0
    early_stop = config.early_stop if config.early_stop > 0 else config.epochs + 1

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        scheduler.step()
        val_loss, val_roc, val_pr, val_ppvn_cont, val_ppvn_prec = validate_epoch(model, val_loader, device, pos_weight)

        clamp_hits = sum(getattr(layer, "clamp_hits", 0) for layer in model.GCN_layers)
        max_abs_cw = max((getattr(layer, "max_abs_cw", 0.0) for layer in model.GCN_layers), default=0.0)
        if clamp_hits > 0 or epoch % 5 == 0 or epoch == 1:
            logger.info(f"[seed {seed}] Ep {epoch}: coord_clamp hits={clamp_hits} | max|coord_weight|={max_abs_cw:.4g}")

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[seed {seed}] Ep {epoch:3d}/{config.epochs} | Train {train_loss:.4f} | "
                        f"Val ROC {val_roc:.4f} | PR {val_pr:.4f} | "
                        f"PPVn_cont {val_ppvn_cont:.4f} | PPVn_prec {val_ppvn_prec:.4f}")

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= early_stop:
                logger.info(f"[seed {seed}] Early stop @ ep {epoch}, best Val ROC {best_val_roc:.4f} @ ep {best_epoch}")
                break

    # Final test eval on best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    preds, labels = get_predictions(model, test_loader, device)
    test_roc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    test_pr = average_precision_score(labels, preds)
    test_ppvn_cont = mean_PPVn(labels, preds)                               # BigMHC continuous def
    test_ppvn_prec = precision_score(labels, preds >= 0.5, zero_division=0)  # paper ~0.514

    logger.info(f"[seed {seed}] TEST ROC {test_roc:.4f} | PR {test_pr:.4f} | "
                f"PPVn_cont {test_ppvn_cont:.4f} | PPVn_prec {test_ppvn_prec:.4f} "
                f"(best Val ROC {best_val_roc:.4f} @ ep {best_epoch})")

    # Free GPU memory before next seed
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "test_roc": float(test_roc),
        "test_pr": float(test_pr),
        "test_ppvn_cont": float(test_ppvn_cont),
        "test_ppvn_prec": float(test_ppvn_prec),
        "best_val_roc": float(best_val_roc),
        "best_epoch": int(best_epoch),
        "n_test": int(len(test_idx)),
        "preds": preds,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 multi-seed finetuning (paper error-bar replication)")
    parser.add_argument("--graph-dir", default=f"{_PROJ_ROOT}/data/graph_pyg_IEDB_esmfold2/")
    parser.add_argument("--property-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_IEDB_data.csv")
    parser.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    parser.add_argument("--phase1-ckpt",
                        default=os.path.join(_PROJ_ROOT, "..", "results",
                                             "vae_presentation_pretrain_meta2",
                                             "vae_presentation_vae.pt"), type=str)
    parser.add_argument("--vae-hidden-dim", default=768, type=int)
    parser.add_argument("--vae-latent-dim", default=64, type=int)
    parser.add_argument("--gnn-dim", default=64, type=int)
    parser.add_argument("--gnn-layers", default=5, type=int)
    parser.add_argument("--property-emb-dim", default=16, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-vae", default=3e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float)
    parser.add_argument("--early-stop", default=30, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--full-sequence", action="store_true", default=True)
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--sequence-pad-count", default=0, type=int)
    parser.add_argument("--structure-pad-count", default=0, type=int)
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/v2_hlaholdout", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--holdout-locus", default="none", choices=["A", "B", "none"],
                        help="hold out an entire HLA locus as the test set "
                             "(A: train on B, test on A; B: train on A, test on B; "
                             "none: original random stratified holdout)")
    parser.add_argument("--test-size", default=3500, type=int,
                        help="fixed stratified holdout size (only used when --holdout-locus none)")
    parser.add_argument("--test-seed", default=42, type=int,
                        help="RNG seed for the fixed test-set holdout")
    parser.add_argument("--gpu", default=-1, type=int,
                        help="GPU index to use (>=0), or -1 for auto cuda")

    config = parser.parse_args()
    if config.holdout_locus in ("A", "B"):
        config.save_dir = f"{config.save_dir}_locus{config.holdout_locus}"
    update_paths(config)

    if config.gpu >= 0:
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.save_dir, exist_ok=True)
    if len(config.seeds) == 1:
        log_path = os.path.join(config.save_dir, f"seed_{config.seeds[0]}.log")
    else:
        log_path = os.path.join(config.save_dir, "multiseed.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    logger.info(f"Device: {device}")
    logger.info(f"Phase 1 ckpt: {config.phase1_ckpt}")
    logger.info(f"Save dir: {config.save_dir}")
    logger.info(f"Seeds: {config.seeds}")
    logger.info(f"Batch size: {config.batch_size} | Epochs: {config.epochs} | Early stop: {config.early_stop}")
    logger.info(f"LR {config.lr} (VAE {config.lr_vae})")

    # Load dataset ONCE
    logger.info("Loading IEDB dataset (once, shared across seeds)...")
    dataset = ImmunoPredDataset(config, graph_directory=config.graph_dir,
                                property_path=config.property_path,
                                hla_path=config.hla_path)
    input_dim = dataset.encoded_full_sequence.shape[1] * dataset.encoded_full_sequence.shape[2]
    logger.info(f"IEDB: {len(dataset):,} samples | Input dim {input_dim} (Phase 1 {PHASE1_INPUT_DIM})")

    # Load Phase 1 weights once (shallow-copied per seed before alignment)
    phase1_ckpt = torch.load(config.phase1_ckpt, map_location="cpu")
    logger.info(f"Phase 1 weights: {len(phase1_ckpt)} keys")

    # Test-set construction: either an entire held-out HLA locus (generalization
    # probe) or the original fixed stratified holdout (baseline).
    if config.holdout_locus in ("A", "B"):
        test_idx = locus_holdout(dataset, config.holdout_locus)
        n_test_pos = int(sum(dataset.immunogenicity[i].item() > 0.5 for i in test_idx))
        n_test = int(len(test_idx))
        n_train_pool = int(len(dataset) - n_test)
        logger.info(f"HLA-holdout: test = all HLA-{config.holdout_locus} "
                    f"({n_test} samples, pos={n_test_pos}, neg={n_test - n_test_pos}, "
                    f"pos_rate={n_test_pos / n_test:.4f}) | train/val pool = {n_train_pool}")
    else:
        test_idx = stratified_holdout(dataset, config.test_size, config.test_seed)
        n_test_pos = int(sum(dataset.immunogenicity[i].item() > 0.5 for i in test_idx))
        logger.info(f"Fixed test set: {len(test_idx)} samples "
                    f"(pos={n_test_pos}, neg={len(test_idx) - n_test_pos}) | test-seed {config.test_seed}")

    # Train under each seed
    results = []
    for seed in config.seeds:
        r = run_seed(seed, config, dataset, phase1_ckpt, input_dim, device, logger, test_idx)
        preds = r.pop("preds")
        labels = r.pop("labels")
        results.append(r)
        # Per-seed artifacts (parallel-safe across separate GPU processes)
        np.savez(os.path.join(config.save_dir, f"seed_{seed}_predictions.npz"),
                 preds=preds, labels=labels)
        with open(os.path.join(config.save_dir, f"seed_{seed}_metrics.json"), "w") as f:
            json.dump(r, f, indent=2)

    # Aggregate summary (only when multiple seeds run in this process;
    # single-seed parallel runs are combined by aggregate_seeds.py)
    n_seeds = len(results)
    if n_seeds < 2:
        logger.info("Single-seed run; combine results with aggregate_seeds.py")
        logger.info("DONE")
        return

    rocs = np.array([r["test_roc"] for r in results])
    prs = np.array([r["test_pr"] for r in results])
    ppvn_conts = np.array([r["test_ppvn_cont"] for r in results])
    ppvn_precs = np.array([r["test_ppvn_prec"] for r in results])

    def mean_sd(x):
        return float(x.mean()), float(x.std(ddof=1))

    mean_roc, sd_roc = mean_sd(rocs)
    mean_pr, sd_pr = mean_sd(prs)
    mean_ppvn_cont, sd_ppvn_cont = mean_sd(ppvn_conts)
    mean_ppvn_prec, sd_ppvn_prec = mean_sd(ppvn_precs)

    # Summary CSV
    csv_path = os.path.join(config.save_dir, "multiseed_summary.csv")
    with open(csv_path, "w") as f:
        f.write("seed,test_roc,test_pr,test_ppvn_cont,test_ppvn_prec,best_val_roc,best_epoch,n_test\n")
        for r in results:
            f.write(f"{r['seed']},{r['test_roc']:.6f},{r['test_pr']:.6f},"
                    f"{r['test_ppvn_cont']:.6f},{r['test_ppvn_prec']:.6f},"
                    f"{r['best_val_roc']:.6f},{r['best_epoch']},{r['n_test']}\n")

    summary = {
        "seeds": config.seeds,
        "per_seed": results,
        "auroc_mean": mean_roc, "auroc_sd": sd_roc,
        "auprc_mean": mean_pr, "auprc_sd": sd_pr,
        "ppvn_cont_mean": mean_ppvn_cont, "ppvn_cont_sd": sd_ppvn_cont,
        "ppvn_prec_mean": mean_ppvn_prec, "ppvn_prec_sd": sd_ppvn_prec,
    }
    with open(os.path.join(config.save_dir, "multiseed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 64}\n  MULTI-SEED SUMMARY (n={n_seeds})\n{'=' * 64}")
    logger.info(f"{'Seed':>6s} | {'Test ROC':>9s} | {'Test PR':>8s} | {'PPVn_cont':>9s} | "
                f"{'PPVn_prec':>10s} | {'Best Val ROC':>12s} | {'Best Ep':>7s}")
    logger.info("-" * 64)
    for r in results:
        logger.info(f"{r['seed']:>6d} | {r['test_roc']:>9.4f} | {r['test_pr']:>8.4f} | "
                    f"{r['test_ppvn_cont']:>9.4f} | {r['test_ppvn_prec']:>10.4f} | "
                    f"{r['best_val_roc']:>12.4f} | {r['best_epoch']:>7d}")
    logger.info("-" * 64)
    logger.info(f"{'mean':>6s} | {mean_roc:>9.4f} | {mean_pr:>8.4f} | {mean_ppvn_cont:>9.4f} | {mean_ppvn_prec:>10.4f}")
    logger.info(f"{'s.d.':>6s} | {sd_roc:>9.4f} | {sd_pr:>8.4f} | {sd_ppvn_cont:>9.4f} | {sd_ppvn_prec:>10.4f}")
    logger.info(f"\n  AUROC: {mean_roc:.4f} +/- {sd_roc:.4f} (sample s.d.)")
    logger.info(f"  AUPRC: {mean_pr:.4f} +/- {sd_pr:.4f} (sample s.d.)")
    logger.info(f"  PPVn continuous:   {mean_ppvn_cont:.4f} +/- {sd_ppvn_cont:.4f} (sample s.d.)")
    logger.info(f"  PPVn precision@0.5:{mean_ppvn_prec:.4f} +/- {sd_ppvn_prec:.4f} (sample s.d.)")
    logger.info(f"  Summary CSV: {csv_path}")
    logger.info(f"  Summary JSON: {os.path.join(config.save_dir, 'multiseed_summary.json')}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
