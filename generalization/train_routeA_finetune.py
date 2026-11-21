#!/usr/bin/env python3
"""
Route A — stage 2: finetune HybridModelv6 on IEDB immunogenicity, initialized
from the stage-1 EL presentation pretrained sequence tower.

Identical to train_v6_hlaholdout.py except:
  * the sequence tower (pep_proj + hla_proj + pep_hla_attn) is initialized from
    a pretrained checkpoint (--pretrained-seq-path, exported by
    train_routeA_pretrain.py); the EGNN structure tower + classifier stay random,
  * save-dir / checkpoint name are routeA-scoped.

Cross-locus HLA-holdout (A->B, B->A) is the generalization readout, compared
against v6's baseline (A->B 0.688, B->A 0.714).

Usage:
    python train_routeA_finetune.py --holdout-locus B --seeds 42 --gpu 0 \
        --pretrained-seq-path results/routeA_pretrain/el_pretrain_seed42_seqtower.pt
"""

import os
import sys
import argparse
import types
import logging
import json
import gc
import copy

import numpy as np

# DGL compatibility patch
_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
sys.modules["dgl.graphbolt"] = _dgl_gb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

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

import dgl
from dgl.dataloading import GraphDataLoader
from models.hybrid_models import HybridModelv6
from data_loading import ImmunoPredDataset
from data_loading.data_utils import RandomRotation
from utils import seed_everything, update_paths
from procedures.metric import mean_PPVn

DEFAULT_SEEDS = [42, 123, 456, 789]
SEQ_EMB_DIM = 2560


class V6Dataset(Dataset):
    def __init__(self, base, pep_perres, hla_perres, train=False):
        self.base = base
        self.pep_perres = pep_perres
        self.hla_perres = hla_perres
        self.train = train
        self.rot = RandomRotation()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        key = self.base.key_list[idx]
        allele, pep = key.split('_', 1)
        graph = self.base.graphs[idx]
        if self.train:
            graph = copy.deepcopy(graph)
            coords = graph.ndata['x'][:, -3:]
            graph.ndata['x'][:, -3:] = torch.tensor(
                self.rot(coords.numpy()), dtype=torch.float32)
        return (graph,
                self.pep_perres[pep],
                self.hla_perres[allele],
                self.base.biochem_property[idx],
                self.base.immunogenicity[idx])


def v6_collate(samples):
    graphs, peps, hlas, props, labels = zip(*samples)
    batched_graph = dgl.batch(list(graphs))

    B = len(samples)
    Lp_max = max(p.shape[0] for p in peps)
    pep = torch.zeros(B, Lp_max, SEQ_EMB_DIM)
    pep_mask = torch.zeros(B, Lp_max, dtype=torch.bool)
    for i, p in enumerate(peps):
        L = p.shape[0]
        pep[i, :L] = p
        pep_mask[i, :L] = True

    hla = torch.stack(list(hlas), dim=0)
    hla_mask = torch.ones(B, hla.shape[1], dtype=torch.bool)

    prop = torch.stack(list(props), dim=0)
    label = torch.stack(list(labels), dim=0)
    return batched_graph, pep, pep_mask, hla, hla_mask, prop, label


def locus_holdout(dataset, locus):
    assert locus in ("A", "B"), f"locus must be 'A' or 'B', got {locus!r}"
    prefix = f"HLA-{locus}"
    test_idx = [i for i, k in enumerate(dataset.key_list) if k.startswith(prefix)]
    return np.asarray(test_idx, dtype=np.int64)


def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss, n, nan_skipped = 0.0, 0, 0
    for batch in loader:
        graph_data, pep, pep_mask, hla, hla_mask, peptide_property, target = batch
        pep = pep.to(device); pep_mask = pep_mask.to(device)
        hla = hla.to(device); hla_mask = hla_mask.to(device)
        peptide_property = peptide_property.to(device); target = target.to(device)
        optimizer.zero_grad()
        output = model(graph_data, pep, hla, pep_mask, hla_mask, peptide_property).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            output, target, pos_weight=pos_weight.to(device), reduction="mean")
        if torch.isnan(loss) or torch.isinf(loss):
            nan_skipped += 1
            optimizer.zero_grad()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = len(target)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        graph_data, pep, pep_mask, hla, hla_mask, peptide_property, target = batch
        pep = pep.to(device); pep_mask = pep_mask.to(device)
        hla = hla.to(device); hla_mask = hla_mask.to(device)
        peptide_property = peptide_property.to(device); target = target.to(device)
        output = model(graph_data, pep, hla, pep_mask, hla_mask, peptide_property).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            output, target, pos_weight=pos_weight.to(device), reduction="mean")
        bs = len(target)
        if not (torch.isnan(loss) or torch.isinf(loss)):
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
    ppvn_cont = mean_PPVn(labels, preds)
    ppvn_prec = precision_score(labels, preds >= 0.5, zero_division=0)
    return total_loss / n, roc, pr, ppvn_cont, ppvn_prec


def run_seed(seed, config, dataset, pep_perres, hla_perres, device, logger, test_idx):
    seed_everything(seed)
    logger.info(f"\n{'=' * 60}\n  SEED {seed}\n{'=' * 60}")

    n_total = len(dataset)
    test_mask = np.ones(n_total, dtype=bool)
    test_mask[test_idx] = False
    pool_idx = np.where(test_mask)[0]
    n_train = int(0.8 * len(pool_idx))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pool_idx), generator=g).tolist()
    train_idx = pool_idx[perm[:n_train]].tolist()
    val_idx = pool_idx[perm[n_train:]].tolist()

    model = HybridModelv6(
        seq_emb_dim=SEQ_EMB_DIM, proj_dim=config.proj_dim, gnn_dim=config.gnn_dim,
        gnn_layers=config.gnn_layers, n_heads=config.n_heads, dropout=config.dropout,
        device=device, property_embedding_dim=config.property_emb_dim,
    ).to(device)

    # Route A: initialize sequence tower from EL pretrained weights
    if config.pretrained_seq_path and os.path.exists(config.pretrained_seq_path):
        state = torch.load(config.pretrained_seq_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"[seed {seed}] loaded pretrained seq tower "
                    f"({len(state)} params) from {config.pretrained_seq_path}; "
                    f"unexpected={len(unexpected)}")

    train_loader = GraphDataLoader(
        Subset(V6Dataset(dataset, pep_perres, hla_perres, train=True), train_idx),
        batch_size=config.batch_size, collate_fn=v6_collate,
        shuffle=True, num_workers=config.num_workers, drop_last=True)
    val_loader = GraphDataLoader(
        Subset(V6Dataset(dataset, pep_perres, hla_perres, train=False), val_idx),
        batch_size=config.batch_size, collate_fn=v6_collate,
        shuffle=False, num_workers=config.num_workers)
    test_loader = GraphDataLoader(
        Subset(V6Dataset(dataset, pep_perres, hla_perres, train=False), test_idx),
        batch_size=config.batch_size, collate_fn=v6_collate,
        shuffle=False, num_workers=config.num_workers)

    pos_count = sum(1 for i in train_idx if dataset.immunogenicity[i].item() > 0.5)
    neg_count = n_train - pos_count
    pos_weight = torch.tensor(neg_count / max(pos_count, 1))
    logger.info(f"Seed {seed}: Train {n_train} | Val {len(val_idx)} | Test {len(test_idx)} | "
                f"pos_weight {pos_weight:.4f} (pos={pos_count}, neg={neg_count})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    seed_dir = os.path.join(config.save_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    best_path = os.path.join(seed_dir, "routeA_phase2_best.pt")

    best_val_roc, best_epoch, no_improve = 0.0, 0, 0
    early_stop = config.early_stop if config.early_stop > 0 else config.epochs + 1

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        scheduler.step()
        val_loss, val_roc, val_pr, val_ppvn_cont, val_ppvn_prec = evaluate(
            model, val_loader, device, pos_weight)
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
                logger.info(f"[seed {seed}] Early stop @ ep {epoch}, best Val ROC "
                            f"{best_val_roc:.4f} @ ep {best_epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))

    if len(test_idx) == 0:
        logger.info(f"[seed {seed}] FULL-DATA best Val ROC {best_val_roc:.4f} @ ep {best_epoch} "
                    f"-> checkpoint {best_path}")
        del model; gc.collect(); torch.cuda.empty_cache()
        return {"seed": seed, "test_roc": None, "test_pr": None,
                "test_ppvn_cont": None, "test_ppvn_prec": None,
                "best_val_roc": float(best_val_roc), "best_epoch": int(best_epoch),
                "n_test": 0}

    test_loss, test_roc, test_pr, test_ppvn_cont, test_ppvn_prec = evaluate(
        model, test_loader, device, pos_weight)
    logger.info(f"[seed {seed}] TEST ROC {test_roc:.4f} | PR {test_pr:.4f} | "
                f"PPVn_cont {test_ppvn_cont:.4f} | PPVn_prec {test_ppvn_prec:.4f} "
                f"(best Val ROC {best_val_roc:.4f} @ ep {best_epoch})")

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            graph_data, pep, pep_mask, hla, hla_mask, peptide_property, target = batch
            out = model(graph_data, pep.to(device), hla.to(device),
                        pep_mask.to(device), hla_mask.to(device),
                        peptide_property.to(device)).squeeze(-1)
            preds.append(torch.sigmoid(out).cpu().numpy())
            labels.append(target.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    test_alleles = np.array([dataset.key_list[i].split('_', 1)[0] for i in test_idx])
    np.savez(os.path.join(config.save_dir, f"seed_{seed}_predictions.npz"),
             preds=preds, labels=labels, test_idx=test_idx, test_alleles=test_alleles)

    del model; gc.collect(); torch.cuda.empty_cache()
    return {"seed": seed, "test_roc": float(test_roc), "test_pr": float(test_pr),
            "test_ppvn_cont": float(test_ppvn_cont), "test_ppvn_prec": float(test_ppvn_prec),
            "best_val_roc": float(best_val_roc), "best_epoch": int(best_epoch),
            "n_test": int(len(test_idx))}


def main():
    parser = argparse.ArgumentParser(description="Route A stage 2: finetune on IEDB")
    parser.add_argument("--graph-dir", default=f"{_PROJ_ROOT}/data/graph_pyg_IEDB_esmfold2/")
    parser.add_argument("--property-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_IEDB_data.csv")
    parser.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    parser.add_argument("--pep-emb-path", default=f"{_PROJ_ROOT}/data/esmc_peptide_perres.pt")
    parser.add_argument("--hla-emb-path", default=f"{_PROJ_ROOT}/data/esmc_hla_pseudo_perres.pt")
    parser.add_argument("--pretrained-seq-path", default="", type=str,
                        help="path to stage-1 exported sequence tower (.pt)")
    parser.add_argument("--proj-dim", default=256, type=int)
    parser.add_argument("--gnn-dim", default=64, type=int)
    parser.add_argument("--gnn-layers", default=6, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--property-emb-dim", default=16, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float)
    parser.add_argument("--early-stop", default=30, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--sequence-pad-count", default=0, type=int)
    parser.add_argument("--structure-pad-count", default=0, type=int)
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/routeA_hlaholdout", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--holdout-locus", default="none", choices=["A", "B", "none"])
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--dry-run", action="store_true")
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
    logger.info(f"Save dir: {config.save_dir}")
    logger.info(f"Seeds: {config.seeds} | holdout-locus: {config.holdout_locus}")
    logger.info(f"Pretrained seq tower: {config.pretrained_seq_path or '(none — random init)'}")

    logger.info("Loading IEDB dataset (once)...")
    dataset = ImmunoPredDataset(config, graph_directory=config.graph_dir,
                                property_path=config.property_path,
                                hla_path=config.hla_path)
    logger.info(f"IEDB: {len(dataset):,} samples")

    logger.info("Loading per-residue ESMC embeddings...")
    pep_perres = torch.load(config.pep_emb_path, map_location="cpu")
    hla_perres = torch.load(config.hla_emb_path, map_location="cpu")
    logger.info(f"peptide perres: {len(pep_perres)} | HLA perres: {len(hla_perres)}")

    missing_pep = [k for k in dataset.key_list if k.split('_', 1)[1] not in pep_perres]
    missing_hla = [k for k in dataset.key_list if k.split('_', 1)[0] not in hla_perres]
    if missing_pep or missing_hla:
        logger.error(f"missing pep perres: {len(missing_pep)}; missing HLA perres: {len(missing_hla)}")
        raise SystemExit("per-residue embedding lookup failed")

    if config.holdout_locus in ("A", "B"):
        test_idx = locus_holdout(dataset, config.holdout_locus)
        n_test = int(len(test_idx))
        n_test_pos = int(sum(dataset.immunogenicity[i].item() > 0.5 for i in test_idx))
        logger.info(f"HLA-holdout: test = all HLA-{config.holdout_locus} "
                    f"({n_test} samples, pos={n_test_pos})")
    else:
        test_idx = np.asarray([], dtype=np.int64)
        logger.info(f"Full-data mode: no in-dataset test")

    if config.dry_run:
        test_idx = test_idx[:256]
        config.epochs = 1
        config.early_stop = 0
        config.seeds = [config.seeds[0]]

    results = []
    for seed in config.seeds:
        r = run_seed(seed, config, dataset, pep_perres, hla_perres, device, logger, test_idx)
        results.append(r)
        with open(os.path.join(config.save_dir, f"seed_{seed}_metrics.json"), "w") as f:
            json.dump(r, f, indent=2)

    if len(results) < 2:
        logger.info("Single-seed run; combine with aggregate_seeds.py")
        logger.info("DONE")
        return

    rocs = np.array([r["test_roc"] for r in results])
    prs = np.array([r["test_pr"] for r in results])
    logger.info(f"\n  AUROC: {rocs.mean():.4f} +/- {rocs.std(ddof=1):.4f} (n={len(results)})")
    logger.info(f"  AUPRC: {prs.mean():.4f} +/- {prs.std(ddof=1):.4f}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
