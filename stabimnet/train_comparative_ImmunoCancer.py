#!/usr/bin/env python3
"""Cancer-Wildtype Contrastive Learning (ImmunoStruct upstream recipe).

Trains ``HybridModelv2_Comparative`` on the CEDAR cancer/wildtype paired data.
Each cancer (mutant) epitope is paired with its wildtype counterpart (natural
negative anchor, immunogenicity=0); the model fuses the two embeddings
(``cat([emb_cancer, emb_wt])``) before the immunogenicity classifier.

Usage (single GPU):
    python train_comparative_ImmunoCancer.py \
        --graph-dir-cancer data/graph_pyg_CEDAR_cancer_esmfold2/ \
        --graph-dir-wildtype data/graph_pyg_CEDAR_wildtype_esmfold2/ \
        --property-path-cancer data/ImmunoStruct_CEDAR_data_cancer.csv \
        --property-path-wildtype data/ImmunoStruct_CEDAR_data_wildtype.csv \
        --num-epochs 40 --batch-size 128 --seed 1

Optional explicit contrastive loss (InfoNCE + Barlow Twins hybrid):
    --coeff-contrastive 0.1   (default 0 = implicit concatenation only)

Self-supervision variant (masked amino-acid prediction):
    --self-supervision
"""

import os
import sys
import argparse
import types

# DGL compatibility patch (PyTorch 2.8 + DGL 2.1)
_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
sys.modules["dgl.graphbolt"] = _dgl_gb

import torch

# PyTorch 2.6+ compat: monkey-patch torch.load to weights_only=False
# (PyG graph files contain non-standard globals that can't all be allowlisted)
_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dgl.dataloading import GraphDataLoader

from models.mapping import model_map
from models.comparative_models import transfer_backbone_from_v2
from data_loading import ImmunoPredDatasetComparative, collate, collate_amino_acid, SplitDataset
from utils import seed_everything, update_paths, Losses
from procedures import inference_comparative, inference_comparative_SSL, train_model_comparative, train_model_comparative_SSL


def stratified_split(dataset, labels, ratios, generator):
    """Class-stratified deterministic split into train/val/test Subsets.

    Plain ``random_split`` can drift the positive rate on this imbalanced
    label distribution; stratifying guarantees each split keeps the global
    ~19% positive rate.
    """
    labels = np.asarray(labels)
    parts = [[], [], []]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[torch.randperm(len(idx), generator=generator).numpy()]
        n_test = int(np.ceil(len(idx) * ratios[2]))
        n_val = int(np.ceil(len(idx) * ratios[1]))
        n_train = len(idx) - n_test - n_val
        parts[0].append(idx[:n_train])
        parts[1].append(idx[n_train:n_train + n_val])
        parts[2].append(idx[n_train + n_val:])
    subsets = []
    for p in parts:
        merged = np.concatenate(p)
        merged = merged[torch.randperm(len(merged), generator=generator).numpy()]
        subsets.append(torch.utils.data.Subset(dataset, merged.tolist()))
    return tuple(subsets)


def main():
    parser = argparse.ArgumentParser(description="Cancer-Wildtype Contrastive Learning training.")
    parser.add_argument("--model", default="HybridModelv2_Comparative", type=str)
    parser.add_argument("--use-wt-for-downstream", action="store_true", default=True)
    parser.add_argument("--self-supervision", action="store_true",
                        help="use the SSL model variant (masked amino-acid prediction)")
    parser.add_argument("--coeff-contrastive", default=0.01, type=float,
                        help="coefficient for explicit PairedContrastiveLoss (paper lambda_CW = 0.01)")
    parser.add_argument("--bce-weight", default=2.0, type=float,
                        help="weight on the BCE/prediction term (paper lambda_pred=2.0; upstream original=5.0)")
    parser.add_argument("--mse-weight", default=0.5, type=float,
                        help="weight on the reconstruction MSE term (paper lambda_recon=0.5; upstream original=0.1)")
    parser.add_argument("--kld-weight", default=0.5, type=float,
                        help="weight on the KL divergence term (paper lambda_normal=0.5; upstream original=0.1)")
    parser.add_argument("--pretrained-v2", default="$ROOT/model_weights/v2_phase2_best.pt", type=str,
                        help="path to an IEDB-pretrained HybridModelv2 checkpoint to initialize the encoder backbone")
    parser.add_argument("--vae-hidden-dim", default=768, type=int)
    parser.add_argument("--vae-latent-dim", default=64, type=int)
    parser.add_argument("--property-emb-dim", default=16, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float,
                        help="AdamW weight decay (paper: 1e-6)")
    parser.add_argument("--warmup-epochs", default=10, type=int,
                        help="linear LR warmup epochs before cosine decay (paper: 10)")
    parser.add_argument("--num-epochs", default=40, type=int,
                        help="total fine-tuning epochs (paper: 10 warmup + 30 cosine = 40)")
    parser.add_argument("--early-stop-patience", default=0, type=int,
                        help="stop early if val loss does not improve for this many epochs (0 = disable; paper: fixed 40 epochs)")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--split-seed", default=42, type=int,
                        help="seed for the train/val/test split; fixed across model seeds (paper uses 42)")
    parser.add_argument("--model-save-dir", default="$ROOT/results/comparative_ImmunoCancer/", type=str)
    parser.add_argument("--graph-dir-cancer", default="$ROOT/data/graph_pyg_CEDAR_cancer_esmfold2/", type=str)
    parser.add_argument("--graph-dir-wildtype", default="$ROOT/data/graph_pyg_CEDAR_wildtype_esmfold2/", type=str)
    parser.add_argument("--property-path-cancer", default="$ROOT/data/ImmunoStruct_CEDAR_data_cancer.csv", type=str)
    parser.add_argument("--property-path-wildtype", default="$ROOT/data/ImmunoStruct_CEDAR_data_wildtype.csv", type=str)
    parser.add_argument("--hla-path", default="$ROOT/data/HLA_allele_sequences.csv", type=str)
    parser.add_argument("--sequence-pad-count", default=0, type=int)
    parser.add_argument("--structure-pad-count", default=0, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; load the existing checkpoint and evaluate")
    config = parser.parse_args()

    update_paths(config)

    if config.self_supervision:
        config.model = "HybridModelv2_Comparative_SSL"

    # Model save path.
    pretrained_tag = "ptv2" if config.pretrained_v2 else "scratch"
    loss_tag = ""
    if (config.bce_weight, config.mse_weight, config.kld_weight) != (2.0, 0.5, 0.5):
        loss_tag = f"-bw_{config.bce_weight}-mw_{config.mse_weight}-kw_{config.kld_weight}"
    model_str = f"{config.model}-wtds_{config.use_wt_for_downstream}" + \
        f"-lr_{config.learning_rate}-cc_{config.coeff_contrastive}" + \
        f"-ssl_{config.self_supervision}-{pretrained_tag}-ep_{config.num_epochs}-bs_{config.batch_size}" + \
        f"-seed_{config.seed}-split_{config.split_seed}" + loss_tag
    config.model_save_path_finetune = os.path.join(config.model_save_dir, model_str + "_finetune.pt")

    device = torch.device("cuda" if (torch.cuda.is_available() and not config.cpu) else "cpu")
    seed_everything(config.seed)
    split_generator = torch.Generator().manual_seed(config.split_seed)

    print(f"[config] model={config.model} device={device} coeff_contrastive={config.coeff_contrastive} "
          f"bce_weight={config.bce_weight} mse_weight={config.mse_weight} kld_weight={config.kld_weight}")

    # Build dataset FIRST to derive the true VAE input dimension (full-seq length
    # is data-dependent, e.g. 272-residue HLA + peptide = 281, NOT hardcoded 283).
    dataset_ft = ImmunoPredDatasetComparative(config,
                                              graph_directory_cancer=config.graph_dir_cancer,
                                              graph_directory_wt=config.graph_dir_wildtype,
                                              property_path_cancer=config.property_path_cancer,
                                              property_path_wt=config.property_path_wildtype,
                                              hla_path=config.hla_path)

    # encoded_full_seq_cancer is [N, full_len, 21]
    vae_input_dim = dataset_ft.encoded_full_seq_cancer.shape[1] * dataset_ft.encoded_full_seq_cancer.shape[2]
    print(f"[config] vae_input_dim={vae_input_dim} n_samples={len(dataset_ft)} class_weights={dict(dataset_ft.class_weights)}")

    model = model_map[config.model](vae_input_dim=vae_input_dim, device=device,
                                    vae_hidden_dim=config.vae_hidden_dim,
                                    vae_latent_dim=config.vae_latent_dim,
                                    property_embedding_dim=config.property_emb_dim,
                                    use_wt_for_downstream=config.use_wt_for_downstream)
    model.to(device)

    if config.pretrained_v2:
        transfer_backbone_from_v2(model, config.pretrained_v2, map_location=device)

    # Paper split: 70/15/15 (test n=407 on 2710 samples, matches Table S3).
    # Stratified so the ~19% positive rate is preserved in every fold.
    train_dataset_ft, val_dataset_ft, test_dataset_ft = stratified_split(
        dataset_ft, dataset_ft.immunogenicity_cancer, [0.7, 0.15, 0.15], split_generator)
    for name, split in [("train", train_dataset_ft), ("val", val_dataset_ft), ("test", test_dataset_ft)]:
        idx = np.asarray(split.indices)
        pos = int(dataset_ft.immunogenicity_cancer[idx].sum())
        print(f"[split] {name}: n={len(split)}  pos={pos}  ({100 * pos / len(split):.1f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Paper: linear warmup for `warmup_epochs` then cosine decay for the rest
    # (10 warmup + 30 cosine = 40 total), matching train_v2_multiseed.py.
    # warmup_epochs <= 0 -> plain cosine over num_epochs (pre-paper protocol).
    warmup_epochs = config.warmup_epochs
    if warmup_epochs > 0:
        cosine_epochs = max(config.num_epochs - warmup_epochs, 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    losses = Losses(vae_input_dim, dataset_ft.class_weights, sequence=True,
                    bce_weight=config.bce_weight, mse_weight=config.mse_weight, kld_weight=config.kld_weight)

    # binary=True -> immunogenicity; full=True -> full (MHC+peptide) sequence; comparative=True -> cancer/wt pair
    train_split = SplitDataset(train_dataset_ft, "train", binary=True, full=True,
                               comparative=True, return_amino_acid=config.self_supervision)
    val_split = SplitDataset(val_dataset_ft, "val", binary=True, full=True,
                             comparative=True, return_amino_acid=config.self_supervision)
    test_split = SplitDataset(test_dataset_ft, "test", binary=True, full=True,
                              comparative=True, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        collate_fn = collate_amino_acid
    else:
        collate_fn = collate

    train_loader = GraphDataLoader(train_split, batch_size=config.batch_size, collate_fn=collate_fn,
                                   shuffle=True, num_workers=config.num_workers)
    val_loader = GraphDataLoader(val_split, batch_size=config.batch_size, collate_fn=collate_fn,
                                 shuffle=False, num_workers=config.num_workers)
    test_loader = GraphDataLoader(test_split, batch_size=config.batch_size, collate_fn=collate_fn,
                                  shuffle=False, num_workers=config.num_workers)

    if not config.eval_only:
        if config.self_supervision:
            train_model_comparative_SSL(config, device, model, train_loader, val_loader, optimizer,
                                        losses.BCE_loss_SSL, scheduler, stage="finetune")
        else:
            train_model_comparative(config, device, model, train_loader, val_loader, optimizer,
                                    losses.BCE_loss, scheduler, stage="finetune")

        print("DONE TRAINING; best checkpoint at", config.model_save_path_finetune)

    model.load_trained(config.model_save_path_finetune, new_head=False)

    if config.self_supervision:
        train_stats = inference_comparative_SSL(config, model, train_loader, device)
        test_stats = inference_comparative_SSL(config, model, test_loader, device,
                                               optimal_threshold=train_stats["optimal_threshold"])
    else:
        train_stats = inference_comparative(config, model, train_loader, device)
        test_stats = inference_comparative(config, model, test_loader, device,
                                           optimal_threshold=train_stats["optimal_threshold"])

    print("\n=== TEST METRICS ===")
    for k in ["roc_auc", "pr_auc", "accuracy", "f1", "precision", "recall", "ppvn", "ppv30"]:
        print(f"{k}: {test_stats[k]:.4f}")


if __name__ == "__main__":
    main()
