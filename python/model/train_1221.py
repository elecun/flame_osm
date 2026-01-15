#!/usr/bin/env python3
"""
Config-driven training script for Multi-Stream STGCN.

Features
- Loads hyperparameters and stream definitions from JSON (config.json by default)
- Supports per-stream customization (num_nodes, in_channels, adjacency, column patterns)
- Supports single GPU or simple DataParallel when --gpus >= 2
- Saves best-performing checkpoint (by validation loss) under checkpoints/
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import create_data_loaders
from multi_stream_stgcn import MultiStreamSTGCN, DEFAULT_STREAM_CONFIGS
from multi_stream_stgcn_xattn import MultiStreamSTGCNXAttn


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def merge_stream_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert stream config in JSON to a dict usable by MultiStreamSTGCN.
    Only keeps streams where enabled=True (default).
    """
    streams_cfg = config.get("streams")
    if streams_cfg is None:
        return DEFAULT_STREAM_CONFIGS

    merged = {}
    # Accept list or dict
    if isinstance(streams_cfg, list):
        for stream in streams_cfg:
            name = stream["name"]
            base = DEFAULT_STREAM_CONFIGS.get(name, {})
            merged[name] = {**base, **stream}
    elif isinstance(streams_cfg, dict):
        for name, stream in streams_cfg.items():
            base = DEFAULT_STREAM_CONFIGS.get(name, {})
            merged[name] = {**base, **stream, "name": name}
    else:
        raise ValueError("streams must be a list or dict in config.json")

    enabled_streams = {name: cfg for name, cfg in merged.items() if cfg.get("enabled", True)}
    if not enabled_streams:
        raise ValueError("At least one stream must be enabled in config.streams[*].enabled.")

    disabled = [name for name in merged if name not in enabled_streams]
    if disabled:
        print(f"[INFO] Disabled streams: {', '.join(disabled)}")

    return enabled_streams


def create_loss_fn(loss_cfg: Dict[str, Any], class_weights: torch.Tensor | None, device: torch.device):
    name = loss_cfg.get("name", "cross_entropy")
    name = name.lower()
    weight = class_weights.to(device) if class_weights is not None else None

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)

    if name == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        gamma = float(focal_cfg.get("gamma", 2.0))
        alpha_cfg = focal_cfg.get("alpha", None)
        if alpha_cfg is None:
            alpha = weight
        else:
            alpha = torch.tensor(alpha_cfg, device=device, dtype=torch.float32)
        return FocalLoss(gamma=gamma, alpha=alpha)

    if name == "ls_cross_entropy":
        ls_cfg = loss_cfg.get("ls_cross_entropy", {})
        smoothing = float(ls_cfg.get("smoothing", 0.1))
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    if name == "triplet":
        triplet_cfg = loss_cfg.get("triplet", {})
        margin = float(triplet_cfg.get("margin", 1.0))
        return TripletLossWrapper(margin=margin)

    raise ValueError(f"Unsupported loss name: {name}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Dict[str, Any], rank: int):
    if not scheduler_cfg:
        return None

    name = scheduler_cfg.get("name") if isinstance(scheduler_cfg, dict) else None
    if name is None or str(name).lower() in ["none", ""]:
        return None

    name = str(name).lower()
    if name == "reduce_on_plateau":
        roc = scheduler_cfg.get("reduce_on_plateau", {})
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(roc.get("factor", 0.5)),
            patience=int(roc.get("patience", 5)),
            threshold=float(roc.get("threshold", 1e-4)),
            cooldown=int(roc.get("cooldown", 0)),
            min_lr=float(roc.get("min_lr", 1e-6)),
        )

    raise ValueError(f"Unsupported scheduler name: {name}")


def plot_learning_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    learning_rates,
    val_interval,
    checkpoint_dir: Path,
    fold_idx=None,
):
    if len(train_losses) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)
    train_epochs = list(range(1, len(train_losses) + 1))
    val_epochs = list(range(val_interval, val_interval * len(val_losses) + 1, val_interval))

    # Loss
    axes[0].plot(train_epochs, train_losses, label="Train Loss")
    if len(val_losses) > 0:
        axes[0].plot(val_epochs, val_losses, label="Val Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_epochs, train_accs, label="Train Acc")
    if len(val_accs) > 0:
        axes[1].plot(val_epochs, val_accs, label="Val Acc")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    lr_epochs = list(range(1, len(learning_rates) + 1))
    axes[2].plot(lr_epochs, learning_rates, label="Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    curve_path = checkpoint_dir / f"learning_curves{suffix}.png"
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved learning curves (loss/acc/lr) to {curve_path}")


def ensure_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill minimal defaults so training can run even with sparse config files."""
    cfg.setdefault("data", {})
    cfg["data"].setdefault("csv_path", None)
    cfg["data"].setdefault("csv_files", [])
    cfg["data"].setdefault("is_directory", False)
    cfg["data"].setdefault("normalize", True)
    cfg["data"].setdefault("num_workers", 4)
    cfg["data"].setdefault("shuffle", True)
    cfg["data"].setdefault("label_column", "attention_level")
    cfg["data"].setdefault(
        "features",
        {
            "use_body_kps": True,
            "use_face_kps_2d": True,
            "use_head_pose": True,
            "body_pattern": [
                "nose_",
                "left_eye_",
                "right_eye_",
                "left_ear_",
                "right_ear_",
                "left_shoulder_",
                "right_shoulder_",
                "left_elbow_",
                "right_elbow_",
                "left_wrist_",
                "right_wrist_",
            ],
            "face_2d_pattern": "landmark_.*_2d",
            "head_pose_cols": [
                "head_pitch",
                "head_yaw",
                "head_roll",
            ],
        },
    )
    cfg["data"].setdefault("filter", {"enabled": False, "sigma": 1.0})

    cfg.setdefault("training", {})
    cfg["training"].setdefault("epochs", 30)
    cfg["training"].setdefault("batch_size", 32)
    cfg["training"].setdefault("learning_rate", 1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("sequence_length", 60)
    cfg["training"].setdefault("n_folds", 5)
    cfg["training"].setdefault("current_fold", 0)
    cfg["training"].setdefault("val_interval", 1)
    cfg["training"].setdefault("grad_clip", 1.0)
    cfg["training"].setdefault("checkpoint_dir", "checkpoints")
    cfg["training"].setdefault(
        "lr_scheduler",
        {
            "name": None,  # set to "reduce_on_plateau" to enable
            "reduce_on_plateau": {
                "factor": 0.5,
                "patience": 5,
                "threshold": 1e-4,
                "min_lr": 1e-6,
                "cooldown": 0,
            },
        },
    )
    cfg["training"].setdefault(
        "loss",
        {
            "name": "cross_entropy",
            "focal": {"gamma": 2.0, "alpha": None},
            "ls_cross_entropy": {"smoothing": 0.1},
            "triplet": {"margin": 1.0},
        },
    )

    cfg.setdefault("model", {})
    cfg["model"].setdefault("num_classes", 5)
    cfg["model"].setdefault("hidden_channels", 64)
    cfg["model"].setdefault("num_gcn_layers", 3)
    cfg["model"].setdefault("temporal_kernel_size", 9)
    cfg["model"].setdefault("dropout", 0.3)
    cfg["model"].setdefault(
        "cross_attention",
        {
            "enabled": False,
            "d_model": 128,
            "num_heads": 4,
            "dropout": 0.1,
        },
    )

    return cfg


def align_feature_flags_with_streams(config: Dict[str, Any], stream_configs: Dict[str, Dict[str, Any]]) -> None:
    """
    Keep data.feature flags in sync with enabled streams so shapes match the model.
    """
    features_cfg = config.setdefault("data", {}).setdefault("features", {})
    stream_to_feature_flag = {
        "body_pose": "use_body_kps",
        "face_landmarks": "use_face_kps_2d",
        "head_pose": "use_head_pose",
    }

    for stream_name, flag_name in stream_to_feature_flag.items():
        features_cfg[flag_name] = stream_name in stream_configs


def verify_attention_values(loader: DataLoader, num_classes: int, split_name: str, rank: int = 0):
    """
    Sanity check for attention labels to avoid CUDA scatter/gather OOB errors.
    Expects loader.dataset.attention in 0..num_classes-1 after preprocessing.
    """
    ds = getattr(loader, "dataset", None)
    if ds is None or not hasattr(ds, "attention"):
        return
    att = np.array(ds.attention)
    if att.size == 0:
        return
    att_min = att.min()
    att_max = att.max()
    if att_min < 0 or att_max >= num_classes:
        bad_idx = np.where((att < 0) | (att >= num_classes))[0][:5]
        if rank == 0:
            raise ValueError(
                f"[{split_name}] attention_level out of range. "
                f"min={att_min}, max={att_max}, expected 0..{num_classes-1}. "
                f"Bad indices sample: {bad_idx}"
            )


def ensure_targets_fit_logits(logits: torch.Tensor, targets: torch.Tensor, split: str):
    """
    Ensure classification targets are within the logits class dimension to avoid CUDA OOB assertions.
    """
    if logits.ndim < 2 or logits.shape[1] == 0 or targets.numel() == 0:
        return
    num_classes = logits.shape[1]
    t_min = int(targets.min().item())
    t_max = int(targets.max().item())
    if t_min < 0 or t_max >= num_classes:
        raise ValueError(
            f"[{split}] attention_level out of range for model output. "
            f"min={t_min}, max={t_max}, classes={num_classes}. "
            "Check config.model.num_classes or label preprocessing."
        )


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def ensure_distributed_sampler(loader: DataLoader, rank: int, world_size: int, shuffle: bool) -> DataLoader:
    """
    Ensure the given DataLoader uses a DistributedSampler. If it already does, return it as-is,
    otherwise rebuild the loader with a DistributedSampler to avoid replicated data across ranks.
    """
    if isinstance(getattr(loader, "sampler", None), DistributedSampler):
        return loader

    sampler = DistributedSampler(loader.dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    kwargs = {
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "pin_memory": getattr(loader, "pin_memory", False),
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "persistent_workers": getattr(loader, "persistent_workers", False) if loader.num_workers > 0 else False,
    }
    if hasattr(loader, "prefetch_factor") and loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    if hasattr(loader, "pin_memory_device"):
        kwargs["pin_memory_device"] = loader.pin_memory_device

    return DataLoader(loader.dataset, sampler=sampler, **kwargs)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.view(-1, 1)
        gather_log = log_probs.gather(1, targets)
        gather_prob = probs.gather(1, targets)

        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets.squeeze()).unsqueeze(1)
        else:
            alpha_factor = 1.0

        focal_weight = (1 - gather_prob) ** self.gamma
        loss = -alpha_factor * focal_weight * gather_log
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TripletLossWrapper(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)
        self._warned = False

    def forward(self, embeddings, labels):
        anchors, positives, negatives = self._sample_triplets(embeddings, labels)
        if anchors is None:
            if not self._warned:
                print("[WARN] Triplet loss skipped due to insufficient class diversity in batch.")
                self._warned = True
            return embeddings.new_tensor(0.0)
        return self.triplet(anchors, positives, negatives)

    def _sample_triplets(self, embeddings, labels):
        device = embeddings.device
        labels = labels.to(device)
        anchors = []
        positives = []
        negatives = []

        for idx, lbl in enumerate(labels):
            pos_idx = (labels == lbl).nonzero(as_tuple=False).view(-1)
            pos_idx = pos_idx[pos_idx != idx]
            neg_idx = (labels != lbl).nonzero(as_tuple=False).view(-1)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            pos_choice = pos_idx[torch.randint(len(pos_idx), (1,), device=device)].item()
            neg_choice = neg_idx[torch.randint(len(neg_idx), (1,), device=device)].item()
            anchors.append(embeddings[idx])
            positives.append(embeddings[pos_choice])
            negatives.append(embeddings[neg_choice])

        if len(anchors) == 0:
            return None, None, None

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=None, distributed=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for batch in tqdm(loader, desc="Train"):
        batch = to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch.get("body_kps"), batch.get("face_kps_2d"), batch.get("head_pose"))
        ensure_targets_fit_logits(outputs, batch["attention_level"], split="train")
        loss = criterion(outputs, batch["attention_level"])
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += batch["attention_level"].size(0)
        correct += (preds == batch["attention_level"]).sum().item()
        batch_count += 1

    if distributed:
        stats = torch.tensor(
            [running_loss, correct, total, batch_count], device=device, dtype=torch.float64
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss, correct, total, batch_count = stats.tolist()

    avg_loss = running_loss / max(1, batch_count)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def validate(model, loader, criterion, device, distributed=False, num_classes=None, return_stats=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    conf_mat = None
    if return_stats and num_classes is not None:
        conf_mat = torch.zeros((num_classes, num_classes), device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            batch = to_device(batch, device)
            outputs = model(batch.get("body_kps"), batch.get("face_kps_2d"), batch.get("head_pose"))
            ensure_targets_fit_logits(outputs, batch["attention_level"], split="val")
            loss = criterion(outputs, batch["attention_level"])
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += batch["attention_level"].size(0)
            correct += (preds == batch["attention_level"]).sum().item()
            if conf_mat is not None:
                for t, p in zip(batch["attention_level"].view(-1), preds.view(-1)):
                    conf_mat[t, p] += 1

    if distributed:
        stats = torch.tensor([running_loss, correct, total, len(loader)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss, correct, total, batches = stats.tolist()
        if conf_mat is not None:
            dist.all_reduce(conf_mat, op=dist.ReduceOp.SUM)
    else:
        batches = len(loader)

    avg_loss = running_loss / max(1, batches)
    acc = 100.0 * correct / max(1, total)
    if conf_mat is None:
        return avg_loss, acc

    diag = conf_mat.diag()
    total = conf_mat.sum()
    tp_per_class = diag
    fp_per_class = conf_mat.sum(dim=0) - diag
    fn_per_class = conf_mat.sum(dim=1) - diag
    tn_per_class = total - (tp_per_class + fp_per_class + fn_per_class)

    # Per-class precision/recall/F1 for aggregate metrics
    precision_per_class = torch.zeros_like(tp_per_class)
    recall_per_class = torch.zeros_like(tp_per_class)
    f1_per_class = torch.zeros_like(tp_per_class)

    prec_den = tp_per_class + fp_per_class
    rec_den = tp_per_class + fn_per_class

    valid_prec = prec_den > 0
    valid_rec = rec_den > 0

    precision_per_class[valid_prec] = tp_per_class[valid_prec] / prec_den[valid_prec]
    recall_per_class[valid_rec] = tp_per_class[valid_rec] / rec_den[valid_rec]

    pr_sum = precision_per_class + recall_per_class
    valid_f1 = pr_sum > 0
    f1_per_class[valid_f1] = 2 * precision_per_class[valid_f1] * recall_per_class[valid_f1] / pr_sum[valid_f1]

    support = tp_per_class + fn_per_class
    total_support = support.sum()
    if total_support > 0:
        weighted_f1 = (f1_per_class * support).sum().item() / total_support.item()
    else:
        weighted_f1 = 0.0

    stats_out = {
        "tp": tp_per_class.sum().item(),
        "fp": fp_per_class.sum().item(),
        "fn": fn_per_class.sum().item(),
        "tn": tn_per_class.sum().item(),
        "weighted_f1": weighted_f1,
    }
    return avg_loss, acc, stats_out


def save_checkpoint(state, checkpoint_dir: Path, is_best: bool):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(state, latest_path)
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(state, best_path)
        print(f"  [CHECKPOINT] Saved best model to {best_path}")


def run_training(rank, world_size, config, stream_configs, args):
    distributed = world_size > 1
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")

    try:
        # Data
        csv_path = config["data"]["csv_path"]
        if csv_path is None:
            raise ValueError("CSV path must be provided via --path or data.csv_path in config.")
        train_loader, val_loader, _, _, feature_dims, class_weights, _ = create_data_loaders(
            config, csv_path, rank=rank if distributed else None, world_size=world_size if distributed else None
        )
        # Validate attention labels to prevent CUDA OOB assertions
        num_classes = config["model"].get("num_classes", 5)
        verify_attention_values(train_loader, num_classes, split_name="train", rank=rank)
        verify_attention_values(val_loader, num_classes, split_name="val", rank=rank)

        if distributed:
            train_loader = ensure_distributed_sampler(train_loader, rank, world_size, shuffle=config["data"]["shuffle"])
            val_loader = ensure_distributed_sampler(val_loader, rank, world_size, shuffle=False)

        # Model
        if args.model == "xattn":
            model = MultiStreamSTGCNXAttn(config, feature_dims=feature_dims, stream_configs=stream_configs).to(device)
        else:
            model = MultiStreamSTGCN(config, feature_dims=feature_dims, stream_configs=stream_configs).to(device)
        if distributed:
            # xattn + disabled streams can leave some params unused; allow unused params to avoid DDP errors
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Loss / Optimizer
        loss_cfg = config["training"].get("loss", {})
        criterion = create_loss_fn(loss_cfg, class_weights, device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        scheduler = create_scheduler(optimizer, config["training"].get("lr_scheduler", {}), rank)

        # Training loop
        best_val_loss = float("inf")
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        num_epochs = config["training"]["epochs"]
        grad_clip = config["training"].get("grad_clip", None)
        val_interval = max(1, int(config["training"].get("val_interval", 1)))
        early_stop_patience = 10
        no_improve_count = 0

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        learning_rates = []

        for epoch in range(1, num_epochs + 1):
            if distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if rank == 0 or not distributed:
                print(f"\nEpoch {epoch}/{num_epochs}")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, grad_clip, distributed=distributed
            )
            if rank == 0 or not distributed:
                print(f"  Train loss: {train_loss:.4f} | acc: {train_acc:.2f}%")
                train_losses.append(train_loss)
                train_accs.append(train_acc)

            stop_flag = False

            if epoch % val_interval == 0:
                val_loss, val_acc, val_stats = validate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    distributed=distributed,
                    num_classes=config["model"].get("num_classes", 5),
                    return_stats=True,
                )
                if rank == 0 or not distributed:
                    print(f"  Val   loss: {val_loss:.4f} | acc: {val_acc:.2f}%")
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    # Save checkpoint (best + latest)
                    state = {
                        "epoch": epoch,
                        "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    save_checkpoint(state, checkpoint_dir, is_best=is_best)
                    if is_best:
                        fold_idx = config["training"].get("current_fold", 0)
                        acc_path = checkpoint_dir / f"accuracy_fold_{fold_idx}.txt"
                        with open(acc_path, "w") as f:
                            f.write(
                                f"val_accuracy: {val_acc:.4f}\n"
                                f"weighted_f1: {val_stats.get('weighted_f1', 0):.4f}\n"
                                "\n"
                                f"true_positive: {val_stats.get('tp', 0):.0f}\n"
                                f"false_positive: {val_stats.get('fp', 0):.0f}\n"
                                f"false_negative: {val_stats.get('fn', 0):.0f}\n"
                                f"true_negative: {val_stats.get('tn', 0):.0f}\n"
                            )
                        print(f"Saved accuracy stats to {acc_path}")

                # Step LR scheduler on validation metric
                if scheduler is not None:
                    scheduler.step(val_loss)

                if args.early_stop and no_improve_count >= early_stop_patience:
                    stop_flag = True
                    if rank == 0 or not distributed:
                        print(f"[EARLY STOP] No val loss improvement for {early_stop_patience} validation steps. Stopping.")

            if distributed:
                stop_tensor = torch.tensor([1 if stop_flag else 0], device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_flag = bool(stop_tensor.item())

            # Track LR after potential scheduler step
            if rank == 0 or not distributed:
                learning_rates.append(optimizer.param_groups[0]["lr"])

            if rank == 0 or not distributed:
                plot_learning_curves(
                    train_losses,
                    val_losses,
                    train_accs,
                    val_accs,
                    learning_rates,
                    val_interval,
                    checkpoint_dir,
                    fold_idx=config["training"].get("current_fold", None),
                )

            if stop_flag:
                break

        # Plot learning curves (rank 0 only)
        if (rank == 0 or not distributed) and len(train_losses) > 0:
            plot_learning_curves(
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                learning_rates,
                val_interval,
                checkpoint_dir,
                fold_idx=config["training"].get("current_fold", None),
            )
    finally:
        if distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Stream STGCN")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (>=2 enables DDP)")
    parser.add_argument("--path", type=str, default=None, help="Path to training CSV file or directory containing CSV files")
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Ground-truth column to use for labels (default: attention_level). Example: attention_level_t30",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Treat --path as a directory containing multiple case folders to load.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "xattn"],
        help="Model type: base (MultiStreamSTGCN) or xattn (cross-stream attention)",
    )
    parser.add_argument("--filter", action="store_true", help="Apply Gaussian filtering to input features")
    parser.add_argument("--filter-sigma", type=float, default=None, help="Override Gaussian filter sigma")
    parser.add_argument(
        "--fold-loop",
        type=int,
        default=None,
        help="Run n folds sequentially (overrides training.current_fold). Example: --fold-loop 5",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping: stop if val loss does not improve for 10 validation steps",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = ensure_defaults(load_config(config_path))
    if args.path is not None:
        if args.multi:
            case_root = Path(args.path)
            if not case_root.exists() or not case_root.is_dir():
                raise ValueError(f"--multi expects a directory path, got: {case_root}")
            case_dirs = sorted([p for p in case_root.iterdir() if p.is_dir()])
            if not case_dirs:
                raise ValueError(f"No case directories found under: {case_root}")
            base_config["data"]["csv_path"] = [str(p) for p in case_dirs]
        else:
            base_config["data"]["csv_path"] = args.path
    if args.gt is not None:
        base_config.setdefault("data", {})["label_column"] = args.gt

    if args.filter:
        base_config["data"]["filter"]["enabled"] = True
    if args.filter_sigma is not None:
        base_config["data"]["filter"]["sigma"] = args.filter_sigma

    total_folds = args.fold_loop if args.fold_loop is not None else 1
    fold_indices = (
        list(range(total_folds))
        if args.fold_loop is not None
        else [base_config["training"].get("current_fold", 0)]
    )

    for fold_idx in fold_indices:
        config = deepcopy(base_config)
        config["training"]["current_fold"] = fold_idx
        stream_configs = merge_stream_configs(config)
        align_feature_flags_with_streams(config, stream_configs)

        print(f"\n==== Running fold {fold_idx} ====")

        # Decide on GPU usage and distributed setup
        if torch.cuda.is_available() and args.gpus > 1:
            world_size = min(args.gpus, torch.cuda.device_count())
            print(f"Using DDP with {world_size} GPUs")
            mp.spawn(run_training, args=(world_size, config, stream_configs, args), nprocs=world_size, join=True)
        else:
            print("Using single process (CPU or single GPU)")
            run_training(rank=0, world_size=1, config=config, stream_configs=stream_configs, args=args)


if __name__ == "__main__":
    main()
