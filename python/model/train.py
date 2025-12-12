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
from pathlib import Path
from typing import Dict, Any

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


def plot_learning_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    learning_rates,
    val_interval,
    checkpoint_dir: Path,
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
    curve_path = checkpoint_dir / "learning_curves.png"
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
                "left_hip_",
                "right_hip_",
                "left_knee_",
                "right_knee_",
                "left_ankle_",
                "right_ankle_",
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


def validate(model, loader, criterion, device, distributed=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            batch = to_device(batch, device)
            outputs = model(batch.get("body_kps"), batch.get("face_kps_2d"), batch.get("head_pose"))
            loss = criterion(outputs, batch["attention_level"])
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += batch["attention_level"].size(0)
            correct += (preds == batch["attention_level"]).sum().item()

    if distributed:
        stats = torch.tensor(
            [running_loss, correct, total, len(loader)], device=device, dtype=torch.float64
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss, correct, total, batches = stats.tolist()
    else:
        batches = len(loader)

    avg_loss = running_loss / max(1, batches)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


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
        train_loader, val_loader, _, _, feature_dims, class_weights = create_data_loaders(
            config, csv_path, rank=rank if distributed else None, world_size=world_size if distributed else None
        )

        if distributed:
            train_loader = ensure_distributed_sampler(train_loader, rank, world_size, shuffle=config["data"]["shuffle"])
            val_loader = ensure_distributed_sampler(val_loader, rank, world_size, shuffle=False)

        # Model
        model = MultiStreamSTGCN(config, feature_dims=feature_dims, stream_configs=stream_configs).to(device)
        if distributed:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # Loss / Optimizer
        loss_cfg = config["training"].get("loss", {})
        criterion = create_loss_fn(loss_cfg, class_weights, device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Training loop
        best_val_loss = float("inf")
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        num_epochs = config["training"]["epochs"]
        grad_clip = config["training"].get("grad_clip", None)
        val_interval = max(1, int(config["training"].get("val_interval", 1)))

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
                learning_rates.append(optimizer.param_groups[0]["lr"])

            if epoch % val_interval == 0:
                val_loss, val_acc = validate(model, val_loader, criterion, device, distributed=distributed)
                if rank == 0 or not distributed:
                    print(f"  Val   loss: {val_loss:.4f} | acc: {val_acc:.2f}%")
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    # Save checkpoint (best + latest)
                    state = {
                        "epoch": epoch,
                        "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    save_checkpoint(state, checkpoint_dir, is_best=is_best)

            if rank == 0 or not distributed:
                plot_learning_curves(
                    train_losses,
                    val_losses,
                    train_accs,
                    val_accs,
                    learning_rates,
                    val_interval,
                    checkpoint_dir,
                )

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
            )
    finally:
        if distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Stream STGCN")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (>=2 enables DDP)")
    parser.add_argument("--path", type=str, default=None, help="Path to training CSV file or directory containing CSV files")
    parser.add_argument("--filter", action="store_true", help="Apply Gaussian filtering to input features")
    parser.add_argument("--filter-sigma", type=float, default=None, help="Override Gaussian filter sigma")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ensure_defaults(load_config(config_path))
    if args.path is not None:
        config["data"]["csv_path"] = args.path

    if args.filter:
        config["data"]["filter"]["enabled"] = True
    if args.filter_sigma is not None:
        config["data"]["filter"]["sigma"] = args.filter_sigma

    stream_configs = merge_stream_configs(config)
    align_feature_flags_with_streams(config, stream_configs)

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
