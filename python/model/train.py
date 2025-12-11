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
import torch.distributed as dist
import torch.multiprocessing as mp
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
    If not provided, fall back to DEFAULT_STREAM_CONFIGS.
    """
    streams_cfg = config.get("streams")
    if streams_cfg is None:
        return DEFAULT_STREAM_CONFIGS

    merged = {}
    # Accept list or dict
    if isinstance(streams_cfg, list):
        for stream in streams_cfg:
            name = stream["name"]
            # Allow partial override of defaults if the name exists there
            base = DEFAULT_STREAM_CONFIGS.get(name, {})
            merged[name] = {**base, **stream}
    elif isinstance(streams_cfg, dict):
        for name, stream in streams_cfg.items():
            base = DEFAULT_STREAM_CONFIGS.get(name, {})
            merged[name] = {**base, **stream, "name": name}
    else:
        raise ValueError("streams must be a list or dict in config.json")
    return merged


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
                "eye_",
                "ear_",
                "shoulder_",
                "elbow_",
                "wrist_",
                "hip_",
                "knee_",
                "ankle_",
            ],
            "face_2d_pattern": "landmark_.*_2d",
            "head_pose_cols": [
                "head_pitch",
                "head_yaw",
                "head_roll",
            ],
        },
    )

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

    cfg.setdefault("model", {})
    cfg["model"].setdefault("num_classes", 5)
    cfg["model"].setdefault("hidden_channels", 64)
    cfg["model"].setdefault("num_gcn_layers", 3)
    cfg["model"].setdefault("temporal_kernel_size", 9)
    cfg["model"].setdefault("dropout", 0.3)

    return cfg


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


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

    # Data
    csv_path = config["data"]["csv_path"]
    if csv_path is None:
        raise ValueError("CSV path must be provided via --path or data.csv_path in config.")
    train_loader, val_loader, _, _, feature_dims, class_weights = create_data_loaders(
        config, csv_path, rank=rank if distributed else None, world_size=world_size if distributed else None
    )

    # Model
    model = MultiStreamSTGCN(config, feature_dims=feature_dims, stream_configs=stream_configs).to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
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

    # Plot learning curves (rank 0 only)
    if (rank == 0 or not distributed) and len(train_losses) > 0:
        plt.figure(figsize=(8, 4))
        epochs_range = list(range(1, len(train_losses) + 1))
        plt.plot(epochs_range, train_losses, label="Train Loss")
        if len(val_losses) > 0:
            val_epochs = list(range(val_interval, val_interval * len(val_losses) + 1, val_interval))
            plt.plot(val_epochs, val_losses, label("Val Loss")
)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = checkpoint_dir / "learning_curve.png"
        plt.savefig(curve_path)
        plt.close()
        print(f"Saved learning curve to {curve_path}")

    if distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Stream STGCN")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (>=2 enables DDP)")
    parser.add_argument("--path", type=str, default=None, help="Path to training CSV file or directory containing CSV files")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ensure_defaults(load_config(config_path))
    if args.path is not None:
        config["data"]["csv_path"] = args.path

    stream_configs = merge_stream_configs(config)

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
