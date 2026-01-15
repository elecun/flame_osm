#!/usr/bin/env python3
"""
Training script for Spatial-Temporal Transformer model (STTR) to predict attention_level.
Features mirror train.py (fold-loop, early-stop, LR scheduler, focal/label-smoothing/triplet losses).
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
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


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def ensure_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
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
            "name": None,
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
    cfg["model"].setdefault("d_model", 128)
    cfg["model"].setdefault("num_heads", 4)
    cfg["model"].setdefault("num_layers", 3)
    cfg["model"].setdefault("dropout", 0.3)
    return cfg


def align_feature_flags(config: Dict[str, Any]):
    features = config.setdefault("data", {}).setdefault("features", {})
    # If a feature list is empty, disable the flag; else keep as-is
    for key, flag in [("body_pattern", "use_body_kps"), ("face_2d_pattern", "use_face_kps_2d"), ("head_pose_cols", "use_head_pose")]:
        pattern = features.get(key)
        if pattern is None or (isinstance(pattern, (list, str)) and len(pattern) == 0):
            features[flag] = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class STTRModel(nn.Module):
    def __init__(self, config, feature_dims):
        super().__init__()
        model_cfg = config["model"]
        self.num_classes = model_cfg.get("num_classes", 5)
        self.d_model = model_cfg.get("d_model", 128)
        self.nhead = model_cfg.get("num_heads", 4)
        self.num_layers = model_cfg.get("num_layers", 3)
        dropout = model_cfg.get("dropout", 0.3)

        self.input_dim = 0
        self.use_body = feature_dims.get("body", 0) > 0
        self.use_face = feature_dims.get("face_2d", 0) > 0
        self.use_head = feature_dims.get("head_pose", 0) > 0

        if self.use_body:
            self.input_dim += feature_dims["body"]
        if self.use_face:
            self.input_dim += feature_dims["face_2d"]
        if self.use_head:
            self.input_dim += feature_dims["head_pose"]

        if self.input_dim == 0:
            raise ValueError("No input features enabled for STTRModel.")

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout, max_len=config["training"]["sequence_length"] + 5)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.num_classes),
        )

    def forward(self, body_kps=None, face_kps_2d=None, head_pose=None):
        feats = []
        if self.use_body and body_kps is not None and body_kps.numel() > 0:
            feats.append(body_kps)
        if self.use_face and face_kps_2d is not None and face_kps_2d.numel() > 0:
            feats.append(face_kps_2d)
        if self.use_head and head_pose is not None and head_pose.numel() > 0:
            feats.append(head_pose)
        if len(feats) == 0:
            raise ValueError("No valid inputs for STTRModel forward.")
        x = torch.cat(feats, dim=-1)  # (B, T, F_total)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # pool over time
        logits = self.classifier(x)
        return logits


def create_loss_fn(loss_cfg: Dict[str, Any], class_weights: torch.Tensor | None, device: torch.device):
    name = loss_cfg.get("name", "cross_entropy").lower()
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

        class FocalLoss(nn.Module):
            def __init__(self, gamma, alpha):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha

            def forward(self, logits, targets):
                log_probs = nn.functional.log_softmax(logits, dim=1)
                probs = log_probs.exp()
                targets = targets.view(-1, 1)
                gather_log = log_probs.gather(1, targets)
                gather_prob = probs.gather(1, targets)
                alpha_factor = 1.0
                if self.alpha is not None:
                    alpha_factor = self.alpha.gather(0, targets.squeeze()).unsqueeze(1)
                focal_weight = (1 - gather_prob) ** self.gamma
                loss = -alpha_factor * focal_weight * gather_log
                return loss.mean()

        return FocalLoss(gamma=gamma, alpha=alpha)

    if name == "ls_cross_entropy":
        ls_cfg = loss_cfg.get("ls_cross_entropy", {})
        smoothing = float(ls_cfg.get("smoothing", 0.1))
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    if name == "triplet":
        triplet_cfg = loss_cfg.get("triplet", {})
        margin = float(triplet_cfg.get("margin", 1.0))
        return nn.TripletMarginLoss(margin=margin, p=2)

    raise ValueError(f"Unsupported loss name: {name}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Dict[str, Any]):
    if not scheduler_cfg:
        return None
    name = scheduler_cfg.get("name")
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


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def ensure_distributed_sampler(loader: DataLoader, rank: int, world_size: int, shuffle: bool) -> DataLoader:
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
        stats = torch.tensor([running_loss, correct, total, batch_count], device=device, dtype=torch.float64)
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
        stats = torch.tensor([running_loss, correct, total, len(loader)], device=device, dtype=torch.float64)
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


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, learning_rates, val_interval, checkpoint_dir: Path, fold_idx=None):
    if len(train_losses) == 0:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)
    train_epochs = list(range(1, len(train_losses) + 1))
    val_epochs = list(range(val_interval, val_interval * len(val_losses) + 1, val_interval))

    axes[0].plot(train_epochs, train_losses, label="Train Loss")
    if len(val_losses) > 0:
        axes[0].plot(val_epochs, val_losses, label="Val Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_epochs, train_accs, label="Train Acc")
    if len(val_accs) > 0:
        axes[1].plot(val_epochs, val_accs, label="Val Acc")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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


def run_training(rank, world_size, config, args):
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
        csv_path = config["data"]["csv_path"]
        if csv_path is None:
            raise ValueError("CSV path must be provided via --path or data.csv_path in config.")
        train_loader, val_loader, _, scaler, feature_dims, class_weights, _ = create_data_loaders(
            config,
            csv_path,
            rank=rank if distributed else None,
            world_size=world_size if distributed else None,
        )
        if distributed:
            train_loader = ensure_distributed_sampler(train_loader, rank, world_size, shuffle=config["data"]["shuffle"])
            val_loader = ensure_distributed_sampler(val_loader, rank, world_size, shuffle=False)

        model = STTRModel(config, feature_dims=feature_dims).to(device)
        if distributed:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        loss_cfg = config["training"].get("loss", {})
        criterion = create_loss_fn(loss_cfg, class_weights, device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        scheduler = create_scheduler(optimizer, config["training"].get("lr_scheduler", {}))

        best_val_loss = float("inf")
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        num_epochs = config["training"]["epochs"]
        grad_clip = config["training"].get("grad_clip", None)
        val_interval = max(1, int(config["training"].get("val_interval", 1)))
        early_stop_patience = 20
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
                val_loss, val_acc = validate(model, val_loader, criterion, device, distributed=distributed)
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

                    state = {
                        "epoch": epoch,
                        "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    save_checkpoint(state, checkpoint_dir, is_best=is_best)

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
    parser = argparse.ArgumentParser(description="Train Spatial-Temporal Transformer for attention prediction")
    parser.add_argument("--config", type=str, default="config_sttr.json", help="Path to config JSON")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (>=2 enables DDP)")
    parser.add_argument("--path", type=str, default=None, help="Path to training CSV file or directory containing CSV files")
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
        base_config["data"]["csv_path"] = args.path
    if args.filter:
        base_config["data"]["filter"]["enabled"] = True
    if args.filter_sigma is not None:
        base_config["data"]["filter"]["sigma"] = args.filter_sigma

    align_feature_flags(base_config)

    total_folds = args.fold_loop if args.fold_loop is not None else 1
    fold_indices = (
        list(range(total_folds))
        if args.fold_loop is not None
        else [base_config["training"].get("current_fold", 0)]
    )

    for fold_idx in fold_indices:
        config = deepcopy(base_config)
        config["training"]["current_fold"] = fold_idx
        print(f"\n==== Running fold {fold_idx} ====")

        if torch.cuda.is_available() and args.gpus > 1:
            world_size = min(args.gpus, torch.cuda.device_count())
            print(f"Using DDP with {world_size} GPUs")
            mp.spawn(run_training, args=(world_size, config, args), nprocs=world_size, join=True)
        else:
            print("Using single process (CPU or single GPU)")
            run_training(rank=0, world_size=1, config=config, args=args)


if __name__ == "__main__":
    main()
