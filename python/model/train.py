import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import confusion_matrix, classification_report

from model import AttentionSTGCN
from dataset import create_data_loaders


def setup_ddp(rank, world_size):
    """Initialize DDP environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP environment"""
    dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main process"""
    return rank == 0


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=15, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        body_kps = batch['body_kps'].to(device)
        face_kps_2d = batch['face_kps_2d'].to(device)
        head_pose = batch['head_pose'].to(device)
        attention = batch['attention_level'].to(device)

        # Forward pass (no 3D face landmarks)
        optimizer.zero_grad()
        outputs = model(body_kps, face_kps_2d, head_pose)

        # Compute loss
        loss = criterion(outputs, attention)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        total_grad_norm += grad_norm.item()
        num_batches += 1

        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += attention.size(0)
        correct += (predicted == attention).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%',
            'grad': f'{grad_norm.item():.3f}'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    avg_grad_norm = total_grad_norm / num_batches

    return avg_loss, accuracy, avg_grad_norm


def validate_epoch(model, val_loader, criterion, device, show_metrics=False):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            body_kps = batch['body_kps'].to(device)
            face_kps_2d = batch['face_kps_2d'].to(device)
            head_pose = batch['head_pose'].to(device)
            attention = batch['attention_level'].to(device)

            # Forward pass (no 3D face landmarks)
            outputs = model(body_kps, face_kps_2d, head_pose)

            # Compute loss
            loss = criterion(outputs, attention)

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += attention.size(0)
            correct += (predicted == attention).sum().item()

            predictions.extend(predicted.cpu().numpy())
            targets.extend(attention.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Print simple metrics per class
    if show_metrics:
        unique_classes = np.unique(np.concatenate([predictions, targets]))
        print(f"\n  Per-class metrics:")
        print(f"  {'Class':<6} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10}")
        print(f"  " + "-" * 60)

        cm = confusion_matrix(targets, predictions, labels=unique_classes)
        for i, cls in enumerate(unique_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"  {cls+1:<6} {tp:6d} {tn:6d} {fp:6d} {fn:6d} {precision:10.4f} {recall:10.4f}")

    return avg_loss, accuracy, predictions, targets


def print_classification_metrics(predictions, targets, class_names=None):
    """
    Print detailed classification metrics including TP, TN, FP, FN for each class

    Args:
        predictions: Predicted labels
        targets: True labels
        class_names: Names of classes (default: auto-detect from data)
    """
    # Get unique classes from the data
    unique_classes = np.unique(np.concatenate([predictions, targets]))

    # Create class names based on actual classes present
    if class_names is None:
        class_names = [str(c + 1) for c in unique_classes]  # Convert 0-indexed to 1-indexed
    else:
        # Filter class_names to only include present classes
        class_names = [class_names[i] for i in range(len(class_names)) if i in unique_classes]

    # Confusion matrix with explicit labels
    cm = confusion_matrix(targets, predictions, labels=unique_classes)

    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*80)

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("(Rows: True labels, Columns: Predicted labels)")
    print("\n       ", end="")
    for name in class_names:
        print(f"Pred-{name:3s}", end=" ")
    print()
    print("       " + "-" * (8 * len(class_names)))
    for i in range(len(unique_classes)):
        print(f"True-{class_names[i]:3s}|", end=" ")
        for j in range(len(unique_classes)):
            print(f"{cm[i, j]:6d}", end=" ")
        print()

    # Calculate metrics for each class
    print("\n" + "-"*80)
    print("Per-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<8} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-"*80)

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for i in range(len(unique_classes)):
        # Calculate TP, TN, FP, FN for class i
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Class {class_names[i]:<3} {tp:8d} {tn:8d} {fp:8d} {fn:8d} {precision:10.4f} {recall:10.4f} {f1:10.4f}")

    print("-"*80)

    # Overall metrics
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {len(targets)}")
    print(f"  Correct predictions (TP for all classes): {(predictions == targets).sum()}")
    print(f"  Incorrect predictions (FP+FN): {len(targets) - (predictions == targets).sum()}")
    print(f"  Overall Accuracy: (TP+TN)/(TP+TN+FP+FN) = {100 * (predictions == targets).sum() / len(targets):.2f}%")
    print(f"\n  Note: For multi-class classification,")
    print(f"    - Total TP across all classes: {total_tp}")
    print(f"    - Total FP across all classes: {total_fp}")
    print(f"    - Total FN across all classes: {total_fn}")
    print(f"    - Total TN across all classes: {total_tn}")

    # Sklearn classification report
    print("\n" + "-"*80)
    print("Classification Report (sklearn):")
    print("-"*80)
    print(classification_report(targets, predictions, labels=unique_classes, target_names=class_names, digits=4))

    print("="*80 + "\n")


def plot_training_curves(history, save_path):
    """
    Plot training curves (loss, accuracy, learning rate)

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (CrossEntropy)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Gradient Norm
    axes[1, 1].plot(epochs, history['grad_norm'], 'm-', linewidth=2, label='Gradient Norm')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Gradient Norm', fontsize=12)
    axes[1, 1].set_title('Gradient Norm (Clipped at 1.0)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clip threshold')
    axes[1, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def train_model(config, csv_file, device='cuda', fold_idx=None, rank=None, world_size=None):
    """Main training function with k-fold cross validation support and DDP

    Args:
        config: Configuration dictionary
        csv_file: Path to CSV file
        device: Device to use for training
        fold_idx: Current fold index (0 to n_folds-1). If None, uses config value.
        rank: Rank of current process for DDP (None for single GPU)
        world_size: Total number of processes for DDP (None for single GPU)
    """
    # Get fold information
    n_folds = config['training'].get('n_folds', 5)
    if fold_idx is None:
        fold_idx = config['training'].get('current_fold', 0)

    # Check if this is the main process (for logging and saving)
    is_main = rank is None or rank == 0
    use_ddp = rank is not None and world_size is not None

    # Create output directories (only main process)
    if is_main:
        fold_save_path = os.path.join(config['output']['model_save_path'], f'fold_{fold_idx}')
        os.makedirs(fold_save_path, exist_ok=True)
        os.makedirs(config['output']['log_dir'], exist_ok=True)

        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(config['output']['log_dir'], f'fold_{fold_idx}_{timestamp}')
        writer = SummaryWriter(log_dir)

        print(f"\n{'='*80}")
        print(f"Training Fold {fold_idx}/{n_folds-1}")
        if use_ddp:
            print(f"Using DDP with {world_size} GPUs")
        print(f"{'='*80}")
    else:
        fold_save_path = os.path.join(config['output']['model_save_path'], f'fold_{fold_idx}')
        writer = None

    # Create data loaders
    if is_main:
        print("Creating data loaders...")
    train_loader, val_loader, test_loader, scaler, feature_dims, class_weights = create_data_loaders(
        config, csv_file, fold_idx, rank, world_size
    )

    # Save scaler and feature dims (only main process)
    if is_main:
        scaler_path = os.path.join(fold_save_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {scaler_path}")

        feature_dims_path = os.path.join(fold_save_path, 'feature_dims.pkl')
        with open(feature_dims_path, 'wb') as f:
            pickle.dump(feature_dims, f)
        print(f"Saved feature dimensions to {feature_dims_path}")

    # Create model
    if is_main:
        print("\nCreating model...")

    # Set device
    if use_ddp:
        device = rank  # Use GPU corresponding to rank
        model = AttentionSTGCN(config, feature_dims).to(device)
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model = AttentionSTGCN(config, feature_dims).to(device)

    # Count parameters (only main process)
    if is_main:
        # Access the underlying model if using DDP
        model_for_counting = model.module if use_ddp else model
        total_params = sum(p.numel() for p in model_for_counting.parameters())
        trainable_params = sum(p.numel() for p in model_for_counting.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer (CrossEntropyLoss with label smoothing and class weights)
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    if use_ddp:
        class_weights = class_weights.to(rank)
    else:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if is_main:
        print(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
        print(f"Using dynamic class weights: {np.round(class_weights.cpu().numpy(), 4)}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['lr_scheduler_patience'],
        factor=config['training']['lr_scheduler_factor']
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        verbose=True
    )

    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')

    # Initialize history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rate': [],
        'grad_norm': []
    }

    if is_main:
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {device}")
        print()

    for epoch in range(num_epochs):
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)

        if is_main:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, show_metrics=False
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record history and log (only main process)
        if is_main:
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['train_acc'].append(float(train_acc))
            history['val_acc'].append(float(val_acc))
            history['learning_rate'].append(float(current_lr))
            history['grad_norm'].append(float(grad_norm))

            # Log metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            writer.add_scalar('GradNorm/train', grad_norm, epoch)

            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, Grad Norm: {grad_norm:.3f}")
            print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            print()

        # Save best model (only main process)
        if is_main and val_loss < best_val_loss:
            best_val_loss = val_loss
            if config['output']['save_best_only']:
                # Save the underlying model if using DDP
                model_to_save = model.module if use_ddp else model
                checkpoint = {
                    'epoch': epoch,
                    'fold_idx': fold_idx,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': float(val_loss),
                    'val_acc': float(val_acc),
                    'config': config
                }
                model_path = os.path.join(fold_save_path, 'best_model.pth')
                torch.save(checkpoint, model_path)
                print(f"✓ Saved best model to {model_path}")
                print()

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            if is_main:
                print("Early stopping triggered!")
            break

    # Synchronize all processes before evaluation
    if use_ddp:
        dist.barrier()

    # Test on test set (only main process shows detailed metrics)
    if is_main:
        print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device, show_metrics=(is_main)
    )

    if is_main:
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

        # Print detailed classification metrics
        class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
        print_classification_metrics(test_preds, test_targets, class_names=class_names)

    # Plot training curves and save results (only main process)
    if is_main:
        print("\nGenerating training curves...")
        curves_path = os.path.join(fold_save_path, 'training_curves.png')
        plot_training_curves(history, curves_path)

        # Save training history
        history_path = os.path.join(fold_save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {history_path}")

        # Save final results (convert to Python native types for JSON serialization)
        results = {
            'fold_idx': fold_idx,
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'best_val_loss': float(best_val_loss)
        }

        results_path = os.path.join(fold_save_path, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {results_path}")

        writer.close()
    else:
        # Non-main processes still need to return results for consistency
        results = {
            'fold_idx': fold_idx,
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'best_val_loss': float(best_val_loss)
        }

    return model, results


def get_default_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def run_ddp_training(rank, world_size, config, csv_path, fold_indices, class_weights):
    """
    Run training for a specific process in DDP setup

    Args:
        rank: Rank of current process 
        world_size: Total number of processes
        config: Configuration dictionary
        csv_path: Path to CSV file
        fold_indices: List of fold indices to train
    """
    # Setup DDP
    setup_ddp(rank, world_size)

    # Train each fold
    all_results = []
    for fold_idx in fold_indices:
        model, results = train_model(
            config, csv_path, device='cuda', fold_idx=fold_idx,
            rank=rank, world_size=world_size
        )
        # Only main process collects results
        if rank == 0:
            all_results.append(results)

    # Cleanup DDP
    cleanup_ddp()

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Train STGCN for attention prediction with k-fold cross validation and DDP')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--csv', type=str, default='merge_0.csv', help='Path to CSV file')
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help='Device to use for training (cuda/mps/cpu)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (0 to n_folds-1). If not specified, trains all folds.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for DDP training (default: 1, single GPU)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    n_folds = config['training'].get('n_folds', 5)

    # Determine which folds to train
    if args.fold is not None:
        # Train specific fold
        if args.fold < 0 or args.fold >= n_folds:
            raise ValueError(f"Fold index must be between 0 and {n_folds-1}")
        fold_indices = [args.fold]
    else:
        # Train all folds
        fold_indices = list(range(n_folds))

    # Validate GPU count
    if args.num_gpus > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot use multiple GPUs.")
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            raise ValueError(f"Requested {args.num_gpus} GPUs but only {available_gpus} are available.")
        print(f"\nUsing DDP with {args.num_gpus} GPUs")
    elif args.num_gpus == 1:
        print(f"\nUsing single GPU training")
    else:
        raise ValueError("num_gpus must be >= 1")

    # Train with DDP or single GPU
    if args.num_gpus > 1:
        # Multi-GPU training with DDP
        mp.spawn(
            run_ddp_training,
            args=(args.num_gpus, config, args.csv, fold_indices, None),  # class_weights will be loaded inside
            nprocs=args.num_gpus,
            join=True
        )
        # Load results from saved files (since only rank 0 saves)
        all_results = []
        for fold_idx in fold_indices:
            results_path = os.path.join(config['output']['model_save_path'], f'fold_{fold_idx}', 'results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    all_results.append(results)
    else:
        # Single GPU training
        all_results = []
        for fold_idx in fold_indices:
            _, results = train_model(config, args.csv, device=args.device, fold_idx=fold_idx)
            all_results.append(results)

    # Print summary
    print("\n" + "="*80)
    print("CROSS VALIDATION SUMMARY")
    print("="*80)

    if len(all_results) > 1:
        # Multiple folds trained
        print(f"\nResults for {len(all_results)} folds:")
        print(f"{'Fold':<6} {'Val Loss':>12} {'Test Acc':>12}")
        print("-" * 32)

        val_losses = []
        test_accs = []

        for result in all_results:
            fold_idx = result['fold_idx']
            val_loss = result['best_val_loss']
            test_acc = result['test_acc']

            val_losses.append(val_loss)
            test_accs.append(test_acc)

            print(f"{fold_idx:<6} {val_loss:12.4f} {test_acc:11.2f}%")

        print("-" * 32)
        print(f"{'Mean':<6} {np.mean(val_losses):12.4f} {np.mean(test_accs):11.2f}%")
        print(f"{'Std':<6} {np.std(val_losses):12.4f} {np.std(test_accs):11.2f}%")

        # Save overall summary
        summary = {
            'n_folds': len(all_results),
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),
            'mean_test_acc': float(np.mean(test_accs)),
            'std_test_acc': float(np.std(test_accs)),
            'fold_results': all_results
        }

        summary_path = os.path.join(config['output']['model_save_path'], 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved cross-validation summary to {summary_path}")

    else:
        # Single fold trained
        result = all_results[0]
        print(f"\nFold {result['fold_idx']} completed!")
        print(f"Best validation loss: {result['best_val_loss']:.4f}")
        print(f"Test Accuracy: {result['test_acc']:.2f}%")

    print("="*80)


if __name__ == '__main__':
    main()
