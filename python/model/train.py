import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

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
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += attention.size(0)
        correct += (predicted == attention).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
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

    return avg_loss, accuracy, predictions, targets


def print_classification_metrics(predictions, targets, class_names=None):
    """
    Print detailed classification metrics including TP, TN, FP, FN for each class

    Args:
        predictions: Predicted labels
        targets: True labels
        class_names: Names of classes (default: 1, 2, 3, 4, 5)
    """
    if class_names is None:
        class_names = ['1', '2', '3', '4', '5']

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)

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
    for i, name in enumerate(class_names):
        print(f"True-{name:3s}|", end=" ")
        for j in range(len(class_names)):
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

    for i, name in enumerate(class_names):
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

        print(f"Class {name:<3} {tp:8d} {tn:8d} {fp:8d} {fn:8d} {precision:10.4f} {recall:10.4f} {f1:10.4f}")

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
    print(classification_report(targets, predictions, target_names=class_names, digits=4))

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

    # Plot 4: Empty (reserved for future use)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def train_model(config, csv_file, device='cuda'):
    """Main training function"""

    # Create output directories
    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['output']['log_dir'], timestamp)
    writer = SummaryWriter(log_dir)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, scaler, feature_dims = create_data_loaders(config, csv_file)

    # Save scaler
    scaler_path = os.path.join(config['output']['model_save_path'], 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # Save feature dimensions
    feature_dims_path = os.path.join(config['output']['model_save_path'], 'feature_dims.pkl')
    with open(feature_dims_path, 'wb') as f:
        pickle.dump(feature_dims, f)
    print(f"Saved feature dimensions to {feature_dims_path}")

    # Create model
    print("\nCreating model...")
    model = AttentionSTGCN(config, feature_dims).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()
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
        'learning_rate': []
    }

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record history (convert to Python native types for JSON serialization)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['learning_rate'].append(float(current_lr))

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config['output']['save_best_only']:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': float(val_loss),
                    'val_acc': float(val_acc),
                    'config': config
                }
                model_path = os.path.join(config['output']['model_save_path'], 'best_model.pth')
                torch.save(checkpoint, model_path)
                print(f"✓ Saved best model to {model_path}")
                print()

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Test on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    # Print detailed classification metrics
    class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    print_classification_metrics(test_preds, test_targets, class_names=class_names)

    # Plot training curves
    print("\nGenerating training curves...")
    curves_path = os.path.join(config['output']['model_save_path'], 'training_curves.png')
    plot_training_curves(history, curves_path)

    # Save training history
    history_path = os.path.join(config['output']['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Save final results (convert to Python native types for JSON serialization)
    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'best_val_loss': float(best_val_loss)
    }

    results_path = os.path.join(config['output']['model_save_path'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {results_path}")

    writer.close()

    return model, results


def get_default_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='Train STGCN for attention prediction')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--csv', type=str, default='merge_0.csv', help='Path to CSV file')
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help='Device to use for training (cuda/mps/cpu)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Train model
    model, results = train_model(config, args.csv, device=args.device)

    print("\nTraining completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.2f}%")


if __name__ == '__main__':
    main()
