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
    predictions = []
    targets = []

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        body_kps = batch['body_kps'].to(device)
        face_kps_2d = batch['face_kps_2d'].to(device)
        face_kps_3d = batch['face_kps_3d'].to(device)
        head_pose = batch['head_pose'].to(device)
        attention = batch['attention'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(body_kps, face_kps_2d, face_kps_3d, head_pose)

        # Compute loss
        loss = criterion(outputs, attention)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(attention.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return avg_loss, mae, rmse


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            body_kps = batch['body_kps'].to(device)
            face_kps_2d = batch['face_kps_2d'].to(device)
            face_kps_3d = batch['face_kps_3d'].to(device)
            head_pose = batch['head_pose'].to(device)
            attention = batch['attention'].to(device)

            # Forward pass
            outputs = model(body_kps, face_kps_2d, face_kps_3d, head_pose)

            # Compute loss
            loss = criterion(outputs, attention)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(attention.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return avg_loss, mae, rmse, predictions, targets


def plot_training_curves(history, save_path):
    """
    Plot training curves (loss, MAE, RMSE, learning rate)

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
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: MAE
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: RMSE
    axes[1, 0].plot(epochs, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
    axes[1, 0].plot(epochs, history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('RMSE', fontsize=12)
    axes[1, 0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

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
    train_loader, val_loader, test_loader, scaler = create_data_loaders(config, csv_file)

    # Save scaler
    scaler_path = os.path.join(config['output']['model_save_path'], 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # Create model
    print("\nCreating model...")
    model = AttentionSTGCN(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
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
        'train_mae': [],
        'val_mae': [],
        'train_rmse': [],
        'val_rmse': [],
        'learning_rate': []
    }

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_mae, train_rmse = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_mae, val_rmse, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['learning_rate'].append(current_lr)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/train', train_rmse, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        print(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
        print()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config['output']['save_best_only']:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
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
    test_loss, test_mae, test_rmse, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    # Plot training curves
    print("\nGenerating training curves...")
    curves_path = os.path.join(config['output']['model_save_path'], 'training_curves.png')
    plot_training_curves(history, curves_path)

    # Save training history
    history_path = os.path.join(config['output']['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Save final results
    results = {
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'best_val_loss': best_val_loss
    }

    results_path = os.path.join(config['output']['model_save_path'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {results_path}")

    writer.close()

    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train STGCN for attention prediction')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--csv', type=str, default='merge_0.csv', help='Path to CSV file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Train model
    model, results = train_model(config, args.csv, device=args.device)

    print("\nTraining completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")


if __name__ == '__main__':
    main()
