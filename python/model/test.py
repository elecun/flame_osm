import torch
import torch.nn as nn
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os

from model import AttentionSTGCN
from dataset import AttentionDataset


def load_model(checkpoint_path, feature_dims_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Load feature dimensions
    with open(feature_dims_path, 'rb') as f:
        feature_dims = pickle.load(f)

    model = AttentionSTGCN(config, feature_dims).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation MAE: {checkpoint['val_mae']:.4f}")
    print(f"Validation RMSE: {checkpoint['val_rmse']:.4f}")

    return model, config, feature_dims


def test_model(model, test_loader, device='cuda'):
    """Test model on test data"""
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            body_kps = batch['body_kps'].to(device)
            face_kps_2d = batch['face_kps_2d'].to(device)
            head_pose = batch['head_pose'].to(device)
            attention = batch['attention'].to(device)

            # Forward pass (no 3D face landmarks)
            outputs = model(body_kps, face_kps_2d, head_pose)

            # Compute loss
            loss = criterion(outputs, attention)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(attention.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100

    # Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]

    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'predictions': predictions,
        'targets': targets
    }


def plot_results(predictions, targets, save_path='test_results.png'):
    """Plot prediction vs ground truth"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Scatter plot
    axes[0, 0].scatter(targets, predictions, alpha=0.5)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Ground Truth Attention')
    axes[0, 0].set_ylabel('Predicted Attention')
    axes[0, 0].set_title('Prediction vs Ground Truth')
    axes[0, 0].grid(True)

    # Time series plot
    indices = np.arange(min(len(predictions), 500))
    axes[0, 1].plot(indices, targets[indices], label='Ground Truth', alpha=0.7)
    axes[0, 1].plot(indices, predictions[indices], label='Prediction', alpha=0.7)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Attention')
    axes[0, 1].set_title('Time Series Comparison (First 500 samples)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Error distribution
    errors = predictions - targets
    axes[1, 0].hist(errors, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Error Distribution (Mean: {errors.mean():.4f}, Std: {errors.std():.4f})')
    axes[1, 0].grid(True, alpha=0.3)

    # Absolute error over time
    abs_errors = np.abs(errors)
    axes[1, 1].plot(indices, abs_errors[indices])
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error Over Time (First 500 samples)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test STGCN for attention prediction')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, default='merge_0.csv',
                        help='Path to CSV file')
    parser.add_argument('--scaler', type=str, default='checkpoints/scaler.pkl',
                        help='Path to scaler file')
    parser.add_argument('--feature_dims', type=str, default='checkpoints/feature_dims.pkl',
                        help='Path to feature dimensions file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for testing')
    parser.add_argument('--output', type=str, default='test_results.png',
                        help='Output path for results plot')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, config, feature_dims = load_model(args.checkpoint, args.feature_dims, device=args.device)

    # Load scaler
    print("\nLoading scaler...")
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)

    # Create test dataset
    print("\nCreating test dataset...")

    # Get feature configuration
    feature_config = config['data'].get('features', None)

    # Load full dataset and use last portion as test
    df = pd.read_csv(args.csv)
    total_samples = len(df)
    train_split = config['training']['train_split']
    val_split = config['training']['validation_split']

    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)

    test_df = df[train_size + val_size:]

    # Save temporary test file
    test_file = 'test_temp.csv'
    test_df.to_csv(test_file, index=False)

    test_dataset = AttentionDataset(
        test_file,
        sequence_length=config['training']['sequence_length'],
        normalize=config['data']['normalize'],
        train=False,
        scaler=scaler,
        feature_config=feature_config
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Test model
    print("\nTesting model...")
    results = test_model(model, test_loader, device=args.device)

    # Print results
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Loss (MSE): {results['loss']:.4f}")
    print(f"MAE:        {results['mae']:.4f}")
    print(f"RMSE:       {results['rmse']:.4f}")
    print(f"MAPE:       {results['mape']:.2f}%")
    print(f"Correlation: {results['correlation']:.4f}")
    print("=" * 50)

    # Plot results
    print("\nGenerating plots...")
    plot_results(results['predictions'], results['targets'], save_path=args.output)

    # Save predictions
    predictions_df = pd.DataFrame({
        'ground_truth': results['targets'],
        'prediction': results['predictions'],
        'error': results['predictions'] - results['targets'],
        'abs_error': np.abs(results['predictions'] - results['targets'])
    })

    predictions_path = args.output.replace('.png', '_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")

    # Clean up
    os.remove(test_file)

    print("\nTesting completed!")


if __name__ == '__main__':
    main()
