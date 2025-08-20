import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from stgcn_model import STGCN
from data_utils import get_dataloader


class OccupantSTGCNTester:
    """Tester for ST-GCN occupant monitoring model"""
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        graph_args = {'layout': 'body_head', 'strategy': 'spatial'}
        self.model = STGCN(
            in_channels=3,
            num_class=config['num_classes'],
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=0.0  # No dropout during testing
        ).to(self.device)
        
        # Load trained weights
        self.load_model(model_path)
        self.model.eval()
        
        # Loss function
        if config['task_type'] == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def load_model(self, model_path):
        """Load trained model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # Loading from training checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from training checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Loading from exported model
            self.model.load_state_dict(checkpoint)
            print("Loaded exported model")
    
    def test(self, test_dataloader):
        """Test the model and return metrics"""
        print(f"Testing on device: {self.device}")
        
        all_predictions = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc='Testing'):
                data = batch['data'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                if self.config['task_type'] == 'regression':
                    loss = self.criterion(output.squeeze(), target.squeeze())
                    predictions = output.squeeze().cpu().numpy()
                    targets = target.squeeze().cpu().numpy()
                else:
                    loss = self.criterion(output, target.long())
                    predictions = output.argmax(dim=1).cpu().numpy()
                    targets = target.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_losses.append(loss.item())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        avg_loss = np.mean(all_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets, avg_loss)
        
        return metrics, all_predictions, all_targets
    
    def calculate_metrics(self, predictions, targets, avg_loss):
        """Calculate evaluation metrics"""
        metrics = {'loss': avg_loss}
        
        if self.config['task_type'] == 'regression':
            # Regression metrics
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(targets, predictions)
            
            # Custom accuracy (within threshold)
            threshold = 0.1
            accurate_predictions = np.abs(predictions - targets) < threshold
            accuracy = np.mean(accurate_predictions) * 100
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'accuracy_threshold': accuracy
            })
            
        else:
            # Classification metrics
            accuracy = accuracy_score(targets, predictions) * 100
            precision = precision_score(targets, predictions, average='weighted') * 100
            recall = recall_score(targets, predictions, average='weighted') * 100
            f1 = f1_score(targets, predictions, average='weighted') * 100
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return metrics
    
    def visualize_results(self, predictions, targets, output_dir='test_results'):
        """Create visualizations of test results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config['task_type'] == 'regression':
            # Scatter plot of predictions vs targets
            plt.figure(figsize=(10, 8))
            plt.scatter(targets, predictions, alpha=0.6)
            plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('Predictions vs True Values')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Residuals plot
            residuals = predictions - targets
            plt.figure(figsize=(10, 6))
            plt.scatter(targets, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('True Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Error distribution
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            # Confusion matrix
            cm = confusion_matrix(targets, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def analyze_temporal_patterns(self, test_dataloader, output_dir='test_results'):
        """Analyze temporal patterns in predictions"""
        print("Analyzing temporal patterns...")
        
        sequence_predictions = []
        sequence_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc='Temporal Analysis'):
                data = batch['data'].to(self.device)
                target = batch['target'].to(self.device)
                occupant_status = batch['occupant_status'].to(self.device)
                
                # Get predictions for each timestep in the sequence
                B, C, T, V, M = data.shape
                timestep_predictions = []
                
                for t in range(T):
                    # Use data up to timestep t
                    partial_data = data[:, :, :t+1, :, :]
                    if partial_data.shape[2] < 10:  # Need minimum sequence length
                        continue
                        
                    output = self.model(partial_data)
                    if self.config['task_type'] == 'regression':
                        pred = output.squeeze().cpu().numpy()
                    else:
                        pred = output.argmax(dim=1).cpu().numpy()
                    
                    timestep_predictions.append(pred)
                
                if timestep_predictions:
                    sequence_predictions.append(np.array(timestep_predictions))
                    sequence_targets.append(occupant_status.cpu().numpy())
        
        # Save temporal analysis results
        temporal_results = {
            'sequence_predictions': [seq.tolist() for seq in sequence_predictions],
            'sequence_targets': [seq.tolist() for seq in sequence_targets]
        }
        
        with open(os.path.join(output_dir, 'temporal_analysis.json'), 'w') as f:
            json.dump(temporal_results, f, indent=2)
        
        print(f"Temporal analysis saved to {output_dir}")
    
    def benchmark_inference_speed(self, test_dataloader, num_iterations=100):
        """Benchmark inference speed"""
        print("Benchmarking inference speed...")
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i >= 5:  # Warmup with 5 batches
                    break
                data = batch['data'].to(self.device)
                _ = self.model(data)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i >= num_iterations:
                    break
                
                data = batch['data'].to(self.device)
                
                # Time inference
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                    _ = self.model(data)
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)  # milliseconds
                else:
                    import time
                    start = time.time()
                    _ = self.model(data)
                    end = time.time()
                    elapsed_time = (end - start) * 1000  # convert to milliseconds
                
                times.append(elapsed_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / avg_time  # frames per second
        
        speed_metrics = {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': fps,
            'batch_size': test_dataloader.batch_size
        }
        
        return speed_metrics


def main():
    parser = argparse.ArgumentParser(description='Test ST-GCN for occupant monitoring')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--temporal_analysis', action='store_true', help='Perform temporal analysis')
    parser.add_argument('--benchmark_speed', action='store_true', help='Benchmark inference speed')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'num_classes': 1,
            'task_type': 'regression',
            'sequence_length': 64
        }
    
    # Create test dataloader
    test_dataloader = get_dataloader(
        args.test_data_path,
        batch_size=args.batch_size,
        sequence_length=config['sequence_length'],
        shuffle=False,
        num_workers=4,
        augmentation=False
    )
    
    # Create tester
    tester = OccupantSTGCNTester(args.model_path, config)
    
    # Run tests
    print("Starting model evaluation...")
    metrics, predictions, targets = tester.test(test_dataloader)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.upper()}: {value:.4f}")
        else:
            print(f"{key.upper()}: {value}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'metrics': metrics,
        'config': config,
        'model_path': args.model_path,
        'test_data_path': args.test_data_path
    }
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Optional analyses
    if args.visualize:
        tester.visualize_results(predictions, targets, args.output_dir)
    
    if args.temporal_analysis:
        tester.analyze_temporal_patterns(test_dataloader, args.output_dir)
    
    if args.benchmark_speed:
        speed_metrics = tester.benchmark_inference_speed(test_dataloader)
        print("\n" + "="*50)
        print("SPEED BENCHMARK")
        print("="*50)
        for key, value in speed_metrics.items():
            print(f"{key.upper()}: {value:.2f}")
        
        # Save speed results
        with open(os.path.join(args.output_dir, 'speed_benchmark.json'), 'w') as f:
            json.dump(speed_metrics, f, indent=2)
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
