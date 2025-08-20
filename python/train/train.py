import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from stgcn_model import STGCN
from data_utils import get_dataloader, create_sample_data
from model_export import export_model


class OccupantSTGCNTrainer:
    """Trainer for ST-GCN occupant monitoring model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        graph_args = {'layout': 'body_head', 'strategy': 'spatial'}
        self.model = STGCN(
            in_channels=3,
            num_class=config['num_classes'],
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=config['dropout']
        ).to(self.device)
        
        # Loss and optimizer
        if config['task_type'] == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.best_loss = float('inf')
        self.start_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            data = batch['data'].to(self.device)  # (B, C, T, V, M)
            target = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate loss
            if self.config['task_type'] == 'regression':
                loss = self.criterion(output.squeeze(), target.squeeze())
            else:
                loss = self.criterion(output, target.long())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                data = batch['data'].to(self.device)
                target = batch['target'].to(self.device)
                
                output = self.model(data)
                
                if self.config['task_type'] == 'regression':
                    loss = self.criterion(output.squeeze(), target.squeeze())
                    # Calculate accuracy for regression (within threshold)
                    pred_error = torch.abs(output.squeeze() - target.squeeze())
                    correct += (pred_error < 0.1).sum().item()
                else:
                    loss = self.criterion(output, target.long())
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                
                total_loss += loss.item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        # Log validation metrics
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['loss']
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return True
        return False
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            if val_dataloader is not None:
                val_loss, val_acc = self.validate(val_dataloader, epoch)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save checkpoint
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best)
            else:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
                self.save_checkpoint(epoch, train_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Early stopping
            if self.config.get('early_stopping_patience'):
                # Implement early stopping logic here if needed
                pass
        
        # Export final model
        if self.config.get('export_model', True):
            self.export_trained_model()
        
        self.writer.close()
        print("Training completed!")
    
    def export_trained_model(self):
        """Export trained model to different formats"""
        print("Exporting trained model...")
        
        # Load best model
        best_model_path = os.path.join(self.config['checkpoint_dir'], 'best.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Export to different formats
        export_model(
            self.model,
            input_shape=(1, 3, self.config['sequence_length'], 20, 1),
            output_dir=self.config['export_dir'],
            device=self.device
        )


def get_default_config():
    """Get default training configuration"""
    return {
        'data_path': 'data/train_data.json',
        'val_data_path': 'data/val_data.json',
        'batch_size': 16,
        'sequence_length': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'dropout': 0.5,
        'grad_clip': 1.0,
        'lr_step_size': 30,
        'lr_gamma': 0.1,
        'num_classes': 1,  # For regression, use 1. For classification, set to number of classes
        'task_type': 'regression',  # 'regression' or 'classification'
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'export_dir': 'exported_models',
        'resume': None,
        'num_workers': 4,
        'early_stopping_patience': 10,
        'export_model': True
    }


def main():
    parser = argparse.ArgumentParser(description='Train ST-GCN for occupant monitoring')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--val_data_path', type=str, help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--create_sample_data', action='store_true', help='Create sample data for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.val_data_path:
        config['val_data_path'] = args.val_data_path
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.resume:
        config['resume'] = args.resume
    
    # Create sample data if requested
    if args.create_sample_data:
        os.makedirs('data', exist_ok=True)
        create_sample_data('data/train_data.json', num_samples=800, sequence_length=config['sequence_length'])
        create_sample_data('data/val_data.json', num_samples=200, sequence_length=config['sequence_length'])
        print("Sample data created successfully!")
        return
    
    # Create data loaders
    if not os.path.exists(config['data_path']):
        print(f"Training data not found at {config['data_path']}")
        print("Run with --create_sample_data to create sample data for testing")
        return
    
    train_dataloader = get_dataloader(
        config['data_path'],
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        shuffle=True,
        num_workers=config['num_workers'],
        augmentation=True
    )
    
    val_dataloader = None
    if config.get('val_data_path') and os.path.exists(config['val_data_path']):
        val_dataloader = get_dataloader(
            config['val_data_path'],
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            shuffle=False,
            num_workers=config['num_workers'],
            augmentation=False
        )
    
    # Create trainer
    trainer = OccupantSTGCNTrainer(config)
    
    # Resume from checkpoint if specified
    if config.get('resume'):
        trainer.load_checkpoint(config['resume'])
    
    # Start training
    trainer.train(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
