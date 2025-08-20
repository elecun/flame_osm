import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Tuple, Dict, Optional


class OccupantDataset(Dataset):
    """Dataset for occupant monitoring with body keypoints, head pose, and status"""
    
    def __init__(self, data_path: str, sequence_length: int = 64, transform=None):
        """
        Args:
            data_path: Path to dataset directory or file
            sequence_length: Number of frames in each sequence
            transform: Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load data
        self.data_samples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load data from files"""
        samples = []
        
        if os.path.isfile(self.data_path):
            # Single file
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                samples.extend(data)
        elif os.path.isdir(self.data_path):
            # Directory with multiple files
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.data_path, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        samples.extend(data)
        
        return samples
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Extract components
        keypoints = np.array(sample['keypoints'])  # Shape: (T, 17, 3) - 17 body keypoints with x,y,confidence
        head_pose = np.array(sample['head_pose'])  # Shape: (T, 6) - 3D position + 3D rotation
        occupant_status = np.array(sample['occupant_status'])  # Shape: (T,) - status values 0-1
        
        # Process data
        processed_data = self._process_sample(keypoints, head_pose, occupant_status)
        
        if self.transform:
            processed_data = self.transform(processed_data)
            
        return processed_data
    
    def _process_sample(self, keypoints: np.ndarray, head_pose: np.ndarray, occupant_status: np.ndarray) -> Dict:
        """Process raw data into model input format"""
        T = keypoints.shape[0]
        
        # Ensure sequence length
        if T > self.sequence_length:
            # Randomly sample a subsequence
            start_idx = np.random.randint(0, T - self.sequence_length + 1)
            keypoints = keypoints[start_idx:start_idx + self.sequence_length]
            head_pose = head_pose[start_idx:start_idx + self.sequence_length]
            occupant_status = occupant_status[start_idx:start_idx + self.sequence_length]
        elif T < self.sequence_length:
            # Pad sequence
            pad_length = self.sequence_length - T
            keypoints = np.pad(keypoints, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
            head_pose = np.pad(head_pose, ((0, pad_length), (0, 0)), mode='constant')
            occupant_status = np.pad(occupant_status, (0, pad_length), mode='constant')
        
        # Create graph input: combine keypoints and head pose
        # Keypoints: (T, 17, 3) -> (T, 17, 3)
        # Head pose: (T, 6) -> (T, 3, 2) for position and orientation nodes
        head_pos = head_pose[:, :3].reshape(self.sequence_length, 1, 3)  # Position
        head_rot = head_pose[:, 3:].reshape(self.sequence_length, 1, 3)  # Rotation
        head_conf = np.ones((self.sequence_length, 2, 1))  # Confidence for head nodes
        
        # Combine head position and rotation as separate nodes
        head_nodes = np.concatenate([
            np.concatenate([head_pos, head_conf[:, :1]], axis=2),  # Position node
            np.concatenate([head_rot, head_conf[:, 1:]], axis=2)   # Rotation node
        ], axis=1)
        
        # Combine all nodes: 17 body keypoints + 2 head nodes + 1 status node
        status_node = np.concatenate([
            occupant_status.reshape(self.sequence_length, 1, 1),
            np.zeros((self.sequence_length, 1, 2))  # Pad to match dimension
        ], axis=2)
        
        # Final graph: (T, 20, 3) - 17 body + 2 head + 1 status
        graph_data = np.concatenate([keypoints, head_nodes, status_node], axis=1)
        
        # Convert to tensor format: (C, T, V, M) where M=1 (single person)
        graph_data = torch.from_numpy(graph_data).float()
        graph_data = graph_data.permute(2, 0, 1).unsqueeze(-1)  # (3, T, 20, 1)
        
        # Target (occupant status for next frame or classification)
        target = torch.from_numpy(occupant_status[-1:]).float()
        
        return {
            'data': graph_data,
            'target': target,
            'keypoints': torch.from_numpy(keypoints).float(),
            'head_pose': torch.from_numpy(head_pose).float(),
            'occupant_status': torch.from_numpy(occupant_status).float()
        }


class DataAugmentation:
    """Data augmentation for occupant monitoring data"""
    
    def __init__(self, noise_std=0.01, rotation_range=0.1, scale_range=0.1):
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.scale_range = scale_range
    
    def __call__(self, sample):
        data = sample['data'].clone()
        
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(data) * self.noise_std
            data += noise
        
        # Random scaling
        if self.scale_range > 0:
            scale = 1 + (torch.rand(1) - 0.5) * 2 * self.scale_range
            data[:2] *= scale  # Only scale x, y coordinates
        
        # Random rotation (only for keypoints)
        if self.rotation_range > 0:
            angle = (torch.rand(1) - 0.5) * 2 * self.rotation_range
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Apply rotation to x, y coordinates of keypoints (first 17 nodes)
            xy_coords = data[:2, :, :17, :]  # (2, T, 17, 1)
            xy_coords_flat = xy_coords.reshape(2, -1)  # (2, T*17*1)
            rotated_coords = torch.mm(rotation_matrix, xy_coords_flat)
            data[:2, :, :17, :] = rotated_coords.reshape(2, data.shape[1], 17, 1)
        
        sample['data'] = data
        return sample


def create_sample_data(output_path: str, num_samples: int = 1000, sequence_length: int = 64):
    """Create sample dataset for testing"""
    samples = []
    
    for i in range(num_samples):
        # Generate random keypoints (17 joints, 3 coordinates each)
        keypoints = np.random.randn(sequence_length, 17, 3)
        keypoints[:, :, 2] = np.random.uniform(0.5, 1.0, (sequence_length, 17))  # Confidence scores
        
        # Generate head pose (position + rotation)
        head_pose = np.random.randn(sequence_length, 6)
        
        # Generate occupant status (0-1 range)
        occupant_status = np.random.uniform(0, 1, sequence_length)
        
        sample = {
            'id': f'sample_{i:04d}',
            'keypoints': keypoints.tolist(),
            'head_pose': head_pose.tolist(),
            'occupant_status': occupant_status.tolist()
        }
        samples.append(sample)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Created {num_samples} sample data points in {output_path}")


def get_dataloader(data_path: str, batch_size: int = 16, sequence_length: int = 64, 
                  shuffle: bool = True, num_workers: int = 4, augmentation: bool = True) -> DataLoader:
    """Create DataLoader for training/testing"""
    
    transform = DataAugmentation() if augmentation else None
    dataset = OccupantDataset(data_path, sequence_length, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Normalize keypoints to [-1, 1] range"""
    # Assume keypoints are in pixel coordinates
    # This should be adapted based on your actual coordinate system
    normalized = keypoints.copy()
    normalized[:, :, 0] = (keypoints[:, :, 0] - 320) / 320  # Assuming 640px width
    normalized[:, :, 1] = (keypoints[:, :, 1] - 240) / 240  # Assuming 480px height
    return normalized


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("data/sample_data.json", num_samples=100)
    
    # Test dataset loading
    dataset = OccupantDataset("data/sample_data.json", sequence_length=32)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Data shape: {sample['data'].shape}")
    print(f"Target shape: {sample['target'].shape}")
