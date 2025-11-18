import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json


class AttentionDataset(Dataset):
    """
    Dataset for attention prediction from body/face keypoints and head pose
    """
    def __init__(self, csv_file, sequence_length=30, normalize=True, train=True, scaler=None):
        """
        Args:
            csv_file: Path to CSV file
            sequence_length: Length of temporal sequences
            normalize: Whether to normalize features
            train: Whether this is training data (for scaler fitting)
            scaler: Pre-fitted scaler (for validation/test data)
        """
        self.sequence_length = sequence_length
        self.normalize = normalize

        # Load data
        df = pd.read_csv(csv_file)

        # Extract column indices
        self.body_cols = [col for col in df.columns if any(
            joint in col for joint in ['nose_', 'eye_', 'ear_', 'shoulder_', 'elbow_', 'wrist_', 'hip_', 'knee_', 'ankle_']
        )]
        self.face_2d_cols = [col for col in df.columns if 'landmark_' in col and '_2d' in col]
        self.face_3d_cols = [col for col in df.columns if 'landmark_' in col and '_3d' in col]
        self.head_pose_cols = ['rotation_x', 'rotation_y', 'rotation_z',
                                'translation_x', 'translation_y', 'translation_z',
                                'pitch', 'yaw', 'roll']

        # Extract features
        self.body_data = df[self.body_cols].values.astype(np.float32)
        self.face_2d_data = df[self.face_2d_cols].values.astype(np.float32)
        self.face_3d_data = df[self.face_3d_cols].values.astype(np.float32)
        self.head_pose_data = df[self.head_pose_cols].values.astype(np.float32)

        # Extract target (attention)
        self.attention = df['attention'].values.astype(np.float32)

        # Normalize features
        if normalize:
            if train:
                # Fit scaler on training data
                self.body_scaler = StandardScaler()
                self.face_2d_scaler = StandardScaler()
                self.face_3d_scaler = StandardScaler()
                self.head_pose_scaler = StandardScaler()

                self.body_data = self.body_scaler.fit_transform(self.body_data)
                self.face_2d_data = self.face_2d_scaler.fit_transform(self.face_2d_data)
                self.face_3d_data = self.face_3d_scaler.fit_transform(self.face_3d_data)
                self.head_pose_data = self.head_pose_scaler.fit_transform(self.head_pose_data)
            else:
                # Use pre-fitted scaler
                if scaler is None:
                    raise ValueError("Scaler must be provided for validation/test data")

                self.body_scaler = scaler['body']
                self.face_2d_scaler = scaler['face_2d']
                self.face_3d_scaler = scaler['face_3d']
                self.head_pose_scaler = scaler['head_pose']

                self.body_data = self.body_scaler.transform(self.body_data)
                self.face_2d_data = self.face_2d_scaler.transform(self.face_2d_data)
                self.face_3d_data = self.face_3d_scaler.transform(self.face_3d_data)
                self.head_pose_data = self.head_pose_scaler.transform(self.head_pose_data)

        # Create sequences
        self.sequences = []
        num_samples = len(df) - sequence_length + 1

        for i in range(num_samples):
            self.sequences.append(i)

        print(f"Created dataset with {len(self.sequences)} sequences")
        print(f"  Body features: {self.body_data.shape[1]}")
        print(f"  Face 2D features: {self.face_2d_data.shape[1]}")
        print(f"  Face 3D features: {self.face_3d_data.shape[1]}")
        print(f"  Head pose features: {self.head_pose_data.shape[1]}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        end_idx = start_idx + self.sequence_length

        # Get sequence data
        body_seq = torch.from_numpy(self.body_data[start_idx:end_idx])
        face_2d_seq = torch.from_numpy(self.face_2d_data[start_idx:end_idx])
        face_3d_seq = torch.from_numpy(self.face_3d_data[start_idx:end_idx])
        head_pose_seq = torch.from_numpy(self.head_pose_data[start_idx:end_idx])

        # Get target (use last frame's attention value)
        attention = torch.tensor(self.attention[end_idx - 1], dtype=torch.float32)

        return {
            'body_kps': body_seq,
            'face_kps_2d': face_2d_seq,
            'face_kps_3d': face_3d_seq,
            'head_pose': head_pose_seq,
            'attention': attention
        }

    def get_scaler(self):
        """Return fitted scalers"""
        if self.normalize:
            return {
                'body': self.body_scaler,
                'face_2d': self.face_2d_scaler,
                'face_3d': self.face_3d_scaler,
                'head_pose': self.head_pose_scaler
            }
        return None


def create_data_loaders(config, csv_file):
    """
    Create train, validation, and test data loaders
    """
    # Load full dataset to split
    df = pd.read_csv(csv_file)
    total_samples = len(df)

    # Calculate split indices
    train_split = config['training']['train_split']
    val_split = config['training']['validation_split']
    test_split = 1.0 - train_split - val_split

    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)

    # Create temporary CSV files for each split
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Save temporary files
    import os
    data_dir = os.path.dirname(csv_file)
    train_file = os.path.join(data_dir, 'train_temp.csv')
    val_file = os.path.join(data_dir, 'val_temp.csv')
    test_file = os.path.join(data_dir, 'test_temp.csv')

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Create datasets
    sequence_length = config['training']['sequence_length']
    normalize = config['data']['normalize']

    train_dataset = AttentionDataset(
        train_file,
        sequence_length=sequence_length,
        normalize=normalize,
        train=True
    )

    scaler = train_dataset.get_scaler()

    val_dataset = AttentionDataset(
        val_file,
        sequence_length=sequence_length,
        normalize=normalize,
        train=False,
        scaler=scaler
    )

    test_dataset = AttentionDataset(
        test_file,
        sequence_length=sequence_length,
        normalize=normalize,
        train=False,
        scaler=scaler
    )

    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config['data']['shuffle'],
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")

    # Clean up temporary files
    os.remove(train_file)
    os.remove(val_file)
    os.remove(test_file)

    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    # Test dataset
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    csv_file = 'merge_0.csv'

    train_loader, val_loader, test_loader, scaler = create_data_loaders(config, csv_file)

    # Test one batch
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"  Body keypoints: {batch['body_kps'].shape}")
        print(f"  Face keypoints 2D: {batch['face_kps_2d'].shape}")
        print(f"  Face keypoints 3D: {batch['face_kps_3d'].shape}")
        print(f"  Head pose: {batch['head_pose'].shape}")
        print(f"  Attention: {batch['attention'].shape}")
        print(f"  Attention range: [{batch['attention'].min():.3f}, {batch['attention'].max():.3f}]")
        break
