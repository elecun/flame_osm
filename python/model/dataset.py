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
    def __init__(self, csv_files, sequence_length=30, normalize=True, train=True, scaler=None, feature_config=None):
        """
        Args:
            csv_files: List of CSV file paths or single CSV file path
            sequence_length: Length of temporal sequences
            normalize: Whether to normalize features
            train: Whether this is training data (for scaler fitting)
            scaler: Pre-fitted scaler (for validation/test data)
            feature_config: Dictionary with feature selection configuration
        """
        self.sequence_length = sequence_length
        self.normalize = normalize

        # Handle single file or list of files
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        # Default feature config
        if feature_config is None:
            feature_config = {
                'use_body_kps': True,
                'use_face_kps_2d': True,
                'use_head_pose': True,
                'body_pattern': ['nose_', 'eye_', 'ear_', 'shoulder_', 'elbow_', 'wrist_', 'hip_', 'knee_', 'ankle_'],
                'face_2d_pattern': 'landmark_.*_2d',
                'head_pose_cols': ['head_rotation_x', 'head_rotation_y', 'head_rotation_z',
                                   'head_translation_x', 'head_translation_y', 'head_translation_z',
                                   'head_pitch', 'head_yaw', 'head_roll']
            }

        self.feature_config = feature_config

        # Load and concatenate all CSV files
        df_list = []
        file_boundaries = [0]  # Track where each file starts/ends

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df_list.append(df)
            file_boundaries.append(file_boundaries[-1] + len(df))

        combined_df = pd.concat(df_list, ignore_index=True)
        self.file_boundaries = file_boundaries

        # Extract column indices based on feature config
        self.use_body = feature_config['use_body_kps']
        self.use_face_2d = feature_config['use_face_kps_2d']
        self.use_head_pose = feature_config['use_head_pose']

        if self.use_body:
            self.body_cols = [col for col in combined_df.columns if any(
                joint in col for joint in feature_config['body_pattern']
            )]
        else:
            self.body_cols = []

        if self.use_face_2d:
            import re
            pattern = re.compile(feature_config['face_2d_pattern'])
            self.face_2d_cols = [col for col in combined_df.columns if pattern.match(col)]
        else:
            self.face_2d_cols = []

        if self.use_head_pose:
            self.head_pose_cols = feature_config['head_pose_cols']
        else:
            self.head_pose_cols = []

        # Extract features
        if self.use_body:
            self.body_data = combined_df[self.body_cols].values.astype(np.float32)
        else:
            self.body_data = np.zeros((len(combined_df), 0), dtype=np.float32)

        if self.use_face_2d:
            self.face_2d_data = combined_df[self.face_2d_cols].values.astype(np.float32)
        else:
            self.face_2d_data = np.zeros((len(combined_df), 0), dtype=np.float32)

        if self.use_head_pose:
            self.head_pose_data = combined_df[self.head_pose_cols].values.astype(np.float32)
        else:
            self.head_pose_data = np.zeros((len(combined_df), 0), dtype=np.float32)

        # Extract target (attention)
        self.attention = combined_df['attention'].values.astype(np.float32)

        # Normalize features
        if normalize:
            if train:
                # Fit scaler on training data
                if self.use_body and self.body_data.shape[1] > 0:
                    self.body_scaler = StandardScaler()
                    self.body_data = self.body_scaler.fit_transform(self.body_data)
                else:
                    self.body_scaler = None

                if self.use_face_2d and self.face_2d_data.shape[1] > 0:
                    self.face_2d_scaler = StandardScaler()
                    self.face_2d_data = self.face_2d_scaler.fit_transform(self.face_2d_data)
                else:
                    self.face_2d_scaler = None

                if self.use_head_pose and self.head_pose_data.shape[1] > 0:
                    self.head_pose_scaler = StandardScaler()
                    self.head_pose_data = self.head_pose_scaler.fit_transform(self.head_pose_data)
                else:
                    self.head_pose_scaler = None
            else:
                # Use pre-fitted scaler
                if scaler is None:
                    raise ValueError("Scaler must be provided for validation/test data")

                if self.use_body and scaler.get('body') is not None:
                    self.body_scaler = scaler['body']
                    self.body_data = self.body_scaler.transform(self.body_data)
                else:
                    self.body_scaler = None

                if self.use_face_2d and scaler.get('face_2d') is not None:
                    self.face_2d_scaler = scaler['face_2d']
                    self.face_2d_data = self.face_2d_scaler.transform(self.face_2d_data)
                else:
                    self.face_2d_scaler = None

                if self.use_head_pose and scaler.get('head_pose') is not None:
                    self.head_pose_scaler = scaler['head_pose']
                    self.head_pose_data = self.head_pose_scaler.transform(self.head_pose_data)
                else:
                    self.head_pose_scaler = None

        # Create sequences and filter out invalid ones (with -1 values)
        # Also ensure sequences don't cross file boundaries
        self.sequences = []

        # Get original (non-normalized) face_2d data to check for -1 values
        if self.use_face_2d and len(self.face_2d_cols) > 0:
            original_face_2d = combined_df[self.face_2d_cols].values.astype(np.float32)
        else:
            original_face_2d = None

        skipped_count = 0
        skipped_boundary = 0

        for file_idx in range(len(self.file_boundaries) - 1):
            file_start = self.file_boundaries[file_idx]
            file_end = self.file_boundaries[file_idx + 1]

            # Create sequences within this file
            num_samples = file_end - file_start - sequence_length + 1

            for i in range(num_samples):
                abs_idx = file_start + i

                # Check if sequence would cross file boundary
                if abs_idx + sequence_length > file_end:
                    skipped_boundary += 1
                    continue

                # Check if sequence contains -1 values in face landmarks
                if original_face_2d is not None:
                    seq_face_data = original_face_2d[abs_idx:abs_idx + sequence_length]
                    if np.any(seq_face_data == -1.0):
                        skipped_count += 1
                        continue

                self.sequences.append(abs_idx)

        print(f"Loaded {len(csv_files)} CSV file(s) with {len(combined_df)} total frames")
        print(f"Created dataset with {len(self.sequences)} valid sequences")
        print(f"  Skipped {skipped_count} sequences with invalid face detection")
        print(f"  Skipped {skipped_boundary} sequences crossing file boundaries")
        if self.use_body:
            print(f"  Body features: {self.body_data.shape[1]}")
        if self.use_face_2d:
            print(f"  Face 2D features: {self.face_2d_data.shape[1]}")
        if self.use_head_pose:
            print(f"  Head pose features: {self.head_pose_data.shape[1]}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        end_idx = start_idx + self.sequence_length

        # Get sequence data (no 3D face landmarks)
        body_seq = torch.from_numpy(self.body_data[start_idx:end_idx])
        face_2d_seq = torch.from_numpy(self.face_2d_data[start_idx:end_idx])
        head_pose_seq = torch.from_numpy(self.head_pose_data[start_idx:end_idx])

        # Get target (use last frame's attention value)
        attention = torch.tensor(self.attention[end_idx - 1], dtype=torch.float32)

        return {
            'body_kps': body_seq,
            'face_kps_2d': face_2d_seq,
            'head_pose': head_pose_seq,
            'attention': attention
        }

    def get_scaler(self):
        """Return fitted scalers"""
        if self.normalize:
            scaler_dict = {}
            if self.use_body and self.body_scaler is not None:
                scaler_dict['body'] = self.body_scaler
            if self.use_face_2d and self.face_2d_scaler is not None:
                scaler_dict['face_2d'] = self.face_2d_scaler
            if self.use_head_pose and self.head_pose_scaler is not None:
                scaler_dict['head_pose'] = self.head_pose_scaler
            return scaler_dict
        return None

    def get_feature_dims(self):
        """Return feature dimensions for model initialization"""
        dims = {}
        if self.use_body:
            dims['body'] = self.body_data.shape[1]
        if self.use_face_2d:
            dims['face_2d'] = self.face_2d_data.shape[1]
        if self.use_head_pose:
            dims['head_pose'] = self.head_pose_data.shape[1]
        return dims


def create_data_loaders(config, csv_path):
    """
    Create train, validation, and test data loaders

    Args:
        config: Configuration dictionary
        csv_path: Path to CSV file or directory containing CSV files
    """
    import os
    import glob

    # Get CSV files
    if config['data'].get('is_directory', False):
        # Load all CSV files from directory
        csv_files = sorted(glob.glob(os.path.join(csv_path, '*.csv')))
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in directory: {csv_path}")
        print(f"\nFound {len(csv_files)} CSV file(s) in directory")
    else:
        # Single CSV file
        csv_files = [csv_path]

    # Load all dataframes to determine split sizes
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)

    total_samples = sum(len(df) for df in df_list)

    # Calculate split indices
    train_split = config['training']['train_split']
    val_split = config['training']['validation_split']

    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)

    # Split files into train/val/test
    train_files = []
    val_files = []
    test_files = []

    current_samples = 0
    for csv_file, df in zip(csv_files, df_list):
        file_samples = len(df)

        if current_samples < train_size:
            # File belongs to training set
            if current_samples + file_samples <= train_size:
                train_files.append(csv_file)
            else:
                # File spans train and val/test - need to split it
                # For simplicity, create temporary files
                split_idx = train_size - current_samples
                train_df = df[:split_idx]
                remaining_df = df[split_idx:]

                temp_dir = os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.'
                temp_train = os.path.join(temp_dir, f'temp_train_{os.path.basename(csv_file)}')
                temp_remain = os.path.join(temp_dir, f'temp_remain_{os.path.basename(csv_file)}')

                train_df.to_csv(temp_train, index=False)
                remaining_df.to_csv(temp_remain, index=False)

                train_files.append(temp_train)

                # Process remaining part
                if current_samples + split_idx < train_size + val_size:
                    if current_samples + file_samples <= train_size + val_size:
                        val_files.append(temp_remain)
                    else:
                        # Remaining spans val and test
                        val_split_idx = train_size + val_size - current_samples - split_idx
                        val_df = remaining_df[:val_split_idx]
                        test_df = remaining_df[val_split_idx:]

                        temp_val = os.path.join(temp_dir, f'temp_val_{os.path.basename(csv_file)}')
                        temp_test = os.path.join(temp_dir, f'temp_test_{os.path.basename(csv_file)}')

                        val_df.to_csv(temp_val, index=False)
                        test_df.to_csv(temp_test, index=False)

                        val_files.append(temp_val)
                        test_files.append(temp_test)
                else:
                    test_files.append(temp_remain)
        elif current_samples < train_size + val_size:
            # File belongs to validation set
            if current_samples + file_samples <= train_size + val_size:
                val_files.append(csv_file)
            else:
                # File spans val and test
                split_idx = train_size + val_size - current_samples
                val_df = df[:split_idx]
                test_df = df[split_idx:]

                temp_dir = os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.'
                temp_val = os.path.join(temp_dir, f'temp_val_{os.path.basename(csv_file)}')
                temp_test = os.path.join(temp_dir, f'temp_test_{os.path.basename(csv_file)}')

                val_df.to_csv(temp_val, index=False)
                test_df.to_csv(temp_test, index=False)

                val_files.append(temp_val)
                test_files.append(temp_test)
        else:
            # File belongs to test set
            test_files.append(csv_file)

        current_samples += file_samples

    # Feature configuration
    feature_config = config['data'].get('features', None)

    # Create datasets
    sequence_length = config['training']['sequence_length']
    normalize = config['data']['normalize']

    train_dataset = AttentionDataset(
        train_files,
        sequence_length=sequence_length,
        normalize=normalize,
        train=True,
        feature_config=feature_config
    )

    scaler = train_dataset.get_scaler()
    feature_dims = train_dataset.get_feature_dims()

    val_dataset = AttentionDataset(
        val_files,
        sequence_length=sequence_length,
        normalize=normalize,
        train=False,
        scaler=scaler,
        feature_config=feature_config
    )

    test_dataset = AttentionDataset(
        test_files,
        sequence_length=sequence_length,
        normalize=normalize,
        train=False,
        scaler=scaler,
        feature_config=feature_config
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
    print(f"  Train: {len(train_dataset)} sequences from {len(train_files)} file(s)")
    print(f"  Validation: {len(val_dataset)} sequences from {len(val_files)} file(s)")
    print(f"  Test: {len(test_dataset)} sequences from {len(test_files)} file(s)")

    # Clean up temporary files
    for f in train_files + val_files + test_files:
        if 'temp_' in os.path.basename(f):
            try:
                os.remove(f)
            except:
                pass

    return train_loader, val_loader, test_loader, scaler, feature_dims


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
        print(f"  Head pose: {batch['head_pose'].shape}")
        print(f"  Attention: {batch['attention'].shape}")
        print(f"  Attention range: [{batch['attention'].min():.3f}, {batch['attention'].max():.3f}]")
        break
