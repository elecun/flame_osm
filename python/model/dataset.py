import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class AttentionDataset(Dataset):
    """
    Dataset for attention prediction from body/face keypoints and head pose
    """
    def __init__(self, csv_files, sequence_length=30, normalize=True, train=True, scaler=None,
                 feature_config=None, dataframe=None, num_classes: int = 5, filter_config=None,
                 file_boundaries=None, sequences=None, fit_indices=None, valid_mask_columns=None):
        """
        Args:
            csv_files: List of CSV file paths or single CSV file path (ignored if dataframe is provided)
            sequence_length: Length of temporal sequences
            normalize: Whether to normalize features
            train: Whether this is training data (for scaler fitting)
            scaler: Pre-fitted scaler (for validation/test data)
            feature_config: Dictionary with feature selection configuration
            dataframe: Optional DataFrame to use instead of loading from CSV files
            num_classes: Total number of attention classes (default: 5, expecting labels 1..num_classes)
            filter_config: Optional dict with Gaussian filter settings (enabled, sigma)
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.num_classes = num_classes
        self.filter_config = filter_config or {"enabled": False, "sigma": 1.0}
        self._fit_indices = fit_indices

        # Use dataframe if provided, otherwise load from CSV files
        if dataframe is not None:
            combined_df = dataframe
            if file_boundaries is None:
                file_boundaries = [0, len(dataframe)]
        else:
            # Handle single file or list of files
            if isinstance(csv_files, str):
                csv_files = [csv_files]

            # Load and concatenate all CSV files
            df_list = []
            file_boundaries = [0]  # Track where each file starts/ends

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                stem = Path(csv_file).stem
                if 'valid_mask' in df.columns:
                    df = df.rename(columns={'valid_mask': f'{stem}_valid_mask'})
                if 'attention_level' not in df.columns:
                    print(f"[WARN] Skipping file without attention_level column: {csv_file}")
                    continue
                df_list.append(df)
                file_boundaries.append(file_boundaries[-1] + len(df))

            if len(df_list) == 0:
                raise ValueError("No CSV files with attention_level column found. Please check input files.")

            combined_df = pd.concat(df_list, ignore_index=True)

        self.file_boundaries = file_boundaries

        # Drop any accidental header-like rows where attention_level is non-numeric
        if 'attention_level' not in combined_df.columns:
            available = list(combined_df.columns)[:10]
            raise ValueError(f"'attention_level' column not found in dataset. "
                             f"Available columns (first 10): {available}")
        numeric_target = pd.to_numeric(combined_df['attention_level'], errors='coerce')
        non_numeric_mask = numeric_target.isna()
        if non_numeric_mask.any():
            dropped = non_numeric_mask.sum()
            combined_df = combined_df.loc[~non_numeric_mask].reset_index(drop=True)
            numeric_target = numeric_target.loc[~non_numeric_mask].reset_index(drop=True)
            print(f"[INFO] Dropped {dropped} row(s) with non-numeric attention_level (likely header rows).")

        # Default feature config
        if feature_config is None:
            feature_config = {
                'use_body_kps': True,
                'use_face_kps_2d': True,
                'use_head_pose': True,
                'body_pattern': [
                    'nose_',
                    'left_eye_',
                    'right_eye_',
                    'left_ear_',
                    'right_ear_',
                    'left_shoulder_',
                    'right_shoulder_',
                    'left_elbow_',
                    'right_elbow_',
                    'left_wrist_',
                    'right_wrist_',
                ],
                'face_2d_pattern': 'landmark_.*_2d',
                'head_pose_cols': ['head_rotation_x', 'head_rotation_y', 'head_rotation_z',
                                   'head_translation_x', 'head_translation_y', 'head_translation_z',
                                   'head_pitch', 'head_yaw', 'head_roll']
            }

        self.feature_config = feature_config
        self.valid_mask_columns = valid_mask_columns

        # Extract column indices based on feature config
        self.use_body = feature_config['use_body_kps']
        self.use_face_2d = feature_config['use_face_kps_2d']
        self.use_head_pose = feature_config['use_head_pose']

        if self.use_body:
            self.body_cols = [col for col in combined_df.columns if any(
                joint in col for joint in feature_config['body_pattern']
            ) and '_aspect_ratio' not in col]
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

        # Validate that required feature columns exist
        if self.use_body and len(self.body_cols) == 0:
            raise ValueError("Body keypoints requested (use_body_kps=True) but no matching columns found "
                             f"for patterns {feature_config['body_pattern']}.")

        if self.use_face_2d and len(self.face_2d_cols) == 0:
            raise ValueError("Face 2D keypoints requested (use_face_kps_2d=True) but no matching columns found "
                             f"for pattern '{feature_config['face_2d_pattern']}'.")

        if self.use_head_pose:
            missing_head = [c for c in feature_config['head_pose_cols'] if c not in combined_df.columns]
            if missing_head:
                raise ValueError(f"Head pose requested (use_head_pose=True) but missing columns: {missing_head}")

        # Replace NaNs in feature columns with 0 to avoid NaN propagation during training
        for cols in [self.body_cols, self.face_2d_cols, self.head_pose_cols]:
            if cols:
                combined_df[cols] = combined_df[cols].fillna(0)

        # Collect per-frame validity mask (AND across specified mask columns, default: all *_valid_mask)
        if valid_mask_columns is None or len(valid_mask_columns) == 0:
            mask_cols = [c for c in combined_df.columns if c.endswith("valid_mask")]
        else:
            mask_cols = [c for c in valid_mask_columns if c in combined_df.columns]
        if mask_cols:
            mask_df = combined_df[mask_cols].fillna(0).clip(lower=0, upper=1)
            mask_arr = mask_df.values.astype(np.float32)
            self.frame_mask = mask_arr.min(axis=1)
            self.has_valid_mask = True
        else:
            self.frame_mask = np.ones(len(combined_df), dtype=np.float32)
            self.has_valid_mask = False

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

        # Optional Gaussian smoothing to reduce white noise before normalization
        if self.filter_config.get("enabled", False):
            sigma = float(self.filter_config.get("sigma", 1.0))
            if sigma > 0:
                if self.use_body and self.body_data.shape[1] > 0:
                    self.body_data = gaussian_filter1d(self.body_data, sigma=sigma, axis=0, mode="nearest")
                if self.use_face_2d and self.face_2d_data.shape[1] > 0:
                    self.face_2d_data = gaussian_filter1d(self.face_2d_data, sigma=sigma, axis=0, mode="nearest")
                if self.use_head_pose and self.head_pose_data.shape[1] > 0:
                    self.head_pose_data = gaussian_filter1d(self.head_pose_data, sigma=sigma, axis=0, mode="nearest")

        # Extract target (attention) - expect labels 1..num_classes, convert to 0-based
        raw_attention = numeric_target.values.astype(np.int64)

        invalid_mask = (raw_attention < 1) | (raw_attention > num_classes)
        if invalid_mask.any():
            bad_vals = np.unique(raw_attention[invalid_mask])
            bad_idx = np.where(invalid_mask)[0][:5]
            raise ValueError(f"Invalid attention_level values found: {bad_vals} at rows {bad_idx} "
                             f"(expected 1..{num_classes}).")
        self.attention = raw_attention - 1

        # Normalize features
        if normalize:
            # Build mask for scaler fitting (only valid frames and optional fit_indices)
            fit_mask = np.ones(len(combined_df), dtype=bool)
            if self.has_valid_mask:
                fit_mask &= self.frame_mask > 0
            if self._fit_indices is not None:
                idx_mask = np.zeros(len(combined_df), dtype=bool)
                idx_mask[self._fit_indices] = True
                fit_mask &= idx_mask

            if train:
                # Fit scaler on training data
                if self.use_body and self.body_data.shape[1] > 0:
                    self.body_scaler = StandardScaler()
                    fit_data = self.body_data[fit_mask] if fit_mask.any() else self.body_data
                    if fit_data.size:
                        self.body_scaler.fit(fit_data)
                        self.body_data = self.body_scaler.transform(self.body_data)
                else:
                    self.body_scaler = None

                if self.use_face_2d and self.face_2d_data.shape[1] > 0:
                    self.face_2d_scaler = StandardScaler()
                    fit_data = self.face_2d_data[fit_mask] if fit_mask.any() else self.face_2d_data
                    if fit_data.size:
                        self.face_2d_scaler.fit(fit_data)
                        self.face_2d_data = self.face_2d_scaler.transform(self.face_2d_data)
                else:
                    self.face_2d_scaler = None

                if self.use_head_pose and self.head_pose_data.shape[1] > 0:
                    self.head_pose_scaler = StandardScaler()
                    fit_data = self.head_pose_data[fit_mask] if fit_mask.any() else self.head_pose_data
                    if fit_data.size:
                        self.head_pose_scaler.fit(fit_data)
                        self.head_pose_data = self.head_pose_scaler.transform(self.head_pose_data)
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

        if sequences is not None:
            self.sequences = list(sequences)
            if dataframe is not None:
                print(f"Loaded DataFrame with {len(combined_df)} total frames")
            else:
                print(f"Loaded {len(csv_files)} CSV file(s) with {len(combined_df)} total frames")
            print(f"Using precomputed sequences: {len(self.sequences)}")
        else:
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

            # Print summary
            if dataframe is not None:
                print(f"Loaded DataFrame with {len(combined_df)} total frames")
            else:
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
        mask_seq = torch.from_numpy(self.frame_mask[start_idx:end_idx])
        sample_mask = mask_seq[-1] if len(mask_seq) > 0 else torch.tensor(0.0, dtype=torch.float32)

        # Get target (use last frame's attention value)
        attention = torch.tensor(self.attention[end_idx - 1], dtype=torch.long)

        return {
            'body_kps': body_seq,
            'face_kps_2d': face_2d_seq,
            'head_pose': head_pose_seq,
            'attention_level': attention,
            'valid_mask_seq': mask_seq,
            'sample_mask': sample_mask
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


def create_data_loaders(config, csv_path, fold_idx=None, rank=None, world_size=None):
    """
    Create train, validation, and test data loaders with k-fold cross validation support

    Args:
        config: Configuration dictionary
    csv_path: Path to CSV file/directory, or list of case directories when using --multi
        fold_idx: Index of current fold (0 to n_folds-1). If None, uses config value.
        rank: Rank of current process for DDP (None for single GPU)
        world_size: Total number of processes for DDP (None for single GPU)
    """
    import os
    import glob
    from collections import Counter

    specified_files = config['data'].get('csv_files', [])
    label_column = config['data'].get('label_column', 'attention_level')

    def load_case_df(case_path):
        # Get CSV files for a single case
        if config['data'].get('is_directory', False):
            if specified_files:
                csv_files = [os.path.join(case_path, f) for f in specified_files]
            else:
                csv_files = sorted(glob.glob(os.path.join(case_path, '*.csv')))
            if len(csv_files) == 0:
                raise ValueError(f"No CSV files found in directory: {case_path}")
        else:
            if specified_files:
                if len(specified_files) > 1:
                    raise ValueError("When is_directory is False, only one CSV file should be listed in data.csv_files")
                csv_files = [specified_files[0]]
            else:
                csv_files = [case_path]

        raw_dfs = []
        ts_list = []
        label_df = None
        label_ts = None

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Sort and drop duplicate timestamps to ensure 1:1 alignment across files
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)

            # Timestamp consistency check within a case
            if 'timestamp' not in df.columns:
                raise ValueError(f"'timestamp' column missing in file: {csv_file}")
            ts_values = df['timestamp'].values
            ts_list.append((csv_file, ts_values))

            if label_column in df.columns:
                if label_df is None:
                    label_df = df[[label_column, 'timestamp']].copy().rename(columns={label_column: 'attention_level'})
                    label_ts = ts_values
                else:
                    if len(df) != len(label_df):
                        raise ValueError(f"Label file length mismatch: {csv_file} has {len(df)} rows, expected {len(label_df)}.")
                    other = df[label_column]
                    if not other.equals(label_df['attention_level']):
                        raise ValueError(f"Multiple label files with differing {label_column} values: {csv_file}")
                df = df.drop(columns=[label_column])

            # Make valid_mask column unique per source to avoid collisions after concat
            if 'valid_mask' in df.columns:
                df = df.rename(columns={'valid_mask': f"{Path(csv_file).stem}_valid_mask"})

            raw_dfs.append((csv_file, df))

        if label_df is None:
            raise ValueError(f"No CSV with '{label_column}' found. Provide at least one file with labels.")

        # Compute common timestamps across all files (including label file)
        common_ts = None
        all_ts_arrays = [ts for _, ts in ts_list]
        if label_ts is not None:
            all_ts_arrays.append(label_ts)
        for ts_values in all_ts_arrays:
            if common_ts is None:
                common_ts = ts_values
            else:
                common_ts = np.intersect1d(common_ts, ts_values)
        if common_ts is None or len(common_ts) == 0:
            raise ValueError(f"No overlapping timestamps across files in case: {case_path}")

        # If any mismatch, filter all dataframes to the common timestamps
        max_len = max(len(ts) for ts in all_ts_arrays)
        if len(common_ts) != max_len:
            print(f"[WARN] Timestamp mismatch detected in case {case_path}. Aligning to {len(common_ts)} common frames.")

        feature_dfs = []
        used_cols = set()
        for csv_file, df in raw_dfs:
            df = df[df['timestamp'].isin(common_ts)].copy()
            df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])

            # Ensure column name uniqueness across files
            new_cols = []
            stem = Path(csv_file).stem
            for col in df.columns:
                if col in used_cols:
                    new_cols.append(f"{stem}_{col}")
                else:
                    new_cols.append(col)
                used_cols.add(new_cols[-1])
            df.columns = new_cols
            feature_dfs.append(df)

        label_df = label_df[label_df['timestamp'].isin(common_ts)].copy()
        label_df = label_df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
        if 'timestamp' in label_df.columns:
            label_df = label_df.drop(columns=['timestamp'])

        lengths = [len(df) for df in feature_dfs] + [len(label_df)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Row count mismatch across files after alignment: {lengths}")

        case_df = pd.concat(feature_dfs, axis=1, copy=False)
        case_df['attention_level'] = label_df['attention_level'].values
        return case_df

    case_paths = csv_path if isinstance(csv_path, (list, tuple)) else [csv_path]
    case_dfs = []
    for case_path in case_paths:
        case_dfs.append(load_case_df(case_path))

    if len(case_dfs) == 0:
        raise ValueError("No case data loaded.")

    case_boundaries = [0]
    for df in case_dfs:
        case_boundaries.append(case_boundaries[-1] + len(df))

    combined_df = pd.concat(case_dfs, ignore_index=True)
    total_samples = len(combined_df)

    # Get k-fold configuration
    n_folds = config['training'].get('n_folds', 5)
    if fold_idx is None:
        fold_idx = config['training'].get('current_fold', 0)

    print(f"\nUsing {n_folds}-fold cross validation, current fold: {fold_idx}")

    # Feature configuration
    feature_config = config['data'].get('features', None)
    filter_config = config['data'].get('filter', {"enabled": False, "sigma": 1.0})
    sequence_length = config['training']['sequence_length']
    normalize = config['data']['normalize']
    num_classes = config['model'].get('num_classes', 5)  # Default to 5 classes
    mask_columns_cfg = config['data'].get('valid_mask_columns', None)

    # Build fold splits (sequence-level stratified when shuffle=True)
    if config['data'].get('shuffle', True):
        # Build sequences over the full data (do not normalize for splitting)
        seq_builder = AttentionDataset(
            csv_files=None,
            sequence_length=sequence_length,
            normalize=False,
            train=False,
            feature_config=feature_config,
            dataframe=combined_df,
            num_classes=num_classes,
            filter_config=filter_config,
            file_boundaries=case_boundaries,
            valid_mask_columns=mask_columns_cfg,
        )
        all_sequences = seq_builder.sequences
        if not all_sequences:
            raise ValueError("No valid sequences available for stratified splitting.")

        seq_labels = np.array(
            [seq_builder.attention[start + sequence_length - 1] for start in all_sequences],
            dtype=np.int64,
        )
        seq_labels_1b = seq_labels + 1

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(np.zeros_like(seq_labels), seq_labels))
        if fold_idx >= len(splits):
            raise ValueError(f"Requested fold_idx {fold_idx} but only {len(splits)} folds available.")
        train_seq_idx, val_seq_idx = splits[fold_idx]
        # Test: use last fold unless it's the current val fold, then use first fold
        test_fold = n_folds - 1
        if test_fold == fold_idx:
            test_fold = 0
        _, test_seq_idx = splits[test_fold]

        train_sequences = [all_sequences[i] for i in train_seq_idx]
        val_sequences = [all_sequences[i] for i in val_seq_idx]
        test_sequences = [all_sequences[i] for i in test_seq_idx]

        train_seq_labels = seq_labels_1b[train_seq_idx]
        val_seq_labels = seq_labels_1b[val_seq_idx]
        test_seq_labels = seq_labels_1b[test_seq_idx]

        # Frame indices used to fit scalers (union of training sequence frames)
        fit_mask = np.zeros(total_samples, dtype=bool)
        for start in train_sequences:
            fit_mask[start:start + sequence_length] = True
        fit_indices = np.where(fit_mask)[0]

        train_df = pd.DataFrame({'attention_level': train_seq_labels})
        val_df = pd.DataFrame({'attention_level': val_seq_labels})
        test_df = pd.DataFrame({'attention_level': test_seq_labels})
    else:
        # Temporal split without shuffling
        fold_size = total_samples // n_folds
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_folds - 1 else total_samples
        val_idx = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, total_samples)])
        test_start = (n_folds - 1) * fold_size
        if fold_idx == n_folds - 1:
            test_idx = np.arange(0, fold_size)
        else:
            test_idx = np.arange(test_start, total_samples)

        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)

        case_boundary_set = set(case_boundaries[1:-1])

        def build_boundaries(indices):
            if len(indices) == 0:
                return [0]
            boundaries = [0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1 or indices[i] in case_boundary_set:
                    boundaries.append(i)
            boundaries.append(len(indices))
            return boundaries

        train_boundaries = build_boundaries(train_idx)
        val_boundaries = build_boundaries(val_idx)
        test_boundaries = build_boundaries(test_idx)

        train_df = combined_df.iloc[train_idx].reset_index(drop=True)
        val_df = combined_df.iloc[val_idx].reset_index(drop=True)
        test_df = combined_df.iloc[test_idx].reset_index(drop=True)
        train_sequences = None
        val_sequences = None
        test_sequences = None
        fit_indices = None

    # Calculate class weights dynamically from the training data
    # Use inverse frequency to give more weight to less frequent classes
    attention_levels = train_df['attention_level'].values
    class_counts = Counter(attention_levels)
    total_train_samples = len(train_df)

    weights = []
    # Classes are 1-based (1, 2, 3, 4, 5) in the CSV
    for i in range(1, num_classes + 1):
        count = class_counts.get(i, 0)
        if count == 0:
            # If a class is not in the training data, its weight is 0.
            weights.append(0.0)
        else:
            # Inverse frequency weighting formula
            weight = total_train_samples / (num_classes * count)
            weights.append(weight)

    class_weights = torch.tensor(weights, dtype=torch.float32)

    def plot_class_distribution(train_df, val_df, test_df, num_classes, fold_idx, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        datasets = {
            "train": train_df['attention_level'],
            "val": val_df['attention_level'],
            "test": test_df['attention_level'],
        }
        classes = list(range(1, num_classes + 1))
        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(8, 4))
        for i, (name, series) in enumerate(datasets.items()):
            counts = [int((series == c).sum()) for c in classes]
            plt.bar(x + i * width, counts, width, label=name)

        plt.xticks(x + width, classes)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(f"Fold {fold_idx} Class Distribution")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"fold_{fold_idx}_class_distribution.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved class distribution plot to {out_path}")

    plot_class_distribution(
        train_df,
        val_df,
        test_df,
        num_classes=config['model'].get('num_classes', 5),
        fold_idx=fold_idx,
        out_dir=config['training'].get('checkpoint_dir', 'checkpoints'),
    )

    # Create datasets using DataFrames directly (preserves original frame indices)
    train_dataset = AttentionDataset(
        csv_files=None,
        sequence_length=sequence_length,
        normalize=normalize,
        train=True,
        feature_config=feature_config,
        dataframe=combined_df if config['data'].get('shuffle', True) else train_df,
        num_classes=num_classes,
        filter_config=filter_config,
        file_boundaries=case_boundaries if config['data'].get('shuffle', True) else train_boundaries,
        sequences=train_sequences,
        fit_indices=fit_indices,
        valid_mask_columns=mask_columns_cfg,
    )

    scaler = train_dataset.get_scaler()
    feature_dims = train_dataset.get_feature_dims()

    val_dataset = AttentionDataset(
        csv_files=None,
        sequence_length=sequence_length,
        normalize=normalize,
        train=False,
        scaler=scaler,
        feature_config=feature_config,
        dataframe=combined_df if config['data'].get('shuffle', True) else val_df,
        num_classes=num_classes,
        filter_config=filter_config,
        file_boundaries=case_boundaries if config['data'].get('shuffle', True) else val_boundaries,
        sequences=val_sequences,
        valid_mask_columns=mask_columns_cfg,
    )

    test_dataset = AttentionDataset(
        csv_files=None,
        sequence_length=sequence_length,
        normalize=normalize,
        train=False,
        scaler=scaler,
        feature_config=feature_config,
        dataframe=combined_df if config['data'].get('shuffle', True) else test_df,
        num_classes=num_classes,
        filter_config=filter_config,
        file_boundaries=case_boundaries if config['data'].get('shuffle', True) else test_boundaries,
        sequences=test_sequences,
        valid_mask_columns=mask_columns_cfg,
    )

    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']

    # Use DistributedSampler if using DDP
    use_ddp = rank is not None and world_size is not None

    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=config['data']['shuffle']
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
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

    print(f"\nDataset splits (Fold {fold_idx}/{n_folds-1}):")
    print(f"  Train: {len(train_dataset)} sequences ({len(train_df)} frames)")
    print(f"  Validation: {len(val_dataset)} sequences ({len(val_df)} frames)")
    print(f"  Test: {len(test_dataset)} sequences ({len(test_df)} frames)")
    print(f"  Total frames: {total_samples}")

    return train_loader, val_loader, test_loader, scaler, feature_dims, class_weights, train_dataset.has_valid_mask


if __name__ == '__main__':
    # Test dataset
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    csv_file = 'merge_0.csv'

    train_loader, val_loader, test_loader, scaler, _, _, _ = create_data_loaders(config, csv_file)

    # Test one batch
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"  Body keypoints: {batch['body_kps'].shape}")
        print(f"  Face keypoints 2D: {batch['face_kps_2d'].shape}")
        print(f"  Head pose: {batch['head_pose'].shape}")
        print(f"  Attention_level: {batch['attention_level'].shape}")
        print(f"  Attention range: [{batch['attention_level'].min():.3f}, {batch['attention_level'].max():.3f}]")
        break
