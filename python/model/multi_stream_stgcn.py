"""
Multi-Stream STGCN Model
Allows users to define custom streams based on CSV column headers
Each stream is processed independently and then fused for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_model import STGCNBlock, create_body_adjacency, create_face_adjacency, create_generic_adjacency


# Define default stream configurations
# Users can customize these by specifying column patterns
DEFAULT_STREAM_CONFIGS = {
    'face_landmarks': {
        'enabled': True,
        'column_pattern': 'landmark_.*_2d',  # Regex pattern for face landmark columns
        'num_nodes': 68,
        'in_channels': 2,  # x, y coordinates
        'adjacency_type': 'face',  # Use face-specific adjacency matrix
        'description': 'Face landmarks (68 points, 2D coordinates)'
    },
    'head_pose': {
        'enabled': True,
        'column_pattern': ['head_rotation_x', 'head_rotation_y', 'head_rotation_z',
                          'head_translation_x', 'head_translation_y', 'head_translation_z',
                          'head_pitch', 'head_yaw', 'head_roll'],
        'num_nodes': 9,  # Treat each head pose parameter as a node
        'in_channels': 1,
        'adjacency_type': 'sequential',  # Sequential connections
        'description': 'Head pose (rotation, translation, pitch/yaw/roll)'
    },
    'body_pose': {
        'enabled': True,
        'column_pattern': [
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
            'left_hip_',
            'right_hip_',
            'left_knee_',
            'right_knee_',
            'left_ankle_',
            'right_ankle_',
        ],
        'num_nodes': 17,  # COCO body keypoints
        'in_channels': 2,  # x, y coordinates
        'adjacency_type': 'body',  # Use body-specific adjacency matrix
        'description': 'Body keypoints (17 COCO keypoints, 2D coordinates)'
    }
}


class StreamProcessor(nn.Module):
    """
    Processes a single stream of data using STGCN blocks
    """
    def __init__(self, stream_config, hidden_channels=64, num_layers=3,
                 temporal_kernel_size=9, dropout=0.3):
        super(StreamProcessor, self).__init__()

        self.stream_config = stream_config
        self.num_nodes = stream_config['num_nodes']
        self.in_channels = stream_config['in_channels']

        # Create adjacency matrix based on stream type
        adj_type = stream_config['adjacency_type']
        if adj_type == 'body':
            self.adjacency = create_body_adjacency()
        elif adj_type == 'face':
            self.adjacency = create_face_adjacency()
        elif adj_type in ['sequential', 'fully_connected', 'star']:
            self.adjacency = create_generic_adjacency(self.num_nodes, adj_type)
        else:
            raise ValueError(f"Unknown adjacency type: {adj_type}")

        # Input projection
        self.input_proj = nn.Linear(self.in_channels, hidden_channels)

        # STGCN blocks
        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(
                hidden_channels,
                hidden_channels,
                self.adjacency,
                temporal_kernel_size=temporal_kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_features)
               where num_features = num_nodes * in_channels
        Returns:
            features: (batch_size, num_nodes * hidden_channels)
        """
        batch_size, seq_len, num_features = x.shape

        # Reshape: (B, T, num_nodes * in_channels) -> (B, T, num_nodes, in_channels)
        x = x.reshape(batch_size, seq_len, self.num_nodes, self.in_channels)

        # Project input features
        x = self.input_proj(x)  # (B, T, num_nodes, hidden_channels)

        # Reshape for STGCN: (B, T, num_nodes, C) -> (B, num_nodes, C, T)
        x = x.permute(0, 2, 3, 1)

        # Apply STGCN blocks
        for block in self.stgcn_blocks:
            x = block(x)

        # Global pooling over time: (B, num_nodes, C, T) -> (B, num_nodes, C)
        x = x.mean(dim=-1)

        # Flatten: (B, num_nodes, C) -> (B, num_nodes * C)
        x = x.reshape(batch_size, -1)

        return x


class MultiStreamSTGCN(nn.Module):
    """
    Multi-Stream STGCN for attention prediction

    Usage:
        # Use default configuration (3 streams: face, head, body)
        model = MultiStreamSTGCN(config)

        # Or customize streams
        custom_streams = {
            'face_landmarks': DEFAULT_STREAM_CONFIGS['face_landmarks'],
            'body_pose': DEFAULT_STREAM_CONFIGS['body_pose']
        }
        model = MultiStreamSTGCN(config, stream_configs=custom_streams)
    """
    def __init__(self, config, feature_dims=None, stream_configs=None):
        super(MultiStreamSTGCN, self).__init__()

        self.config = config
        model_cfg = config['model']

        # Use default streams if not provided
        if stream_configs is None:
            stream_configs = DEFAULT_STREAM_CONFIGS

        self.stream_configs = stream_configs
        self.stream_names = list(stream_configs.keys())

        # Model hyperparameters
        hidden_channels = model_cfg.get('hidden_channels', 64)
        num_layers = model_cfg.get('num_gcn_layers', 3)
        temporal_kernel_size = model_cfg.get('temporal_kernel_size', 9)
        dropout = model_cfg.get('dropout', 0.3)

        # Create stream processors
        self.streams = nn.ModuleDict()
        for stream_name, stream_config in stream_configs.items():
            self.streams[stream_name] = StreamProcessor(
                stream_config,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                temporal_kernel_size=temporal_kernel_size,
                dropout=dropout
            )

        # Calculate total features after stream processing
        total_features = 0
        for stream_name, stream_config in stream_configs.items():
            num_nodes = stream_config['num_nodes']
            total_features += num_nodes * hidden_channels

        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(total_features, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 5)  # 5 classes for attention levels
        )

        # Store feature mapping for data loading
        self._setup_feature_mapping(feature_dims)

    def _setup_feature_mapping(self, feature_dims):
        """
        Setup mapping between input data and streams
        This helps the model know which features belong to which stream
        """
        if feature_dims is None:
            # Use default mapping
            self.feature_mapping = {
                'body_pose': 'body_kps',
                'face_landmarks': 'face_kps_2d',
                'head_pose': 'head_pose'
            }
        else:
            # Create mapping based on feature_dims
            self.feature_mapping = {}
            if 'body' in feature_dims:
                self.feature_mapping['body_pose'] = 'body_kps'
            if 'face_2d' in feature_dims:
                self.feature_mapping['face_landmarks'] = 'face_kps_2d'
            if 'head_pose' in feature_dims:
                self.feature_mapping['head_pose'] = 'head_pose'

    def forward(self, body_kps=None, face_kps_2d=None, head_pose=None):
        """
        Args:
            body_kps: (batch_size, sequence_length, body_features) or None
            face_kps_2d: (batch_size, sequence_length, face_features) or None
            head_pose: (batch_size, sequence_length, head_pose_features) or None
        Returns:
            logits: (batch_size, 5) - logits for 5 attention classes
        """
        # Map inputs to streams
        stream_inputs = {
            'body_pose': body_kps,
            'face_landmarks': face_kps_2d,
            'head_pose': head_pose
        }

        # Process each stream
        stream_features = []
        missing_streams = []
        for stream_name in self.stream_names:
            if stream_name in self.streams and stream_name in stream_inputs:
                stream_input = stream_inputs[stream_name]
                if stream_input is None or stream_input.numel() == 0 or stream_input.shape[-1] == 0:
                    missing_streams.append(stream_name)
                    continue
                features = self.streams[stream_name](stream_input)
                stream_features.append(features)

        if len(stream_features) == 0:
            msg = "No valid input features provided"
            if missing_streams:
                msg += f". Missing/empty streams: {', '.join(missing_streams)}"
            raise ValueError(msg)

        # Concatenate all stream features
        combined_features = torch.cat(stream_features, dim=-1)

        # Fusion and classification
        logits = self.fusion(combined_features)

        return logits

    def get_stream_info(self):
        """
        Get information about configured streams
        Returns a human-readable description of the model architecture
        """
        info = []
        info.append("Multi-Stream STGCN Configuration:")
        info.append("=" * 60)
        for stream_name, stream_config in self.stream_configs.items():
            info.append(f"\nStream: {stream_name}")
            info.append(f"  Description: {stream_config['description']}")
            info.append(f"  Nodes: {stream_config['num_nodes']}")
            info.append(f"  Input channels: {stream_config['in_channels']}")
            info.append(f"  Adjacency type: {stream_config['adjacency_type']}")
        info.append("\n" + "=" * 60)
        return "\n".join(info)


def create_multi_stream_model(config, feature_dims=None, stream_configs=None):
    """
    Factory function to create a MultiStreamSTGCN model

    Args:
        config: Configuration dictionary
        feature_dims: Dictionary with actual feature dimensions from data
        stream_configs: Custom stream configurations (optional)

    Returns:
        MultiStreamSTGCN model instance
    """
    model = MultiStreamSTGCN(config, feature_dims, stream_configs)
    print(model.get_stream_info())
    return model


if __name__ == '__main__':
    # Test the multi-stream model
    import json

    # Load config
    config = {
        'model': {
            'hidden_channels': 64,
            'num_gcn_layers': 3,
            'temporal_kernel_size': 9,
            'dropout': 0.3
        },
        'training': {
            'sequence_length': 30
        }
    }

    # Create model with default streams
    print("Creating Multi-Stream STGCN with default configuration...")
    model = create_multi_stream_model(config)

    # Test forward pass
    batch_size = 4
    seq_len = config['training']['sequence_length']

    body_kps = torch.randn(batch_size, seq_len, 17 * 2)
    face_kps_2d = torch.randn(batch_size, seq_len, 68 * 2)
    head_pose = torch.randn(batch_size, seq_len, 9)

    output = model(body_kps, face_kps_2d, head_pose)

    print(f"\nInput shapes:")
    print(f"  Body keypoints: {body_kps.shape}")
    print(f"  Face keypoints 2D: {face_kps_2d.shape}")
    print(f"  Head pose: {head_pose.shape}")
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with custom streams (only face and body)
    print("\n\n" + "=" * 60)
    print("Creating Multi-Stream STGCN with custom streams (face + body only)...")
    custom_streams = {
        'face_landmarks': DEFAULT_STREAM_CONFIGS['face_landmarks'],
        'body_pose': DEFAULT_STREAM_CONFIGS['body_pose']
    }
    model_custom = create_multi_stream_model(config, stream_configs=custom_streams)

    output_custom = model_custom(body_kps, face_kps_2d, None)
    print(f"\nOutput shape: {output_custom.shape}")

    total_params_custom = sum(p.numel() for p in model_custom.parameters())
    print(f"Total parameters: {total_params_custom:,}")
