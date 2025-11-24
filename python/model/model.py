import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution layer
    """
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Register adjacency matrix as buffer (not a parameter)
        self.register_buffer('adj', adjacency_matrix)

        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
        Returns:
            out: (batch_size, num_nodes, out_channels)
        """
        # x: (B, N, C_in)
        # weight: (C_in, C_out)
        # adj: (N, N)

        support = torch.matmul(x, self.weight)  # (B, N, C_out)
        output = torch.matmul(self.adj, support)  # (N, N) x (B, N, C_out) -> (B, N, C_out)
        output = output + self.bias

        return output


class TemporalConvolution(nn.Module):
    """
    Temporal convolution layer using 1D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps)
        Returns:
            out: (batch_size, out_channels, time_steps)
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network Block
    """
    def __init__(self, in_channels, out_channels, adjacency_matrix, temporal_kernel_size=9, dropout=0.3):
        super(STGCNBlock, self).__init__()

        # Temporal convolution 1
        self.temporal1 = TemporalConvolution(in_channels, out_channels, kernel_size=temporal_kernel_size)

        # Spatial graph convolution
        self.spatial = GraphConvolution(out_channels, out_channels, adjacency_matrix)

        # Temporal convolution 2
        self.temporal2 = TemporalConvolution(out_channels, out_channels, kernel_size=temporal_kernel_size)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_nodes, in_channels, time_steps)
        Returns:
            out: (batch_size, num_nodes, out_channels, time_steps)
        """
        batch_size, num_nodes, in_channels, time_steps = x.shape

        # Residual connection
        residual = x

        # Reshape for temporal convolution: (B, N, C, T) -> (B*N, C, T)
        x = x.reshape(batch_size * num_nodes, in_channels, time_steps)

        # Temporal convolution 1
        x = F.relu(self.temporal1(x))

        # Reshape for spatial convolution: (B*N, C, T) -> (B, N, C, T) -> (B, T, N, C)
        x = x.reshape(batch_size, num_nodes, -1, time_steps)
        x = x.permute(0, 3, 1, 2)  # (B, T, N, C)

        # Spatial graph convolution for each time step
        outputs = []
        for t in range(time_steps):
            out_t = self.spatial(x[:, t, :, :])  # (B, N, C)
            outputs.append(out_t)
        x = torch.stack(outputs, dim=1)  # (B, T, N, C)
        x = F.relu(x)

        # Reshape back: (B, T, N, C) -> (B, N, C, T)
        x = x.permute(0, 2, 3, 1)

        # Reshape for temporal convolution: (B, N, C, T) -> (B*N, C, T)
        x = x.reshape(batch_size * num_nodes, -1, time_steps)

        # Temporal convolution 2
        x = self.temporal2(x)

        # Reshape back: (B*N, C, T) -> (B, N, C, T)
        x = x.reshape(batch_size, num_nodes, -1, time_steps)

        # Residual connection
        residual = residual.reshape(batch_size * num_nodes, in_channels, time_steps)
        residual = self.residual(residual)
        residual = residual.reshape(batch_size, num_nodes, -1, time_steps)

        x = x + residual
        x = self.dropout(x)

        return x


class AttentionSTGCN(nn.Module):
    """
    STGCN model for attention prediction from body and face keypoints
    """
    def __init__(self, config, feature_dims=None):
        super(AttentionSTGCN, self).__init__()

        self.config = config
        model_cfg = config['model']

        # Get feature configuration
        feature_config = config['data'].get('features', {})
        self.use_body = feature_config.get('use_body_kps', True)
        self.use_face_2d = feature_config.get('use_face_kps_2d', True)
        self.use_head_pose = feature_config.get('use_head_pose', True)

        # Determine feature dimensions
        if feature_dims is not None:
            # Use actual dimensions from data
            body_features = feature_dims.get('body', 0)
            face_2d_features = feature_dims.get('face_2d', 0)
            head_pose_features = feature_dims.get('head_pose', 0)
        else:
            # Use config defaults (for backward compatibility)
            body_features = model_cfg['num_body_nodes'] * model_cfg['body_in_channels'] if self.use_body else 0
            face_2d_features = model_cfg['num_face_nodes'] * model_cfg['face_2d_in_channels'] if self.use_face_2d else 0
            head_pose_features = model_cfg['head_pose_features'] if self.use_head_pose else 0

        # Calculate number of nodes
        self.num_body_nodes = body_features // model_cfg.get('body_in_channels', 2) if body_features > 0 else 17
        self.num_face_nodes = face_2d_features // model_cfg.get('face_2d_in_channels', 2) if face_2d_features > 0 else 68

        hidden_channels = model_cfg['hidden_channels']

        # Create adjacency matrices for body and face skeletons
        if self.use_body and body_features > 0:
            self.body_adj = self._create_body_adjacency()
        else:
            self.body_adj = None

        if self.use_face_2d and face_2d_features > 0:
            self.face_adj = self._create_face_adjacency()
        else:
            self.face_adj = None

        # Input projection layers
        if self.use_body and body_features > 0:
            self.body_proj = nn.Linear(model_cfg.get('body_in_channels', 2), hidden_channels)
        else:
            self.body_proj = None

        if self.use_face_2d and face_2d_features > 0:
            self.face_2d_proj = nn.Linear(model_cfg.get('face_2d_in_channels', 2), hidden_channels)
        else:
            self.face_2d_proj = None

        # STGCN blocks for body
        if self.use_body and body_features > 0:
            self.body_stgcn_blocks = nn.ModuleList([
                STGCNBlock(
                    hidden_channels,
                    hidden_channels,
                    self.body_adj,
                    temporal_kernel_size=model_cfg['temporal_kernel_size'],
                    dropout=model_cfg['dropout']
                ) for _ in range(model_cfg['num_gcn_layers'])
            ])
        else:
            self.body_stgcn_blocks = None

        # STGCN blocks for face
        if self.use_face_2d and face_2d_features > 0:
            self.face_stgcn_blocks = nn.ModuleList([
                STGCNBlock(
                    hidden_channels,
                    hidden_channels,
                    self.face_adj,
                    temporal_kernel_size=model_cfg['temporal_kernel_size'],
                    dropout=model_cfg['dropout']
                ) for _ in range(model_cfg['num_gcn_layers'])
            ])
        else:
            self.face_stgcn_blocks = None

        # Head pose processing
        if self.use_head_pose and head_pose_features > 0:
            self.head_pose_fc = nn.Sequential(
                nn.Linear(head_pose_features, hidden_channels),
                nn.ReLU(),
                nn.Dropout(model_cfg['dropout'])
            )
        else:
            self.head_pose_fc = None

        # Calculate total features for fusion layer
        total_features = 0
        if self.use_body and body_features > 0:
            total_features += self.num_body_nodes * hidden_channels
        if self.use_face_2d and face_2d_features > 0:
            total_features += self.num_face_nodes * hidden_channels
        if self.use_head_pose and head_pose_features > 0:
            total_features += hidden_channels

        if total_features == 0:
            raise ValueError("At least one feature type must be enabled")

        # Fusion and prediction layers (5-class classification)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(model_cfg['dropout']),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(model_cfg['dropout']),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(model_cfg['dropout']),
            nn.Linear(hidden_channels, 5)  # 5 classes: 1,2,3,4,5
        )

    def _create_body_adjacency(self):
        """
        Create adjacency matrix for body skeleton (COCO format)
        17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        """
        num_nodes = 17
        adj = torch.zeros(num_nodes, num_nodes)

        # Define body skeleton connections
        connections = [
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            (0, 5), (0, 6),  # nose to shoulders
            (5, 6),  # shoulders
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12),  # shoulders to hips
            (11, 12),  # hips
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16)  # right leg
        ]

        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1

        # Add self-loops
        adj = adj + torch.eye(num_nodes)

        # Normalize adjacency matrix
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        adj_normalized = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)

        return adj_normalized

    def _create_face_adjacency(self):
        """
        Create adjacency matrix for face landmarks (68 points)
        Simplified connectivity based on facial structure
        """
        num_nodes = 68
        adj = torch.zeros(num_nodes, num_nodes)

        # Face contour (0-16)
        for i in range(16):
            adj[i, i+1] = 1

        # Eyebrows
        for i in range(17, 21):  # Right eyebrow
            adj[i, i+1] = 1
        for i in range(22, 26):  # Left eyebrow
            adj[i, i+1] = 1

        # Nose
        for i in range(27, 30):  # Nose bridge
            adj[i, i+1] = 1
        for i in range(31, 35):  # Nose bottom
            adj[i, i+1] = 1

        # Eyes
        for i in range(36, 41):  # Right eye
            adj[i, i+1] = 1
        adj[41, 36] = 1  # Close right eye
        for i in range(42, 47):  # Left eye
            adj[i, i+1] = 1
        adj[47, 42] = 1  # Close left eye

        # Mouth
        for i in range(48, 59):  # Outer mouth
            adj[i, i+1] = 1
        adj[59, 48] = 1  # Close outer mouth
        for i in range(60, 67):  # Inner mouth
            adj[i, i+1] = 1
        adj[67, 60] = 1  # Close inner mouth

        # Make symmetric
        adj = adj + adj.T

        # Add self-loops
        adj = adj + torch.eye(num_nodes)

        # Normalize adjacency matrix
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        adj_normalized = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)

        return adj_normalized

    def forward(self, body_kps, face_kps_2d, head_pose):
        """
        Args:
            body_kps: (batch_size, sequence_length, num_body_nodes * 2) or None
            face_kps_2d: (batch_size, sequence_length, num_face_nodes * 2) or None
            head_pose: (batch_size, sequence_length, head_pose_features) or None
        Returns:
            attention: (batch_size, 1)
        """
        features_list = []

        # Process body keypoints
        if self.use_body and self.body_proj is not None and body_kps is not None:
            batch_size, seq_len = body_kps.shape[0], body_kps.shape[1]

            # Reshape body keypoints: (B, T, N*2) -> (B, T, N, 2)
            body_kps = body_kps.reshape(batch_size, seq_len, self.num_body_nodes, 2)

            # Project features
            body_features = self.body_proj(body_kps)  # (B, T, N_body, hidden)

            # Reshape for STGCN: (B, T, N, C) -> (B, N, C, T)
            body_features = body_features.permute(0, 2, 3, 1)

            # Apply STGCN blocks
            for block in self.body_stgcn_blocks:
                body_features = block(body_features)

            # Global pooling over time: (B, N, C, T) -> (B, N, C)
            body_features = body_features.mean(dim=-1)

            # Flatten spatial features: (B, N, C) -> (B, N*C)
            body_features = body_features.reshape(batch_size, -1)
            features_list.append(body_features)
        else:
            batch_size = face_kps_2d.shape[0] if face_kps_2d is not None else head_pose.shape[0]
            seq_len = face_kps_2d.shape[1] if face_kps_2d is not None else head_pose.shape[1]

        # Process face keypoints (2D)
        if self.use_face_2d and self.face_2d_proj is not None and face_kps_2d is not None:
            # Reshape face keypoints
            face_kps_2d = face_kps_2d.reshape(batch_size, seq_len, self.num_face_nodes, 2)

            # Project features
            face_features = self.face_2d_proj(face_kps_2d)  # (B, T, N_face, hidden)

            # Reshape for STGCN: (B, T, N, C) -> (B, N, C, T)
            face_features = face_features.permute(0, 2, 3, 1)

            # Apply STGCN blocks
            for block in self.face_stgcn_blocks:
                face_features = block(face_features)

            # Global pooling over time: (B, N, C, T) -> (B, N, C)
            face_features = face_features.mean(dim=-1)

            # Flatten spatial features: (B, N, C) -> (B, N*C)
            face_features = face_features.reshape(batch_size, -1)
            features_list.append(face_features)

        # Process head pose
        if self.use_head_pose and self.head_pose_fc is not None and head_pose is not None:
            # (B, T, head_pose_features) -> (B, head_pose_features)
            head_pose_avg = head_pose.mean(dim=1)
            head_pose_features = self.head_pose_fc(head_pose_avg)  # (B, hidden)
            features_list.append(head_pose_features)

        # Concatenate all features
        if len(features_list) == 0:
            raise ValueError("No features available for prediction")

        combined = torch.cat(features_list, dim=-1)

        # Predict attention (logits for 5 classes)
        attention_logits = self.fusion(combined)

        return attention_logits


def load_config(config_path):
    """Load configuration from JSON file"""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    # Test model
    config = load_config('config.json')

    # Create dummy feature dims
    feature_dims = {
        'body': 17 * 2,
        'face_2d': 68 * 2,
        'head_pose': 9
    }

    model = AttentionSTGCN(config, feature_dims)

    batch_size = 4
    seq_len = config['training']['sequence_length']

    # Create dummy input (no 3D face landmarks)
    body_kps = torch.randn(batch_size, seq_len, 17 * 2)
    face_kps_2d = torch.randn(batch_size, seq_len, 68 * 2)
    head_pose = torch.randn(batch_size, seq_len, 9)

    # Forward pass
    output = model(body_kps, face_kps_2d, head_pose)

    print(f"Input shapes:")
    print(f"  Body keypoints: {body_kps.shape}")
    print(f"  Face keypoints 2D: {face_kps_2d.shape}")
    print(f"  Head pose: {head_pose.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
