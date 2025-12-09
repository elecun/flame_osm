"""
Attention-based STGCN with Dynamic Graph Constructor
Automatically learns important nodes and constructs adaptive subgraphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_model import TemporalConvolution, create_body_adjacency, create_face_adjacency


class NodeAttentionModule(nn.Module):
    """
    Learns importance scores for each node (keypoint)
    Uses self-attention mechanism to compute node importance
    """
    def __init__(self, in_channels, hidden_channels=64):
        super(NodeAttentionModule, self).__init__()

        self.query = nn.Linear(in_channels, hidden_channels)
        self.key = nn.Linear(in_channels, hidden_channels)
        self.value = nn.Linear(in_channels, hidden_channels)

        # Node importance score predictor
        self.importance_score = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()  # Importance score between 0 and 1
        )

        self.scale = hidden_channels ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
        Returns:
            importance_scores: (batch_size, num_nodes) - importance score for each node
            attended_features: (batch_size, num_nodes, hidden_channels)
        """
        batch_size, num_nodes, in_channels = x.shape

        # Self-attention
        q = self.query(x)  # (B, N, H)
        k = self.key(x)    # (B, N, H)
        v = self.value(x)  # (B, N, H)

        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        # Attended features
        attended = torch.matmul(attn, v)  # (B, N, H)

        # Compute importance scores for each node
        importance_scores = self.importance_score(attended).squeeze(-1)  # (B, N)

        return importance_scores, attended


class DynamicGraphConstructor(nn.Module):
    """
    Constructs dynamic subgraph by selecting important nodes
    Uses learned threshold or top-k selection
    """
    def __init__(self, selection_method='topk', k_ratio=0.5, threshold=0.5):
        super(DynamicGraphConstructor, self).__init__()

        self.selection_method = selection_method
        self.k_ratio = k_ratio
        self.threshold = nn.Parameter(torch.tensor(threshold)) if selection_method == 'learned_threshold' else threshold

    def forward(self, importance_scores, features, adjacency_matrix):
        """
        Args:
            importance_scores: (batch_size, num_nodes)
            features: (batch_size, num_nodes, channels)
            adjacency_matrix: (num_nodes, num_nodes)
        Returns:
            selected_features: (batch_size, k, channels) - features of selected nodes
            selected_indices: (batch_size, k) - indices of selected nodes
            selected_adjacency: (batch_size, k, k) - adjacency matrix for selected nodes
        """
        batch_size, num_nodes, channels = features.shape

        if self.selection_method == 'topk':
            # Select top-k nodes based on importance scores
            k = max(1, int(num_nodes * self.k_ratio))
            topk_scores, topk_indices = torch.topk(importance_scores, k, dim=1)  # (B, k)

        elif self.selection_method == 'threshold':
            # Select nodes above threshold
            threshold = self.threshold if isinstance(self.threshold, float) else self.threshold.item()
            # Note: This results in variable number of nodes per sample
            # For batch processing, we use top-k with dynamic k
            k = max(1, int(num_nodes * self.k_ratio))
            topk_scores, topk_indices = torch.topk(importance_scores, k, dim=1)

        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        # Gather selected node features
        # topk_indices: (B, k)
        # features: (B, N, C)
        batch_indices = torch.arange(batch_size, device=features.device).unsqueeze(1).expand(-1, k)
        selected_features = features[batch_indices, topk_indices]  # (B, k, C)

        # Construct adjacency matrix for selected nodes
        # adjacency_matrix: (N, N)
        # We need to extract submatrix for selected nodes
        selected_adjacency = self._construct_selected_adjacency(
            adjacency_matrix, topk_indices, k
        )  # (B, k, k)

        return selected_features, topk_indices, selected_adjacency, topk_scores

    def _construct_selected_adjacency(self, adjacency_matrix, selected_indices, k):
        """
        Construct adjacency matrix for selected nodes

        Args:
            adjacency_matrix: (num_nodes, num_nodes)
            selected_indices: (batch_size, k)
            k: number of selected nodes
        Returns:
            selected_adjacency: (batch_size, k, k)
        """
        batch_size = selected_indices.shape[0]
        device = selected_indices.device

        # For each batch, extract the submatrix
        selected_adjacency = torch.zeros(batch_size, k, k, device=device)

        for b in range(batch_size):
            indices = selected_indices[b]  # (k,)
            # Extract submatrix
            sub_adj = adjacency_matrix[indices][:, indices]  # (k, k)
            selected_adjacency[b] = sub_adj

        return selected_adjacency


class AdaptiveGraphConvolution(nn.Module):
    """
    Graph convolution that works with dynamic adjacency matrices
    """
    def __init__(self, in_channels, out_channels):
        super(AdaptiveGraphConvolution, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
            adjacency: (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        Returns:
            out: (batch_size, num_nodes, out_channels)
        """
        # x: (B, N, C_in)
        # weight: (C_in, C_out)
        # adjacency: (B, N, N) or (N, N)

        support = torch.matmul(x, self.weight)  # (B, N, C_out)

        if adjacency.dim() == 2:
            # Static adjacency matrix
            output = torch.matmul(adjacency, support)  # (N, N) x (B, N, C_out)
        else:
            # Dynamic adjacency matrix (batch-wise)
            output = torch.bmm(adjacency, support)  # (B, N, N) x (B, N, C_out) -> (B, N, C_out)

        output = output + self.bias

        return output


class AdaptiveSTGCNBlock(nn.Module):
    """
    STGCN block with adaptive graph convolution
    """
    def __init__(self, in_channels, out_channels, temporal_kernel_size=9, dropout=0.3):
        super(AdaptiveSTGCNBlock, self).__init__()

        # Temporal convolution 1
        self.temporal1 = TemporalConvolution(in_channels, out_channels, kernel_size=temporal_kernel_size)

        # Adaptive spatial graph convolution
        self.spatial = AdaptiveGraphConvolution(out_channels, out_channels)

        # Temporal convolution 2
        self.temporal2 = TemporalConvolution(out_channels, out_channels, kernel_size=temporal_kernel_size)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, adjacency):
        """
        Args:
            x: (batch_size, num_nodes, in_channels, time_steps)
            adjacency: (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
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
            out_t = self.spatial(x[:, t, :, :], adjacency)  # (B, N, C)
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
    Attention-based STGCN with Dynamic Graph Constructor

    Key features:
    - Learns importance of each node (keypoint) automatically
    - Constructs dynamic subgraphs based on node importance
    - Adaptive processing with selected important nodes
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

        # Model hyperparameters
        hidden_channels = model_cfg.get('hidden_channels', 64)
        num_layers = model_cfg.get('num_gcn_layers', 3)
        temporal_kernel_size = model_cfg.get('temporal_kernel_size', 9)
        dropout = model_cfg.get('dropout', 0.3)

        # Node selection parameters
        self.k_ratio = model_cfg.get('node_selection_ratio', 0.5)  # Keep top 50% nodes
        self.selection_method = model_cfg.get('selection_method', 'topk')

        # Determine feature dimensions
        if feature_dims is not None:
            body_features = feature_dims.get('body', 0)
            face_2d_features = feature_dims.get('face_2d', 0)
            head_pose_features = feature_dims.get('head_pose', 0)
        else:
            body_features = model_cfg['num_body_nodes'] * model_cfg['body_in_channels'] if self.use_body else 0
            face_2d_features = model_cfg['num_face_nodes'] * model_cfg['face_2d_in_channels'] if self.use_face_2d else 0
            head_pose_features = model_cfg['head_pose_features'] if self.use_head_pose else 0

        # Calculate number of nodes
        self.num_body_nodes = body_features // model_cfg.get('body_in_channels', 2) if body_features > 0 else 17
        self.num_face_nodes = face_2d_features // model_cfg.get('face_2d_in_channels', 2) if face_2d_features > 0 else 68

        # Create adjacency matrices
        if self.use_body and body_features > 0:
            self.body_adj = create_body_adjacency()
        else:
            self.body_adj = None

        if self.use_face_2d and face_2d_features > 0:
            self.face_adj = create_face_adjacency()
        else:
            self.face_adj = None

        # Input projection layers
        if self.use_body and body_features > 0:
            self.body_proj = nn.Linear(model_cfg.get('body_in_channels', 2), hidden_channels)
            self.body_node_attention = NodeAttentionModule(hidden_channels, hidden_channels)
            self.body_graph_constructor = DynamicGraphConstructor(
                selection_method=self.selection_method,
                k_ratio=self.k_ratio
            )
        else:
            self.body_proj = None
            self.body_node_attention = None
            self.body_graph_constructor = None

        if self.use_face_2d and face_2d_features > 0:
            self.face_2d_proj = nn.Linear(model_cfg.get('face_2d_in_channels', 2), hidden_channels)
            self.face_node_attention = NodeAttentionModule(hidden_channels, hidden_channels)
            self.face_graph_constructor = DynamicGraphConstructor(
                selection_method=self.selection_method,
                k_ratio=self.k_ratio
            )
        else:
            self.face_2d_proj = None
            self.face_node_attention = None
            self.face_graph_constructor = None

        # STGCN blocks for body (with adaptive graphs)
        if self.use_body and body_features > 0:
            self.body_stgcn_blocks = nn.ModuleList([
                AdaptiveSTGCNBlock(
                    hidden_channels,
                    hidden_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        else:
            self.body_stgcn_blocks = None

        # STGCN blocks for face (with adaptive graphs)
        if self.use_face_2d and face_2d_features > 0:
            self.face_stgcn_blocks = nn.ModuleList([
                AdaptiveSTGCNBlock(
                    hidden_channels,
                    hidden_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        else:
            self.face_stgcn_blocks = None

        # Head pose processing (no graph structure needed)
        if self.use_head_pose and head_pose_features > 0:
            self.head_pose_fc = nn.Sequential(
                nn.Linear(head_pose_features, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.head_pose_fc = None

        # Calculate total features for fusion layer
        total_features = 0
        if self.use_body and body_features > 0:
            # After node selection, we have k_ratio * num_body_nodes nodes
            selected_body_nodes = max(1, int(self.num_body_nodes * self.k_ratio))
            total_features += selected_body_nodes * hidden_channels
        if self.use_face_2d and face_2d_features > 0:
            selected_face_nodes = max(1, int(self.num_face_nodes * self.k_ratio))
            total_features += selected_face_nodes * hidden_channels
        if self.use_head_pose and head_pose_features > 0:
            total_features += hidden_channels

        if total_features == 0:
            raise ValueError("At least one feature type must be enabled")

        # Fusion and prediction layers
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
            nn.Linear(hidden_channels, 5)  # 5 classes
        )

    def forward(self, body_kps, face_kps_2d, head_pose):
        """
        Args:
            body_kps: (batch_size, sequence_length, num_body_nodes * 2) or None
            face_kps_2d: (batch_size, sequence_length, num_face_nodes * 2) or None
            head_pose: (batch_size, sequence_length, head_pose_features) or None
        Returns:
            logits: (batch_size, 5) - logits for 5 attention classes
        """
        features_list = []

        # Process body keypoints with node attention
        if self.use_body and self.body_proj is not None and body_kps is not None:
            batch_size, seq_len = body_kps.shape[0], body_kps.shape[1]

            # Reshape: (B, T, N*2) -> (B, T, N, 2)
            body_kps = body_kps.reshape(batch_size, seq_len, self.num_body_nodes, 2)

            # Project features
            body_features = self.body_proj(body_kps)  # (B, T, N, hidden)

            # Apply node attention (average over time for node selection)
            body_features_avg = body_features.mean(dim=1)  # (B, N, hidden)
            importance_scores, attended = self.body_node_attention(body_features_avg)

            # Select important nodes
            selected_features, selected_indices, selected_adj, _ = self.body_graph_constructor(
                importance_scores, attended, self.body_adj
            )

            # Gather temporal features for selected nodes
            k = selected_features.shape[1]
            batch_indices = torch.arange(batch_size, device=body_features.device).unsqueeze(1).unsqueeze(2).expand(-1, seq_len, k)
            node_indices = selected_indices.unsqueeze(1).expand(-1, seq_len, -1)
            body_features_selected = body_features[batch_indices, torch.arange(seq_len, device=body_features.device).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, k), node_indices]

            # Reshape for STGCN: (B, T, k, C) -> (B, k, C, T)
            body_features_selected = body_features_selected.permute(0, 2, 3, 1)

            # Apply STGCN blocks with dynamic adjacency
            for block in self.body_stgcn_blocks:
                body_features_selected = block(body_features_selected, selected_adj)

            # Global pooling over time
            body_features_selected = body_features_selected.mean(dim=-1)  # (B, k, C)

            # Flatten
            body_features_selected = body_features_selected.reshape(batch_size, -1)
            features_list.append(body_features_selected)
        else:
            batch_size = face_kps_2d.shape[0] if face_kps_2d is not None else head_pose.shape[0]
            seq_len = face_kps_2d.shape[1] if face_kps_2d is not None else head_pose.shape[1]

        # Process face keypoints with node attention
        if self.use_face_2d and self.face_2d_proj is not None and face_kps_2d is not None:
            # Reshape: (B, T, N*2) -> (B, T, N, 2)
            face_kps_2d = face_kps_2d.reshape(batch_size, seq_len, self.num_face_nodes, 2)

            # Project features
            face_features = self.face_2d_proj(face_kps_2d)  # (B, T, N, hidden)

            # Apply node attention
            face_features_avg = face_features.mean(dim=1)  # (B, N, hidden)
            importance_scores, attended = self.face_node_attention(face_features_avg)

            # Select important nodes
            selected_features, selected_indices, selected_adj, _ = self.face_graph_constructor(
                importance_scores, attended, self.face_adj
            )

            # Gather temporal features for selected nodes
            k = selected_features.shape[1]
            batch_indices = torch.arange(batch_size, device=face_features.device).unsqueeze(1).unsqueeze(2).expand(-1, seq_len, k)
            node_indices = selected_indices.unsqueeze(1).expand(-1, seq_len, -1)
            face_features_selected = face_features[batch_indices, torch.arange(seq_len, device=face_features.device).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, k), node_indices]

            # Reshape for STGCN: (B, T, k, C) -> (B, k, C, T)
            face_features_selected = face_features_selected.permute(0, 2, 3, 1)

            # Apply STGCN blocks with dynamic adjacency
            for block in self.face_stgcn_blocks:
                face_features_selected = block(face_features_selected, selected_adj)

            # Global pooling over time
            face_features_selected = face_features_selected.mean(dim=-1)  # (B, k, C)

            # Flatten
            face_features_selected = face_features_selected.reshape(batch_size, -1)
            features_list.append(face_features_selected)

        # Process head pose
        if self.use_head_pose and self.head_pose_fc is not None and head_pose is not None:
            head_pose_avg = head_pose.mean(dim=1)
            head_pose_features = self.head_pose_fc(head_pose_avg)
            features_list.append(head_pose_features)

        # Concatenate all features
        if len(features_list) == 0:
            raise ValueError("No features available for prediction")

        combined = torch.cat(features_list, dim=-1)

        # Predict attention
        logits = self.fusion(combined)

        return logits


if __name__ == '__main__':
    # Test the attention STGCN model
    import json

    config = {
        'model': {
            'hidden_channels': 64,
            'num_gcn_layers': 3,
            'temporal_kernel_size': 9,
            'dropout': 0.3,
            'num_body_nodes': 17,
            'body_in_channels': 2,
            'num_face_nodes': 68,
            'face_2d_in_channels': 2,
            'head_pose_features': 9,
            'node_selection_ratio': 0.5,  # Keep top 50% important nodes
            'selection_method': 'topk'
        },
        'data': {
            'features': {
                'use_body_kps': True,
                'use_face_kps_2d': True,
                'use_head_pose': True
            }
        },
        'training': {
            'sequence_length': 30
        }
    }

    feature_dims = {
        'body': 17 * 2,
        'face_2d': 68 * 2,
        'head_pose': 9
    }

    print("Creating Attention STGCN with Dynamic Graph Constructor...")
    model = AttentionSTGCN(config, feature_dims)

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

    print("\nModel with dynamic graph constructor created successfully!")
