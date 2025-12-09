"""
Base components for STGCN models
Common layers and utilities shared across different model architectures
"""

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


def create_body_adjacency():
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


def create_face_adjacency():
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


def create_generic_adjacency(num_nodes, connection_type='sequential'):
    """
    Create a generic adjacency matrix for arbitrary number of nodes

    Args:
        num_nodes: Number of nodes in the graph
        connection_type: Type of connections ('sequential', 'fully_connected', 'star')

    Returns:
        Normalized adjacency matrix
    """
    adj = torch.zeros(num_nodes, num_nodes)

    if connection_type == 'sequential':
        # Connect nodes sequentially
        for i in range(num_nodes - 1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
    elif connection_type == 'fully_connected':
        # Fully connected graph
        adj = torch.ones(num_nodes, num_nodes)
    elif connection_type == 'star':
        # Star topology with first node as center
        for i in range(1, num_nodes):
            adj[0, i] = 1
            adj[i, 0] = 1
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

    # Add self-loops
    adj = adj + torch.eye(num_nodes)

    # Normalize adjacency matrix
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    adj_normalized = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)

    return adj_normalized
