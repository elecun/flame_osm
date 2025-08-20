import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """Graph convolution layer for ST-GCN"""
    
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )
        
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return x.contiguous()


class STGCNBlock(nn.Module):
    """ST-GCN block with spatial and temporal convolutions"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCN(nn.Module):
    """Spatial Temporal Graph Convolutional Network"""
    
    def __init__(self, in_channels=3, num_class=256, graph_args=None, edge_importance_weighting=True, **kwargs):
        super().__init__()
        
        # Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # Build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        ))
        
        # Initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            
        # FCN for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        
    def forward(self, x):
        # Data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
            
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        
        # Prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        
        return x


class Graph:
    """Graph structure for human body keypoints with head pose integration"""
    
    def __init__(self, layout='body_head', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)
        
    def __str__(self):
        return self.A
    
    def get_edge(self, layout):
        if layout == 'body_head':
            # 17 body keypoints + 3 head pose nodes (position + orientation)
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            
            # Body keypoint connections (COCO-style)
            neighbor_link = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # head connections
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # arms
                (5, 11), (6, 12), (11, 12),  # torso
                (11, 13), (12, 14), (13, 15), (14, 16),  # legs
                # Head pose connections (nodes 17, 18, 19)
                (0, 17), (17, 18), (17, 19)  # connect head keypoint to head pose
            ]
            
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError("Unsupported layout: " + layout)
            
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
            
        normalize_adjacency = normalize_digraph(adjacency)
        
        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
                    
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Unsupported strategy: " + strategy)


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
        
    # Compute hop distance
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
        
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD
