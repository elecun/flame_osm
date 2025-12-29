"""
Multi-Stream STGCN with optional cross-stream attention.

Interface matches multi_stream_stgcn.py:
    model = MultiStreamSTGCNXAttn(config, feature_dims=None, stream_configs=None)
    model(body_kps, face_kps_2d, head_pose)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import STGCNBlock, create_body_adjacency, create_face_adjacency, create_generic_adjacency
from multi_stream_stgcn import DEFAULT_STREAM_CONFIGS


class StreamProcessor(nn.Module):
    def __init__(self, stream_config, hidden_channels=64, num_layers=3, temporal_kernel_size=9, dropout=0.3):
        super().__init__()
        self.stream_config = stream_config
        self.num_nodes = stream_config["num_nodes"]
        self.in_channels = stream_config["in_channels"]

        adj_type = stream_config["adjacency_type"]
        if adj_type == "body":
            self.adjacency = create_body_adjacency(self.num_nodes)
        elif adj_type == "face":
            self.adjacency = create_face_adjacency(self.num_nodes)
        elif adj_type in ["sequential", "fully_connected", "star"]:
            self.adjacency = create_generic_adjacency(self.num_nodes, adj_type)
        else:
            raise ValueError(f"Unknown adjacency type: {adj_type}")

        self.input_proj = nn.Linear(self.in_channels, hidden_channels)
        self.stgcn_blocks = nn.ModuleList(
            [
                STGCNBlock(
                    hidden_channels,
                    hidden_channels,
                    self.adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_nodes, self.in_channels)
        x = self.input_proj(x)
        x = x.permute(0, 2, 3, 1)
        for block in self.stgcn_blocks:
            x = block(x)
        x = x.mean(dim=-1)
        x = x.reshape(batch_size, -1)
        return x


class MultiStreamSTGCNXAttn(nn.Module):
    def __init__(self, config, feature_dims=None, stream_configs=None):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        if stream_configs is None:
            stream_configs = DEFAULT_STREAM_CONFIGS
        self.stream_configs = stream_configs
        self.stream_names = list(stream_configs.keys())

        hidden_channels = model_cfg.get("hidden_channels", 64)
        num_layers = model_cfg.get("num_gcn_layers", 3)
        temporal_kernel_size = model_cfg.get("temporal_kernel_size", 9)
        dropout = model_cfg.get("dropout", 0.3)

        # Cross attention config
        attn_cfg = model_cfg.get("cross_attention", {})
        self.attn_enabled = attn_cfg.get("enabled", False)
        self.attn_dim = attn_cfg.get("d_model", hidden_channels)
        self.attn_heads = attn_cfg.get("num_heads", 4)
        self.attn_dropout = attn_cfg.get("dropout", 0.1)

        # Stream processors
        self.streams = nn.ModuleDict()
        for stream_name, stream_config in stream_configs.items():
            self.streams[stream_name] = StreamProcessor(
                stream_config,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                temporal_kernel_size=temporal_kernel_size,
                dropout=dropout,
            )

        # Feature sizes and projections for attention
        self.total_features = 0
        self.attn_projs = nn.ModuleDict()
        for stream_name, stream_config in stream_configs.items():
            num_nodes = stream_config["num_nodes"]
            self.total_features += num_nodes * hidden_channels
            self.attn_projs[stream_name] = nn.Linear(num_nodes * hidden_channels, self.attn_dim)

        if self.attn_enabled and len(self.stream_names) > 1:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.attn_dim,
                num_heads=self.attn_heads,
                dropout=self.attn_dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(self.attn_dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.attn_dim, self.attn_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.attn_dropout),
                nn.Linear(self.attn_dim * 2, self.attn_dim),
            )
            self.ffn_norm = nn.LayerNorm(self.attn_dim)
            fusion_in_dim = len(self.stream_names) * self.attn_dim
        else:
            self.cross_attn = None
            fusion_in_dim = self.total_features

        # Fusion/classifier
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, model_cfg.get("num_classes", 5)),
        )

        self._setup_feature_mapping(feature_dims)

    def _setup_feature_mapping(self, feature_dims):
        if feature_dims is None:
            self.feature_mapping = {
                "body_pose": "body_kps",
                "face_landmarks": "face_kps_2d",
                "head_pose": "head_pose",
            }
        else:
            self.feature_mapping = {}
            if "body" in feature_dims:
                self.feature_mapping["body_pose"] = "body_kps"
            if "face_2d" in feature_dims:
                self.feature_mapping["face_landmarks"] = "face_kps_2d"
            if "head_pose" in feature_dims:
                self.feature_mapping["head_pose"] = "head_pose"

    def forward(self, body_kps=None, face_kps_2d=None, head_pose=None):
        stream_inputs = {
            "body_pose": body_kps,
            "face_landmarks": face_kps_2d,
            "head_pose": head_pose,
        }

        stream_features = []
        attn_inputs = []
        missing_streams = []

        for stream_name in self.stream_names:
            if stream_name not in self.streams or stream_name not in stream_inputs:
                continue
            stream_input = stream_inputs[stream_name]
            if stream_input is None or stream_input.numel() == 0 or stream_input.shape[-1] == 0:
                missing_streams.append(stream_name)
                continue
            feats = self.streams[stream_name](stream_input)
            stream_features.append(feats)
            attn_inputs.append(self.attn_projs[stream_name](feats))

        if len(stream_features) == 0:
            msg = "No valid input features provided"
            if missing_streams:
                msg += f". Missing/empty streams: {', '.join(missing_streams)}"
            raise ValueError(msg)

        if self.attn_enabled and len(attn_inputs) > 1 and self.cross_attn is not None:
            # B x S x D
            stacked = torch.stack(attn_inputs, dim=1)
            attn_out, _ = self.cross_attn(stacked, stacked, stacked)
            attn_out = self.cross_attn_norm(attn_out + stacked)
            ffn_out = self.ffn(attn_out)
            attn_out = self.ffn_norm(ffn_out + attn_out)
            combined_features = attn_out.reshape(attn_out.shape[0], -1)
        elif self.attn_enabled and len(attn_inputs) == 1:
            # Ensure projection weights get gradients even with a single stream
            combined_features = attn_inputs[0]
        else:
            combined_features = torch.cat(stream_features, dim=-1)

        logits = self.fusion(combined_features)
        return logits


def create_multi_stream_model(config, feature_dims=None, stream_configs=None):
    model = MultiStreamSTGCNXAttn(config, feature_dims, stream_configs)
    return model
