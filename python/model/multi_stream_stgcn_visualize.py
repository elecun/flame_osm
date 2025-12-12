#!/usr/bin/env python3
"""
Utility to visualize the Multi-Stream STGCN computation graph with torchviz.

Usage:
    python python/model/multi_stream_stgcn_visualize.py \
        --config python/model/config.json \
        --output stgcn_graph --format pdf
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor
from torchviz import make_dot

from multi_stream_stgcn import MultiStreamSTGCN, DEFAULT_STREAM_CONFIGS


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def ensure_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal defaults so the visualizer can run even with sparse config files."""
    cfg.setdefault("training", {})
    cfg["training"].setdefault("sequence_length", 60)
    cfg.setdefault("model", {})
    cfg["model"].setdefault("num_classes", 5)
    cfg["model"].setdefault("hidden_channels", 64)
    cfg["model"].setdefault("num_gcn_layers", 3)
    cfg["model"].setdefault("temporal_kernel_size", 9)
    cfg["model"].setdefault("dropout", 0.3)
    return cfg


def merge_stream_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert stream config in JSON to a dict usable by MultiStreamSTGCN.
    Only keeps streams where enabled=True (default). Face stream defaults to 468 landmarks.
    """
    streams_cfg = config.get("streams")

    defaults = deepcopy(DEFAULT_STREAM_CONFIGS)
    # Override face defaults to use 468 landmarks and a generic adjacency that scales
    if "face_landmarks" in defaults:
        defaults["face_landmarks"].update({"num_nodes": 468, "adjacency_type": "fully_connected"})

    if streams_cfg is None:
        merged = defaults
    else:
        merged = {}
        if isinstance(streams_cfg, list):
            for stream in streams_cfg:
                name = stream["name"]
                base = defaults.get(name, {})
                merged[name] = {**base, **stream}
        elif isinstance(streams_cfg, dict):
            for name, stream in streams_cfg.items():
                base = defaults.get(name, {})
                merged[name] = {**base, **stream, "name": name}
        else:
            raise ValueError("streams must be a list or dict in config.json")

    enabled = {name: cfg for name, cfg in merged.items() if cfg.get("enabled", True)}
    if not enabled:
        raise ValueError("At least one stream must be enabled in config.streams[*].enabled.")
    return enabled


def create_dummy_inputs(stream_configs: Dict[str, Dict[str, Any]], seq_len: int, batch_size: int = 1) -> Dict[str, Tensor]:
    """Create random tensors that match the enabled stream shapes."""
    inputs: Dict[str, Tensor] = {"body_kps": None, "face_kps_2d": None, "head_pose": None}
    mapping = {
        "body_pose": "body_kps",
        "face_landmarks": "face_kps_2d",
        "head_pose": "head_pose",
    }

    for stream_name, cfg in stream_configs.items():
        target = mapping.get(stream_name)
        if target is None:
            raise ValueError(f"Unknown stream '{stream_name}' – update mapping in create_dummy_inputs.")
        num_nodes = cfg.get("num_nodes")
        in_channels = cfg.get("in_channels")
        if num_nodes is None or in_channels is None:
            raise ValueError(f"Stream '{stream_name}' missing num_nodes or in_channels.")
        feature_dim = num_nodes * in_channels
        inputs[target] = torch.randn(batch_size, seq_len, feature_dim)
    return inputs


def visualize(config_path: Path, output: str, fmt: str) -> None:
    config = ensure_defaults(load_config(config_path))
    stream_configs = merge_stream_configs(config)

    model = MultiStreamSTGCN(config, stream_configs=stream_configs)
    model.eval()

    seq_len = config["training"]["sequence_length"]
    dummy_inputs = create_dummy_inputs(stream_configs, seq_len=seq_len, batch_size=1)

    logits = model(dummy_inputs.get("body_kps"), dummy_inputs.get("face_kps_2d"), dummy_inputs.get("head_pose"))
    graph = make_dot(logits, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    graph.render(output, format=fmt, cleanup=True)
    print(f"Saved graph to {output}.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Multi-Stream STGCN graph with torchviz")
    parser.add_argument("--config", type=str, default="python/model/config.json", help="Path to config JSON")
    parser.add_argument("--output", type=str, default="stgcn_graph", help="Output file prefix (no extension)")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"], help="Output file format")
    args = parser.parse_args()

    visualize(Path(args.config), args.output, args.format)


if __name__ == "__main__":
    main()
