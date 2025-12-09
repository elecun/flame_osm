"""
Model Training Usage Examples
==============================

This file contains examples of how to train different models with the updated training script.
"""

# ============================================================================
# Example 1: Train with GeneralSTGCN (original model)
# ============================================================================
"""
python train.py --config config.json --csv merge_0.csv --model general --device cuda

This uses the GeneralSTGCN model (previously named AttentionSTGCN) with body keypoints,
face landmarks, and head pose.
"""

# ============================================================================
# Example 2: Train with AttentionSTGCN (NEW - with dynamic graph constructor)
# ============================================================================
"""
python train.py --config config_attention.json --csv merge_0.csv --model attention --device cuda

This uses the NEW AttentionSTGCN model with dynamic graph constructor that:
- Automatically learns node (keypoint) importance
- Selects top-k important nodes dynamically
- Constructs adaptive subgraphs based on node importance
- More efficient and potentially more accurate
"""

# ============================================================================
# Example 3: Train with Multi-Stream STGCN model (default 3 streams)
# ============================================================================
"""
python train.py --config config_multi_stream.json --csv merge_0.csv --model multi_stream --device cuda

The multi-stream model uses 3 default streams:
  1. Face landmarks (68 points, 2D coordinates)
  2. Head pose (rotation, translation, pitch/yaw/roll)
  3. Body pose (17 COCO keypoints, 2D coordinates)
"""

# ============================================================================
# Example 4: Train specific fold only
# ============================================================================
"""
python train.py --config config.json --csv merge_0.csv --model attention --fold 0

This trains only fold 0 instead of all 5 folds.
"""

# ============================================================================
# Example 5: Multi-GPU training with DDP
# ============================================================================
"""
python train.py --config config.json --csv merge_0.csv --model multi_stream --num_gpus 2

This uses 2 GPUs for distributed data parallel training.
"""

# ============================================================================
# Example 6: Custom stream configuration
# ============================================================================
"""
To use custom streams, modify config_multi_stream.json and uncomment the 'stream_configs' section.

You can define your own streams by specifying:
  - column_pattern: Which columns from CSV to use (regex pattern or list)
  - num_nodes: Number of nodes in the graph
  - in_channels: Input channels per node (e.g., 2 for x,y coordinates)
  - adjacency_type: How nodes are connected ('body', 'face', 'sequential', 'fully_connected', 'star')
  - description: Human-readable description

Example custom stream configuration in config.json:
{
  "stream_configs": {
    "face_landmarks": {
      "column_pattern": "landmark_.*_2d",
      "num_nodes": 68,
      "in_channels": 2,
      "adjacency_type": "face",
      "description": "Face landmarks"
    },
    "custom_features": {
      "column_pattern": ["feature_1", "feature_2", "feature_3"],
      "num_nodes": 3,
      "in_channels": 1,
      "adjacency_type": "sequential",
      "description": "Custom features"
    }
  }
}

Then train with:
python train.py --config config_with_custom_streams.json --csv merge_0.csv --model multi_stream
"""

# ============================================================================
# Programmatic Usage Example
# ============================================================================

if __name__ == '__main__':
    """
    Example of using the models programmatically (not through train.py)
    """
    import torch
    from model import GeneralSTGCN
    from attention_stgcn import AttentionSTGCN
    from multi_stream_stgcn import MultiStreamSTGCN, DEFAULT_STREAM_CONFIGS
    import json

    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Define feature dimensions (from your data)
    feature_dims = {
        'body': 17 * 2,      # 17 body keypoints, 2D (x, y)
        'face_2d': 68 * 2,   # 68 face landmarks, 2D (x, y)
        'head_pose': 9        # 9 head pose features
    }

    # ========================================
    # Option 1: Use GeneralSTGCN (original model)
    # ========================================
    print("Creating GeneralSTGCN model...")
    model_general = GeneralSTGCN(config, feature_dims)
    print(f"Total parameters: {sum(p.numel() for p in model_general.parameters()):,}")

    # ========================================
    # Option 2: Use AttentionSTGCN (with dynamic graph constructor)
    # ========================================
    print("\nCreating AttentionSTGCN with dynamic graph constructor...")
    model_attention = AttentionSTGCN(config, feature_dims)
    print(f"Total parameters: {sum(p.numel() for p in model_attention.parameters()):,}")

    # ========================================
    # Option 3: Use Multi-Stream STGCN with default streams
    # ========================================
    print("\nCreating Multi-Stream STGCN with default configuration...")
    model_multi = MultiStreamSTGCN(config, feature_dims)
    print(model_multi.get_stream_info())
    print(f"Total parameters: {sum(p.numel() for p in model_multi.parameters()):,}")

    # ========================================
    # Option 4: Use Multi-Stream STGCN with custom streams
    # ========================================
    print("\nCreating Multi-Stream STGCN with custom streams (face + body only)...")
    custom_streams = {
        'face_landmarks': DEFAULT_STREAM_CONFIGS['face_landmarks'],
        'body_pose': DEFAULT_STREAM_CONFIGS['body_pose']
    }
    model_custom = MultiStreamSTGCN(config, feature_dims, stream_configs=custom_streams)
    print(model_custom.get_stream_info())
    print(f"Total parameters: {sum(p.numel() for p in model_custom.parameters()):,}")

    # ========================================
    # Test forward pass
    # ========================================
    batch_size = 4
    seq_len = config['training']['sequence_length']

    # Create dummy input data
    body_kps = torch.randn(batch_size, seq_len, 17 * 2)
    face_kps_2d = torch.randn(batch_size, seq_len, 68 * 2)
    head_pose = torch.randn(batch_size, seq_len, 9)

    # Forward pass with GeneralSTGCN
    output_general = model_general(body_kps, face_kps_2d, head_pose)
    print(f"\nGeneralSTGCN output shape: {output_general.shape}")

    # Forward pass with AttentionSTGCN
    output_attention = model_attention(body_kps, face_kps_2d, head_pose)
    print(f"AttentionSTGCN output shape: {output_attention.shape}")

    # Forward pass with Multi-Stream STGCN
    output_multi = model_multi(body_kps, face_kps_2d, head_pose)
    print(f"Multi-Stream STGCN output shape: {output_multi.shape}")

    # Forward pass with custom Multi-Stream STGCN (no head pose)
    output_custom = model_custom(body_kps, face_kps_2d, None)
    print(f"Custom Multi-Stream STGCN output shape: {output_custom.shape}")

    print("\nAll models working correctly!")
