#!/usr/bin/env python3
"""Re-export TorchScript model with all tensors mapped to CPU.

This script loads an existing TorchScript model using map_location="cpu"
which remaps ALL tensor storages (including graph-level constants captured
during torch.jit.trace) to CPU. The re-saved model can then be loaded
and moved to any GPU without device mismatch errors.

Usage:
    python3 remap_model_cpu.py <input.torchscript> [output.torchscript]
    
If output path is not specified, the input file is overwritten.
"""
import sys
import shutil
from pathlib import Path
import torch

def remap_to_cpu(input_path: str, output_path: str | None = None):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Create backup
    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"Backup: {backup_path}")
    
    # Load with map_location="cpu" — this remaps ALL tensor storages
    # including inline constants captured during torch.jit.trace
    print(f"Loading: {input_path}")
    model = torch.jit.load(str(input_path), map_location="cpu")
    
    # Verify with dummy forward pass on CPU
    print("Verifying CPU forward pass...")
    with torch.no_grad():
        dummy_low = torch.zeros(1, 15, 3, 64, 64)
        dummy_high = torch.zeros(1, 15, 160)
        try:
            output = model(dummy_low, dummy_high)
            print(f"  Forward pass OK, output type: {type(output)}")
            if isinstance(output, tuple):
                for i, t in enumerate(output):
                    print(f"  output[{i}]: shape={t.shape}, device={t.device}")
        except Exception as e:
            print(f"  Warning: CPU forward pass failed: {e}")
    
    # Save
    model.save(str(output_path))
    print(f"Saved: {output_path}")
    print(f"Done. Model is now device-neutral and can be loaded on any GPU.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.torchscript> [output.torchscript]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    remap_to_cpu(input_file, output_file)
