#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn

from blinklinmult.models import DenseNet121, BlinkLinT, BlinkLinMulT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pt2ts")


class BlinkLinMulTWrapper(nn.Module):
    """Wrapper to accept multi-tensor inputs for TorchScript tracing."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rgb_texture: torch.Tensor, high_level_features: torch.Tensor):
        return self.model([rgb_texture, high_level_features])


def convert_densenet121(pt_path: Path, output_dir: Path, device: torch.device):
    logger.info(f"Converting DenseNet121: {pt_path.name}")
    model = DenseNet121(output_dim=1, weights=str(pt_path)).to(device).eval()

    dummy_input = torch.randn(1, 3, 64, 64, device=device)

    with torch.no_grad():
        # Warmup for LazyLinear initialization
        model(dummy_input)
        traced_model = torch.jit.trace(model, dummy_input, check_trace=False)

        # Verification
        out_orig = model(dummy_input)
        out_ts = traced_model(dummy_input)
        if isinstance(out_orig, (tuple, list)):
            out_orig = out_orig[0]
        if isinstance(out_ts, (tuple, list)):
            out_ts = out_ts[0]
        diff = (out_orig - out_ts).abs().max().item()
        logger.info(f"  Verification max diff: {diff:.6e}")

    # Save torchscript files
    ts_path = output_dir / f"{pt_path.stem}.torchscript"
    traced_model.save(str(ts_path))
    logger.info(f"  Saved -> {ts_path}")

    ts_ext_path = output_dir / f"{pt_path.stem}.ts"
    traced_model.save(str(ts_ext_path))
    logger.info(f"  Saved -> {ts_ext_path}")

    return ts_path


def convert_blinklint(pt_path: Path, output_dir: Path, device: torch.device):
    logger.info(f"Converting BlinkLinT: {pt_path.name}")
    model = BlinkLinT(output_dim=1, weights=str(pt_path)).to(device).eval()

    dummy_input = torch.randn(1, 15, 3, 64, 64, device=device)

    with torch.no_grad():
        model(dummy_input)
        traced_model = torch.jit.trace(model, dummy_input, check_trace=False)

        # Verification
        out_orig = model(dummy_input)
        out_ts = traced_model(dummy_input)
        if isinstance(out_orig, (tuple, list)):
            for i, (orig, ts) in enumerate(zip(out_orig, out_ts)):
                diff = (orig - ts).abs().max().item()
                logger.info(f"  Verification output[{i}] diff: {diff:.6e}")
        else:
            diff = (out_orig - out_ts).abs().max().item()
            logger.info(f"  Verification diff: {diff:.6e}")

    ts_path = output_dir / f"{pt_path.stem}.torchscript"
    traced_model.save(str(ts_path))
    logger.info(f"  Saved -> {ts_path}")

    ts_ext_path = output_dir / f"{pt_path.stem}.ts"
    traced_model.save(str(ts_ext_path))
    logger.info(f"  Saved -> {ts_ext_path}")

    return ts_path


def convert_blinklinmult(pt_path: Path, output_dir: Path, device: torch.device, backbone_weights: Path | None = None):
    logger.info(f"Converting BlinkLinMulT: {pt_path.name}")
    backbone_weights_str = str(backbone_weights) if backbone_weights else "densenet121-union"
    model = BlinkLinMulT(
        input_dim=160,
        output_dim=1,
        weights=str(pt_path),
        weights_backbone=backbone_weights_str
    ).to(device).eval()

    wrapper = BlinkLinMulTWrapper(model).to(device).eval()

    dummy_x1 = torch.randn(1, 15, 3, 64, 64, device=device)
    dummy_x2 = torch.randn(1, 15, 160, device=device)

    with torch.no_grad():
        wrapper(dummy_x1, dummy_x2)
        traced_model = torch.jit.trace(wrapper, (dummy_x1, dummy_x2), check_trace=False)

        # Verification
        out_orig = model([dummy_x1, dummy_x2])
        out_ts = traced_model(dummy_x1, dummy_x2)
        if isinstance(out_orig, (tuple, list)):
            for i, (orig, ts) in enumerate(zip(out_orig, out_ts)):
                diff = (orig - ts).abs().max().item()
                logger.info(f"  Verification output[{i}] diff: {diff:.6e}")
        else:
            diff = (out_orig - out_ts).abs().max().item()
            logger.info(f"  Verification diff: {diff:.6e}")

    ts_path = output_dir / f"{pt_path.stem}.torchscript"
    traced_model.save(str(ts_path))
    logger.info(f"  Saved -> {ts_path}")

    ts_ext_path = output_dir / f"{pt_path.stem}.ts"
    traced_model.save(str(ts_ext_path))
    logger.info(f"  Saved -> {ts_ext_path}")

    return ts_path


def main():
    base_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"=== PyTorch to TorchScript Converter (pt2ts) ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Working Directory: {base_dir}")

    pt_files = sorted(list(base_dir.glob("*.pt")))
    if not pt_files:
        logger.warning(f"No .pt files found in {base_dir}")
        return

    logger.info(f"Found {len(pt_files)} .pt file(s): {[f.name for f in pt_files]}")

    # Check for backbone weights
    densenet_pt = base_dir / "densenet121-union-64.pt"
    backbone_weights = densenet_pt if densenet_pt.exists() else None

    converted_count = 0
    for pt_file in pt_files:
        name = pt_file.name.lower()
        try:
            if "blinklinmult" in name or "linmult" in name:
                convert_blinklinmult(pt_file, base_dir, device, backbone_weights)
                converted_count += 1
            elif "blinklint" in name or "densenetlint" in name or "lint" in name:
                convert_blinklint(pt_file, base_dir, device)
                converted_count += 1
            elif "densenet121" in name or "densenet" in name:
                convert_densenet121(pt_file, base_dir, device)
                converted_count += 1
            else:
                logger.warning(f"Unrecognized model type for {pt_file.name}, skipping.")
        except Exception as e:
            logger.error(f"Failed to convert {pt_file.name}: {e}", exc_info=True)

    logger.info(f"Successfully converted {converted_count}/{len(pt_files)} model(s) to TorchScript.")


if __name__ == "__main__":
    main()
