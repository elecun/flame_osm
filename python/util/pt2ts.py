#!/usr/bin/env python3
import argparse
import sys
import os
import torch

# Ensure python directory and model subdirectories are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
python_model_dir = os.path.join(project_root, "python/model")
python_train_dir = os.path.join(project_root, "python/train")

for d in [python_model_dir, python_train_dir, project_root]:
    if d not in sys.path and os.path.exists(d):
        sys.path.insert(0, d)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert complete PyTorch (.pt) model to TorchScript format."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str,
        help="Path to input PyTorch model file (.pt)."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to save output TorchScript file (.torchscript / .pt). Default: <input_name>.torchscript"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda",
        help="Device to use for conversion ('cuda' or 'cpu'). Default: cuda"
    )
    parser.add_argument(
        "-m", "--method", type=str, choices=["script", "trace"], default="trace",
        help="Conversion method: 'script' (torch.jit.script) or 'trace' (torch.jit.trace). Default: trace"
    )
    parser.add_argument(
        "--input-shape-body", type=int, nargs="+", default=[1, 64, 22],
        help="Dummy input shape for body pose stream in tracing (batch, seq_len, features). Default: 1 64 22"
    )
    parser.add_argument(
        "--input-shape-head", type=int, nargs="+", default=[1, 64, 3],
        help="Dummy input shape for head pose stream in tracing (batch, seq_len, features). Default: 1 64 3"
    )
    return parser.parse_args()

def convert_pt_to_ts(input_path, output_path, device_str, method, input_shape_body, input_shape_head):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but CUDA is not available on system. Falling back to CPU.")
        device = torch.device("cpu")

    print(f"Loading PyTorch model from: {input_path}")
    try:
        model = torch.load(input_path, map_location=device)
    except Exception as e:
        print(f"Error loading model with torch.load: {e}")
        sys.exit(1)

    if not isinstance(model, torch.nn.Module):
        print(f"Error: Loaded object is not a PyTorch Module. Type: {type(model).__name__}")
        sys.exit(1)

    model.to(device)
    model.eval()

    print(f"Converting PyTorch model using torch.jit.{method} on device: {device}")
    ts_model = None
    if method == "script":
        try:
            ts_model = torch.jit.script(model)
        except Exception as e:
            print(f"torch.jit.script failed ({e}). Falling back to torch.jit.trace...")
            method = "trace"

    if method == "trace" or ts_model is None:
        dummy_body = torch.randn(*input_shape_body, device=device)
        dummy_head = torch.randn(*input_shape_head, device=device)
        dummy_face = torch.zeros((1, 64, 0), device=device)
        print(f"Tracing model on device '{device}' with dummy inputs: body={dummy_body.shape}, face={dummy_face.shape}, head={dummy_head.shape}")
        
        with torch.no_grad():
            ts_model = torch.jit.trace(model, (dummy_body, dummy_face, dummy_head), check_trace=False)

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.torchscript"

    print(f"Saving TorchScript model to: {output_path}")
    ts_model.save(output_path)
    print("TorchScript conversion completed successfully!")

def main():
    args = parse_args()
    convert_pt_to_ts(
        input_path=args.input,
        output_path=args.output,
        device_str=args.device,
        method=args.method,
        input_shape_body=args.input_shape_body,
        input_shape_head=args.input_shape_head
    )

if __name__ == "__main__":
    main()
