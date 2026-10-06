import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Fix bug in onnx2torch for negative axes in OnnxReduceStaticAxes
from onnx2torch.node_converters import reduce

def fixed_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    if self.axes is None or len(self.axes) == 0:
        if not self.keepdims:
            return self.math_op_function(input_tensor)
        axes = list(range(input_tensor.dim()))
    else:
        axes = [a if a >= 0 else a + input_tensor.dim() for a in self.axes]
        axes = sorted(axes)

    if self.operation_type in ['ReduceMax', 'ReduceMin']:
        op = torch.amax if self.operation_type == 'ReduceMax' else torch.amin
        return op(input_tensor, dim=axes, keepdim=self.keepdims)
    elif self.operation_type == 'ReduceProd':
        result = input_tensor
        for passed_dims, axis in enumerate(axes):
            result = torch.prod(result, dim=axis if self.keepdims else axis - passed_dims, keepdim=self.keepdims)
        return result
    return self.math_op_function(input_tensor, dim=axes, keepdim=self.keepdims)

reduce.OnnxReduceStaticAxes.forward = fixed_forward

from onnx2torch import convert
import onnxruntime as ort

def convert_ocec_onnx_to_torchscript(onnx_path, ts_path):
    print(f"Loading ONNX from {onnx_path}...")
    ort_sess = ort.InferenceSession(onnx_path)

    print("Converting ONNX to PyTorch module via onnx2torch...")
    torch_model = convert(onnx_path)
    torch_model.eval()
    torch_model.cpu()

    print("Verifying numerical consistency between ONNX and PyTorch...")
    max_err = 0.0
    for i in range(10):
        inp = np.random.uniform(0.0, 1.0, size=(1, 3, 24, 40)).astype(np.float32)
        ort_out = ort_sess.run(None, {'images': inp})[0]
        with torch.no_grad():
            th_out = torch_model(torch.from_numpy(inp)).numpy()
        err = float(np.max(np.abs(ort_out - th_out)))
        if err > max_err:
            max_err = err
    print(f"Max error against ONNXRuntime: {max_err:.8e}")

    print("Tracing TorchScript on CPU (device-agnostic)...")
    dummy = torch.zeros(1, 3, 24, 40, dtype=torch.float32)
    traced_model = torch.jit.trace(torch_model, dummy)

    os.makedirs(os.path.dirname(os.path.abspath(ts_path)), exist_ok=True)
    traced_model.save(ts_path)
    print(f"Successfully saved TorchScript to {ts_path}")

    # Verification on CPU and all available GPUs
    print("Testing reloaded model on CPU...")
    loaded_cpu = torch.jit.load(ts_path, map_location="cpu")
    out_cpu = loaded_cpu(dummy)
    print(f"CPU reload success. Output: {out_cpu.item():.4f}")

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} CUDA GPU(s). Testing device portability...")
    for gid in range(num_gpus):
        dev = torch.device(f"cuda:{gid}")
        loaded_gpu = torch.jit.load(ts_path, map_location=dev)
        out_gpu = loaded_gpu(dummy.to(dev))
        print(f"CUDA:{gid} reload success. Output: {out_gpu.cpu().item():.4f}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    onnx_file = os.path.join(project_root, "bin/x86_64/models/ocec_l.onnx")
    ts_file = os.path.join(project_root, "bin/x86_64/models/ocec_l.torchscript")

    if len(sys.argv) > 1:
        onnx_file = sys.argv[1]
    if len(sys.argv) > 2:
        ts_file = sys.argv[2]

    convert_ocec_onnx_to_torchscript(onnx_file, ts_file)
