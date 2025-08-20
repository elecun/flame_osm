import torch
import torch.onnx
import os
import numpy as np
from typing import Tuple, Optional
import logging

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Install TensorRT for engine export functionality.")


class ModelExporter:
    """Export trained ST-GCN model to various formats"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def export_pytorch(self, output_path: str, input_shape: Tuple[int, ...]):
        """Export model as PyTorch .pt file"""
        print(f"Exporting PyTorch model to {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Test model with dummy input
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Save model state dict
        torch.save(self.model.state_dict(), output_path)
        print(f"PyTorch model saved to {output_path}")
        
        # Also save complete model for easier loading
        complete_path = output_path.replace('.pt', '_complete.pt')
        torch.save(self.model, complete_path)
        print(f"Complete PyTorch model saved to {complete_path}")
        
    def export_onnx(self, output_path: str, input_shape: Tuple[int, ...], 
                   dynamic_axes: Optional[dict] = None):
        """Export model to ONNX format"""
        print(f"Exporting ONNX model to {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"ONNX model saved to {output_path}")
        
    def export_tensorrt(self, onnx_path: str, output_path: str, 
                       max_batch_size: int = 16, precision: str = 'fp16'):
        """Export ONNX model to TensorRT engine"""
        if not TRT_AVAILABLE:
            print("TensorRT not available. Skipping TensorRT export.")
            return False
            
        print(f"Converting ONNX model to TensorRT engine: {output_path}")
        
        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Set precision
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
        else:
            print("Using FP32 precision")
        
        # Build engine
        print("Building TensorRT engine... This may take a while.")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return False
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {output_path}")
        return True
        
    def benchmark_tensorrt(self, engine_path: str, input_shape: Tuple[int, ...], 
                          num_iterations: int = 100):
        """Benchmark TensorRT engine performance"""
        if not TRT_AVAILABLE:
            print("TensorRT not available for benchmarking")
            return None
            
        print(f"Benchmarking TensorRT engine: {engine_path}")
        
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = input_size  # Assuming similar output size
        
        h_input = cuda.pagelocked_empty(np.prod(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(np.prod(input_shape), dtype=np.float32)
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / avg_time
        
        benchmark_results = {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': fps,
            'input_shape': input_shape
        }
        
        print(f"TensorRT Benchmark Results:")
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Standard deviation: {std_time:.2f} ms")
        print(f"FPS: {fps:.2f}")
        
        return benchmark_results


def export_model(model, input_shape: Tuple[int, ...], output_dir: str, 
                device: str = 'cuda', export_formats: list = None):
    """Export model to multiple formats"""
    
    if export_formats is None:
        export_formats = ['pytorch', 'onnx', 'tensorrt']
    
    os.makedirs(output_dir, exist_ok=True)
    exporter = ModelExporter(model, device)
    
    results = {}
    
    # Export PyTorch model
    if 'pytorch' in export_formats:
        pt_path = os.path.join(output_dir, 'stgcn_model.pt')
        exporter.export_pytorch(pt_path, input_shape)
        results['pytorch'] = pt_path
    
    # Export ONNX model
    onnx_path = None
    if 'onnx' in export_formats or 'tensorrt' in export_formats:
        onnx_path = os.path.join(output_dir, 'stgcn_model.onnx')
        exporter.export_onnx(onnx_path, input_shape)
        results['onnx'] = onnx_path
    
    # Export TensorRT engine
    if 'tensorrt' in export_formats and onnx_path:
        engine_path = os.path.join(output_dir, 'stgcn_model.engine')
        success = exporter.export_tensorrt(onnx_path, engine_path)
        if success:
            results['tensorrt'] = engine_path
            
            # Benchmark TensorRT performance
            benchmark_results = exporter.benchmark_tensorrt(engine_path, input_shape)
            if benchmark_results:
                results['tensorrt_benchmark'] = benchmark_results
    
    # Save export summary
    export_summary = {
        'input_shape': input_shape,
        'exported_files': results,
        'device': device
    }
    
    summary_path = os.path.join(output_dir, 'export_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(export_summary, f, indent=2)
    
    print(f"\nExport completed! Summary saved to {summary_path}")
    return results


class TensorRTInference:
    """TensorRT inference wrapper for deployed models"""
    
    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
            
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output shapes
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        
        # Allocate buffers
        self.input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = np.prod(self.output_shape) * np.dtype(np.float32).itemsize
        
        self.h_input = cuda.pagelocked_empty(np.prod(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(np.prod(self.output_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        self.stream = cuda.Stream()
        
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data"""
        # Copy input data
        np.copyto(self.h_input, input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)], 
            stream_handle=self.stream.handle
        )
        
        # Transfer output data to CPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        # Reshape output
        output = self.h_output.reshape(self.output_shape)
        return output
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()


if __name__ == "__main__":
    # Example usage
    from stgcn_model import STGCN
    
    # Create model
    graph_args = {'layout': 'body_head', 'strategy': 'spatial'}
    model = STGCN(
        in_channels=3,
        num_class=1,
        graph_args=graph_args,
        edge_importance_weighting=True
    )
    
    # Export model
    input_shape = (1, 3, 64, 20, 1)  # (batch, channels, time, vertices, persons)
    export_model(model, input_shape, 'exported_models')
