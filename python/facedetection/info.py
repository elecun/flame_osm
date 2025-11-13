import onnx

# 1. 모델 로드
model = onnx.load("model_gpu.onnx")

# 2. 그래프 입력/출력 정보 확인
print("=== Inputs ===")
for input_tensor in model.graph.input:
    name = input_tensor.name
    shape = [dim.dim_value if (dim.dim_value > 0) else "?" 
             for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"{name}: {shape}")

print("\n=== Outputs ===")
for output_tensor in model.graph.output:
    name = output_tensor.name
    shape = [dim.dim_value if (dim.dim_value > 0) else "?" 
             for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"{name}: {shape}")