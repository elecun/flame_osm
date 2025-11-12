import onnx
onnx_model = onnx.load("yolo11x-pose.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 모델이 정상적으로 로드되었습니다.")