import torch
import torch.onnx

# 1. PyTorch 모델 로드
model = torch.load("model.pt", map_location="cpu")  # 또는 model = MyModel(); model.load_state_dict(...)
print(model)
model.eval()


# 2. 더미 입력 (모델의 입력 크기에 맞게)
dummy_input = torch.randn(1, 3, 512, 512)

# 3. ONNX로 변환
torch.onnx.export(
    model,                          # 변환할 모델
    dummy_input,                    # 더미 입력
    "model.onnx",                   # 출력 파일 이름
    input_names=["input"],           # 입력 텐서 이름
    output_names=["output"],         # 출력 텐서 이름
    opset_version=18,                # ONNX opset 버전 (추천: 17~19)
    do_constant_folding=True,        # 상수 폴딩 최적화
    dynamic_axes=None                # None이면 고정 shape로 export
)

print("model.onnx 파일이 생성되었습니다.")