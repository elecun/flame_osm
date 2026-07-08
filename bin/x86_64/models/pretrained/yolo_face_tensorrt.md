# YOLO11n-face 모델의 tensorrt 변환

## 관련 패키지 설치 (가상환경)
```
$ pip install ultralytics onnx
```

## 모델 변환 (*.pt to onnx)
```
$ yolo export model=yolo11n-face.pt format=onnx dynamic=True opset=20 nms=True device=0
```
- 출력 결과
Ultralytics 8.4.90 🚀 Python-3.10.12 torch-2.12.1+cu130 CUDA:0 (NVIDIA GeForce RTX 4080 SUPER, 15944MiB)
WARNING ⚠️ 'dynamic=True' model with 'nms=True' requires max batch size, i.e. 'batch=16'
YOLO11n summary (fused): 100 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs

PyTorch: starting from 'yolo11n-face.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.2 MB)

ONNX: starting export with onnx 1.22.0 opset 20...
ONNX: slimming with onnxslim 0.1.94...
ONNX: export success ✅ 1.3s, saved as 'yolo11n-face.onnx' (10.5 MB)

Export complete (2.2s)
Results saved to /home/iae-vc/dev/flame_osm/bin/x86_64/models/pretrained/yolo11n-face.onnx
Predict:         yolo predict task=detect model=yolo11n-face.onnx imgsz=640 
Validate:        yolo val task=detect model=yolo11n-face.onnx imgsz=640 data=datasets/widerface.yaml  
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export

# 모델 변환 (onnx to tensorrt)
- [주의] tensorrt로 변환할때 TensorRT가 모델 내부의 Concat(결합) 레이어에서 차원(Dimension) 불일치 오류가 발생할 수있어서, 실제 해상도는 1080x1920이지만, 32의 배수인 1088로 해야 에러가 나지않음에 유의.
```
trtexec --onnx=yolo11n-face.onnx --saveEngine=yolo11n-face.engine --fp16 --minShapes="images:1x3x640x640" --optShapes="images:1x3x1088x1920" --maxShapes="images:1x3x1440x2560" 
```