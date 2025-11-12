


(1) Pose Estimation Model
# Download YoloV11 Pose Estimation Pretrained Models (*.pt)

https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt

# Convert to ONNX

* TensorRT Version : 10.14.1.48-1+cuda13.0
* requirements
```
$ python -m venv venv
$ pip install ultralytics onnx onnxruntime onnxsim
$ yolo export model=yolo11x-pose.pt format=onnx dynamic=True opset=20 nms=True device=0
```

* check ONNX model file (in python)
```
import onnx
onnx_model = onnx.load("yolo11x-pose.onnx")
onnx.checker.check_model(onnx_model)
print("OK")
```

# Conver ONNX to TensorRT
```
$ /usr/src/tensorrt/bin/trtexec --onnx=yolo11x-pose.onnx --saveEngine=yolo11x-pose.engine --fp16 --minShapes="images:1x3x640x640" --optShapes="images:1x3x1088x1920" --maxShapes="images:1x3x1440x2560" 


[11/12/2025-20:00:02] [I] TensorRT version: 10.14.1
[11/12/2025-20:00:02] [I] Loading standard plugins
[11/12/2025-20:00:02] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 29, GPU 616 (MiB)
[11/12/2025-20:00:02] [I] Start parsing network model.
[11/12/2025-20:00:02] [I] [TRT] ----------------------------------------------------------------
[11/12/2025-20:00:02] [I] [TRT] Input filename:   yolo11x-pose.onnx
[11/12/2025-20:00:02] [I] [TRT] ONNX IR version:  0.0.9
[11/12/2025-20:00:02] [I] [TRT] Opset version:    20
[11/12/2025-20:00:02] [I] [TRT] Producer name:    pytorch
[11/12/2025-20:00:02] [I] [TRT] Producer version: 2.9.0
[11/12/2025-20:00:02] [I] [TRT] Domain:           
[11/12/2025-20:00:02] [I] [TRT] Model version:    0
[11/12/2025-20:00:02] [I] [TRT] Doc string:       
[11/12/2025-20:00:02] [I] [TRT] ----------------------------------------------------------------


```
* Note! Yolo use stride=32, height and width must be multiple of 32