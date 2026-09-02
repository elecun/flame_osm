# Blink Detection Inference Component

## Description
This component performs real-time driver eye blink detection and drowsiness metric (PERCLOS) calculation using the `BlinkLinMulT` deep learning sequence model with LibTorch on CUDA/CPU.

- **Model**: `blinklinmult-union.torchscript`
- **Sequence Length**: 15 frames temporal window
- **Eye Patch Resolution**: 64 x 64 pixels (ImageNet normalized)
- **High-Level Features**: 160 dimensions (headpose, landmarks, iris, EAR)

## Data Ports

### Input
- `image_stream_1` / `image_stream_2` (or configured `input_port`): Camera or video frames (cv::Mat / JPEG).

### Output
- `blink_result` (or configured `output_data_port`): JSON metadata containing `blink_prob`, `is_blinking`, `blink_count`, `perclos`, `fps`.
- `blink_monitor` / `image_stream_1_processed_monitor` (or configured `output_monitor_port`): Visualized JPEG stream showing face/eye bounding boxes, blink state badge, probability, PERCLOS, and FPS.

## Parameters
```json
{
  "model_path": "bin/x86_64/models/blinklinmult-union.torchscript",
  "gpu_id": 0,
  "seq_len": 15,
  "crop_width": 64,
  "crop_height": 64,
  "threshold": 0.5,
  "eye_selection": "left",
  "show_info": true,
  "visualize": true,
  "input_port": "image_stream_1",
  "output_data_port": "blink_result",
  "output_monitor_port": "image_stream_1_processed_monitor"
}
```
