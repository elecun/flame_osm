# Head Pose Estimation using Google Mediapipe Face Mesh + PnP
* ref : https://github.com/google-ai-edge/mediapipe

1. Build Mediapipe
```
$ git clone https://github.com/google/mediapipe.git
$ cd mediapipe

bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_gpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt
```

2. 