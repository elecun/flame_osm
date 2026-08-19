#ifndef OSM_MONOLITHIC_INFERENCE_BODY_POSE_ESTIMATION_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_BODY_POSE_ESTIMATION_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

namespace body_pose {
    struct KeyPoint {
        float x;
        float y;
        float confidence;
    };
    struct PoseResult {
        cv::Rect bbox;
        float bbox_confidence;
        std::vector<KeyPoint> keypoints;
    };
}

class body_pose_estimation {
public:
    body_pose_estimation();
    ~body_pose_estimation();

    bool loadModel(const std::string& model_path, int gpu_id = 0);
    std::vector<body_pose::PoseResult> process(const cv::Mat& image, float conf_threshold = 0.5f, float nms_threshold = 0.45f);

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _input_width = 640;
    int _input_height = 640;
    int _num_keypoints = 17;
    int _gpu_id = 0;
};

#endif
