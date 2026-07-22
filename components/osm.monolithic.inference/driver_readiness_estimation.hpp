#ifndef OSM_MONOLITHIC_INFERENCE_DRIVER_READINESS_ESTIMATION_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_DRIVER_READINESS_ESTIMATION_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <deque>
#include "body_pose_estimation.hpp"
#include "head_pose_estimation_from_2d.hpp"

namespace driver_readiness {
    struct FrameInput {
        // 11 upper body keypoints (2D coordinates x, y) -> size 22
        std::vector<float> body_kps_22;
        // Head pose Euler angles (pitch, yaw, roll) -> size 3
        std::vector<float> head_pose_3;
        bool is_valid = false;
    };

    struct ReadinessResult {
        std::vector<float> probabilities; // 5 classes
        int predicted_class = -1;
        float confidence = 0.0f;
        bool is_ready = false;
    };
}

class driver_readiness_estimation {
public:
    driver_readiness_estimation();
    ~driver_readiness_estimation();

    bool loadModel(const std::string& model_path, int gpu_id = 1);
    
    // Add current frame features to queue and run inference if queue is full (seq_len = 64)
    driver_readiness::ReadinessResult process(
        const std::vector<body_pose::PoseResult>& body_poses,
        const head_pose::PoseResult& head_pose_res,
        bool has_valid_head_pose
    );

    void drawResult(cv::Mat& image, const driver_readiness::ReadinessResult& result);

private:
    driver_readiness::FrameInput extractFeatures(
        const std::vector<body_pose::PoseResult>& body_poses,
        const head_pose::PoseResult& head_pose_res,
        bool has_valid_head_pose
    );

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _gpu_id = 1;
    bool _is_loaded = false;

    // Sequence queue parameters
    static constexpr size_t SEQ_LEN = 64;
    static constexpr size_t BODY_FEAT_DIM = 22; // 11 keypoints * 2
    static constexpr size_t HEAD_FEAT_DIM = 3;  // pitch, yaw, roll

    std::deque<driver_readiness::FrameInput> _sequence_queue;
};

#endif
