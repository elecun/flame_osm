#include "driver_readiness_estimation.hpp"
#include <flame/log.hpp>
#include <iostream>

driver_readiness_estimation::driver_readiness_estimation() {
}

driver_readiness_estimation::~driver_readiness_estimation() {
}

bool driver_readiness_estimation::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    if (torch::cuda::is_available() && _gpu_id >= 0) {
        _device = torch::Device(torch::kCUDA, _gpu_id);
    } else {
        _device = torch::Device(torch::kCPU);
    }

    try {
        _module = torch::jit::load(model_path, _device);
        _module.eval();
        _is_loaded = true;
        logger::info("[driver_readiness_estimation] Successfully loaded model from {} on device: {}", 
                     model_path, _device.str());
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[driver_readiness_estimation] Error loading model from {}: {}", model_path, e.what());
        _is_loaded = false;
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[driver_readiness_estimation] Exception loading model from {}: {}", model_path, e.what());
        _is_loaded = false;
        return false;
    }
}

driver_readiness::FrameInput driver_readiness_estimation::extractFeatures(
    const std::vector<body_pose::PoseResult>& body_poses,
    const head_pose::PoseResult& head_pose_res,
    bool has_valid_head_pose
) {
    driver_readiness::FrameInput frame;
    frame.body_kps_22.assign(BODY_FEAT_DIM, 0.0f);
    frame.head_pose_3.assign(HEAD_FEAT_DIM, 0.0f);

    // 1. Extract 11 Upper body keypoints (indices 0 to 10)
    if (!body_poses.empty() && body_poses[0].keypoints.size() >= 11) {
        const auto& kps = body_poses[0].keypoints;
        for (size_t i = 0; i < 11; ++i) {
            frame.body_kps_22[i * 2 + 0] = kps[i].x;
            frame.body_kps_22[i * 2 + 1] = kps[i].y;
        }
        frame.is_valid = true;
    }

    // 2. Extract Head Pose Euler Angles (pitch, yaw, roll)
    if (has_valid_head_pose && head_pose_res.success) {
        frame.head_pose_3[0] = static_cast<float>(head_pose_res.euler[0]); // pitch
        frame.head_pose_3[1] = static_cast<float>(head_pose_res.euler[1]); // yaw
        frame.head_pose_3[2] = static_cast<float>(head_pose_res.euler[2]); // roll
        frame.is_valid = true;
    }

    return frame;
}

driver_readiness::ReadinessResult driver_readiness_estimation::process(
    const std::vector<body_pose::PoseResult>& body_poses,
    const head_pose::PoseResult& head_pose_res,
    bool has_valid_head_pose
) {
    driver_readiness::ReadinessResult result;
    if (!_is_loaded) {
        return result;
    }

    // Extract current frame feature
    driver_readiness::FrameInput current_frame = extractFeatures(body_poses, head_pose_res, has_valid_head_pose);

    // Queue management (Maintain fixed size SEQ_LEN = 64)
    _sequence_queue.push_back(current_frame);
    if (_sequence_queue.size() > SEQ_LEN) {
        _sequence_queue.pop_front();
    }

    // Wait until queue is filled with 64 frames
    if (_sequence_queue.size() < SEQ_LEN) {
        return result;
    }

    try {
        // Construct Tensor buffers: [1, 64, 22] for body, [1, 64, 0] for face (unused), [1, 64, 3] for head
        std::vector<float> body_flat;
        std::vector<float> head_flat;
        body_flat.reserve(SEQ_LEN * BODY_FEAT_DIM);
        head_flat.reserve(SEQ_LEN * HEAD_FEAT_DIM);

        for (const auto& item : _sequence_queue) {
            body_flat.insert(body_flat.end(), item.body_kps_22.begin(), item.body_kps_22.end());
            head_flat.insert(head_flat.end(), item.head_pose_3.begin(), item.head_pose_3.end());
        }

        torch::Tensor body_tensor = torch::from_blob(body_flat.data(), {1, (long)SEQ_LEN, (long)BODY_FEAT_DIM}, torch::kFloat32).clone().to(_device);
        torch::Tensor face_tensor = torch::zeros({1, (long)SEQ_LEN, 0}, torch::TensorOptions().dtype(torch::kFloat32).device(_device));
        torch::Tensor head_tensor = torch::from_blob(head_flat.data(), {1, (long)SEQ_LEN, (long)HEAD_FEAT_DIM}, torch::kFloat32).clone().to(_device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(body_tensor);
        inputs.push_back(face_tensor);
        inputs.push_back(head_tensor);

        c10::IValue output_ivalue = _module.forward(inputs);
        torch::Tensor output_tensor = output_ivalue.toTensor().to(torch::kCPU);

        // Apply Softmax to get probabilities
        torch::Tensor probs = torch::softmax(output_tensor, 1);
        auto accessor = probs.accessor<float, 2>();

        int num_classes = probs.size(1);
        result.probabilities.resize(num_classes);
        int best_cls = 0;
        float max_p = -1.0f;

        for (int c = 0; c < num_classes; ++c) {
            float p = accessor[0][c];
            result.probabilities[c] = p;
            if (p > max_p) {
                max_p = p;
                best_cls = c;
            }
        }

        result.predicted_class = best_cls;
        result.confidence = max_p;
        result.is_ready = true;
    }
    catch (const std::exception& e) {
        logger::error("[driver_readiness_estimation] Inference exception: {}", e.what());
        result.is_ready = false;
    }

    return result;
}

void driver_readiness_estimation::drawResult(cv::Mat& image, const driver_readiness::ReadinessResult& result) {
    if (!result.is_ready) {
        std::string text = "Readiness: Buffering (" + std::to_string(_sequence_queue.size()) + "/" + std::to_string(SEQ_LEN) + ")";
        cv::putText(image, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        return;
    }

    std::string text = "Readiness Class: " + std::to_string(result.predicted_class) +
                       " (" + std::to_string(static_cast<int>(result.confidence * 100.0f)) + "%)";
    cv::putText(image, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}
