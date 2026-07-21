#include "body_pose_estimation.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <algorithm>
namespace fs = std::filesystem;

body_pose_estimation::body_pose_estimation() {}
body_pose_estimation::~body_pose_estimation() {}

bool body_pose_estimation::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    try {
        std::string path = model_path;

        if (!fs::exists(path)) {
            logger::error("[BodyPoseEstimation] Model file not found: {}", path);
            return false;
        }

        // Set device (GPU or CPU)
        if (torch::cuda::is_available() && gpu_id >= 0) {
            _device = torch::Device(torch::kCUDA, gpu_id);
            logger::info("[BodyPoseEstimation] CUDA is available. Using GPU: {}", gpu_id);
        } else {
            _device = torch::Device(torch::kCPU);
            logger::warn("[BodyPoseEstimation] CUDA is not available. Using CPU");
        }

        // Load the TorchScript module
        _module = torch::jit::load(path);
        _module.to(_device);
        _module.eval(); // set to evaluation mode

        logger::info("[BodyPoseEstimation] Loaded TorchScript model successfully from {}", path);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[BodyPoseEstimation] Failed to load TorchScript model: {}", e.what());
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[BodyPoseEstimation] Exception during model load: {}", e.what());
        return false;
    }
}

std::vector<body_pose::PoseResult> body_pose_estimation::process(const cv::Mat& image, float conf_threshold, float nms_threshold) {
    std::vector<body_pose::PoseResult> poses;
    if (image.empty()) {
        logger::warn("[BodyPoseEstimation] Input image is empty");
        return poses;
    }

    try {
        // 1. Preprocess using OpenCV: Resize and BGR to RGB conversion
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(_input_width, _input_height));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // Convert data type to float and scale to [0, 1]
        cv::Mat float_image;
        resized.convertTo(float_image, CV_32FC3, 1.0f / 255.0f);

        // 2. Create Torch Tensor from OpenCV Mat
        auto input_tensor = torch::from_blob(float_image.data, {1, _input_height, _input_width, 3}, torch::kFloat32);

        // Convert BHWC layout to BCHW (necessary for YOLO PyTorch model)
        input_tensor = input_tensor.permute({0, 3, 1, 2});

        // Move to target device (GPU or CPU)
        input_tensor = input_tensor.to(_device);

        // 3. Run model inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Disable gradient calculations during inference
        torch::NoGradGuard no_grad;
        auto outputs = _module.forward(inputs);

        // 4. Handle output formats
        torch::Tensor output_tensor;
        if (outputs.isTensor()) {
            output_tensor = outputs.toTensor();
        } else if (outputs.isTuple()) {
            output_tensor = outputs.toTuple()->elements()[0].toTensor();
        } else {
            logger::error("[BodyPoseEstimation] Unexpected model output format");
            return poses;
        }

        // Output shape expected: [1, 300, 57]
        output_tensor = output_tensor.to(torch::kCPU);
        output_tensor = output_tensor.squeeze(0);       // shape: [300, 57]

        int cols = output_tensor.size(0);     // 300
        int channels = output_tensor.size(1); // 57

        output_tensor = output_tensor.contiguous();   // shape: [300, 57]
        float* data = output_tensor.data_ptr<float>();

        std::vector<cv::Rect> candidate_boxes;
        std::vector<float> confidences;
        std::vector<int> box_indices;

        for (int i = 0; i < cols; i++) {
            float obj_conf = data[i * channels + 4];
            if (obj_conf > conf_threshold) {
                float xc = data[i * channels + 0];
                float yc = data[i * channels + 1];
                float w = data[i * channels + 2];
                float h = data[i * channels + 3];

                // Scale bounding box back to original image size
                float scale_x = (float)image.cols / _input_width;
                float scale_y = (float)image.rows / _input_height;

                float x1 = (xc - w / 2.0f) * scale_x;
                float y1 = (yc - h / 2.0f) * scale_y;
                float bw = w * scale_x;
                float bh = h * scale_y;

                cv::Rect bbox(
                    std::max(0.0f, x1),
                    std::max(0.0f, y1),
                    std::max(0.0f, bw),
                    std::max(0.0f, bh)
                );

                bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
                if (bbox.width > 0 && bbox.height > 0) {
                    candidate_boxes.push_back(bbox);
                    confidences.push_back(obj_conf);
                    box_indices.push_back(i);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(candidate_boxes, confidences, conf_threshold, nms_threshold, indices);

        for (int idx : indices) {
            body_pose::PoseResult pose;
            pose.bbox = candidate_boxes[idx];
            pose.bbox_confidence = confidences[idx];

            int raw_box_idx = box_indices[idx];
            float* box_data = data + raw_box_idx * channels;

            float scale_x = (float)image.cols / _input_width;
            float scale_y = (float)image.rows / _input_height;

            for (int k = 0; k < _num_keypoints; k++) {
                int kpt_offset = 5 + k * 3;
                float kpt_conf = box_data[kpt_offset + 0];
                float kpt_x = box_data[kpt_offset + 1] * scale_x;
                float kpt_y = box_data[kpt_offset + 2] * scale_y;

                body_pose::KeyPoint kpt;
                kpt.x = kpt_x;
                kpt.y = kpt_y;
                kpt.confidence = kpt_conf;
                pose.keypoints.push_back(kpt);
            }

            poses.push_back(pose);
        }
    }
    catch (const c10::Error& e) {
        logger::error("[BodyPoseEstimation] LibTorch error during process: {}", e.what());
    }
    catch (const std::exception& e) {
        logger::error("[BodyPoseEstimation] Exception during process: {}", e.what());
    }

    return poses;
}
