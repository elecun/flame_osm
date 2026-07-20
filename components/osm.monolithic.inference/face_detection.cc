#include "face_detection.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

face_detection::face_detection() {}
face_detection::~face_detection() {}

bool face_detection::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    try {
        std::string path = model_path;

        if (!fs::exists(path)) {
            logger::error("[FaceDetection] Model file not found: {}", path);
            return false;
        }

        _net = cv::dnn::readNet(path);
        
        // Set CPU backend for portability and to avoid compile-time/run-time CUDA mismatches on host
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        logger::info("[FaceDetection] Loaded OpenCV DNN model successfully from {}", path);
        return true;
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during model load: {}", e.what());
        return false;
    }
}

std::vector<cv::Rect> face_detection::process(const cv::Mat& image) {
    std::vector<cv::Rect> bboxes;
    if (image.empty()) {
        logger::warn("[FaceDetection] Input image is empty");
        return bboxes;
    }

    try {
        // Preprocess: scale to [0, 1], resize to 640x640, BGR to RGB
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(_input_width, _input_height), cv::Scalar(), true, false);
        _net.setInput(blob);

        // Run forward pass
        cv::Mat output = _net.forward(); // shape: [1, 6, 8400]

        // Parse outputs
        int rows = output.size[1]; // 6
        int cols = output.size[2]; // 8400

        cv::Mat output_reshaped = output.reshape(1, rows); // [6, 8400]
        cv::Mat output_transposed;
        cv::transpose(output_reshaped, output_transposed); // [8400, 6]

        const float conf_threshold = 0.5f;
        float best_score = 0.0f;
        int best_index = -1;

        for (int i = 0; i < cols; i++) {
            float obj_conf = output_transposed.at<float>(i, 4);
            if (obj_conf > conf_threshold) {
                float w = output_transposed.at<float>(i, 2);
                float h = output_transposed.at<float>(i, 3);
                float score = obj_conf * (w * h);
                if (score > best_score) {
                    best_score = score;
                    best_index = i;
                }
            }
        }

        if (best_index >= 0) {
            float xc = output_transposed.at<float>(best_index, 0);
            float yc = output_transposed.at<float>(best_index, 1);
            float w = output_transposed.at<float>(best_index, 2);
            float h = output_transposed.at<float>(best_index, 3);

            // Scale back to original image coordinates
            float scale_x = (float)image.cols / _input_width;
            float scale_y = (float)image.rows / _input_height;

            float x1 = (xc - w / 2.0f) * scale_x;
            float y1 = (yc - h / 2.0f) * scale_y;
            float width = w * scale_x;
            float height = h * scale_y;

            cv::Rect bbox(
                std::max(0.0f, x1),
                std::max(0.0f, y1),
                std::max(0.0f, width),
                std::max(0.0f, height)
            );

            bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
            if (bbox.width > 0 && bbox.height > 0) {
                bboxes.push_back(bbox);
            }
        }
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during process: {}", e.what());
    }

    return bboxes;
}
