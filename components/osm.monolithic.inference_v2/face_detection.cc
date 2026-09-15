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

        // Set device (GPU or CPU)
        if (torch::cuda::is_available() && gpu_id >= 0) {
            _device = torch::Device(torch::kCUDA, gpu_id);
            logger::info("[FaceDetection] CUDA is available. Using GPU: {}", gpu_id);
        } else {
            _device = torch::Device(torch::kCPU);
            logger::warn("[FaceDetection] CUDA is not available. Using CPU");
        }

        // Load the TorchScript module
        _module = torch::jit::load(path);
        _module.to(_device);
        _module.eval(); // set to evaluation mode

        logger::info("[FaceDetection] Loaded TorchScript model successfully from {}", path);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[FaceDetection] Failed to load TorchScript model: {}", e.what());
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during model load: {}", e.what());
        return false;
    }
}

cv::Mat face_detection::letterbox(const cv::Mat& img, int new_shape, float& out_ratio, float& out_dw, float& out_dh) {
    int h0 = img.rows;
    int w0 = img.cols;
    float r = std::min((float)new_shape / h0, (float)new_shape / w0);
    int new_unpad_w = static_cast<int>(std::round(w0 * r));
    int new_unpad_h = static_cast<int>(std::round(h0 * r));

    float dw = (new_shape - new_unpad_w) / 2.0f;
    float dh = (new_shape - new_unpad_h) / 2.0f;

    cv::Mat resized;
    if (w0 != new_unpad_w || h0 != new_unpad_h) {
        cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img.clone();
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat img_lb;
    cv::copyMakeBorder(resized, img_lb, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    out_ratio = r;
    out_dw = dw;
    out_dh = dh;
    return img_lb;
}

std::vector<FaceBox> face_detection::detect(const cv::Mat& image, float conf_thresh, float nms_thresh, float padding_scale) {
    std::vector<FaceBox> detected_faces;
    if (image.empty()) {
        logger::warn("[FaceDetection] Input image is empty");
        return detected_faces;
    }

    try {
        int w0 = image.cols;
        int h0 = image.rows;

        // 1. Letterbox preserving aspect ratio
        float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
        cv::Mat img_lb = letterbox(image, _input_width, ratio, dw, dh);

        // Convert BGR to RGB
        cv::Mat rgb_lb;
        cv::cvtColor(img_lb, rgb_lb, cv::COLOR_BGR2RGB);

        // Scale to [0, 1]
        cv::Mat float_image;
        rgb_lb.convertTo(float_image, CV_32FC3, 1.0f / 255.0f);

        // 2. Create Torch Tensor: [1, 3, 640, 640]
        auto input_tensor = torch::from_blob(float_image.data, {1, _input_height, _input_width, 3}, torch::kFloat32);
        input_tensor = input_tensor.permute({0, 3, 1, 2}).to(_device);

        // 3. Inference
        torch::Tensor preds;
        {
            torch::NoGradGuard no_grad;
            auto outputs = _module.forward({input_tensor});
            if (outputs.isTensor()) {
                preds = outputs.toTensor();
            } else if (outputs.isTuple()) {
                preds = outputs.toTuple()->elements()[0].toTensor();
            } else {
                logger::error("[FaceDetection] Unexpected model output format");
                return detected_faces;
            }
        }

        // preds shape: [1, 5, 8400] -> [8400, 5]
        preds = preds.squeeze(0).transpose(0, 1).contiguous().to(torch::kCPU);
        int channels = preds.size(1);
        int num_candidates = preds.size(0);
        float* data = preds.data_ptr<float>();

        std::vector<cv::Rect2d> candidate_boxes;
        std::vector<float> confidences;

        for (int i = 0; i < num_candidates; ++i) {
            float score = data[i * channels + 4];
            if (score >= conf_thresh) {
                float cx = data[i * channels + 0];
                float cy = data[i * channels + 1];
                float w = data[i * channels + 2];
                float h = data[i * channels + 3];

                // Scale back to original coordinates
                float x1 = (cx - w / 2.0f - dw) / ratio;
                float y1 = (cy - h / 2.0f - dh) / ratio;
                float x2 = (cx + w / 2.0f - dw) / ratio;
                float y2 = (cy + h / 2.0f - dh) / ratio;

                float bw = std::max(0.0f, x2 - x1);
                float bh = std::max(0.0f, y2 - y1);
                candidate_boxes.emplace_back(x1, y1, bw, bh);
                confidences.push_back(score);
            }
        }

        if (candidate_boxes.empty()) {
            return detected_faces;
        }

        // 4. NMS
        std::vector<int> keep;
        cv::dnn::NMSBoxes(candidate_boxes, confidences, conf_thresh, nms_thresh, keep);

        // 5. Expand each kept bounding box to a symmetric square head box (1:1 aspect ratio)
        for (int idx : keep) {
            const auto& box = candidate_boxes[idx];
            float score = confidences[idx];

            double bx1 = box.x;
            double by1 = box.y;
            double bx2 = box.x + box.width;
            double by2 = box.y + box.height;
            double bw = bx2 - bx1;
            double bh = by2 - by1;
            double cx = bx1 + bw / 2.0;
            double cy = by1 + bh / 2.0;

            // 1:1 Aspect ratio square crop with padding centered at (cx, cy)
            double max_side = std::max(bw, bh) * padding_scale;
            int px1 = std::max(0, static_cast<int>(cx - max_side / 2.0));
            int py1 = std::max(0, static_cast<int>(cy - max_side / 2.0));
            int px2 = std::min(w0, static_cast<int>(cx + max_side / 2.0));
            int py2 = std::min(h0, static_cast<int>(cy + max_side / 2.0));

            int pw = std::max(1, px2 - px1);
            int ph = std::max(1, py2 - py1);

            FaceBox fb;
            fb.bbox = cv::Rect(px1, py1, pw, ph);
            fb.score = score;
            fb.raw_bbox = cv::Rect(static_cast<int>(bx1), static_cast<int>(by1), static_cast<int>(bw), static_cast<int>(bh));
            detected_faces.push_back(fb);
        }

        // Sort detected faces by score descending
        std::sort(detected_faces.begin(), detected_faces.end(), [](const FaceBox& a, const FaceBox& b) {
            return a.score > b.score;
        });
    }
    catch (const c10::Error& e) {
        logger::error("[FaceDetection] LibTorch error during detect: {}", e.what());
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during detect: {}", e.what());
    }

    return detected_faces;
}
