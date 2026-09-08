/**
 * @file blink_analysis.cc
 * @brief Blink Analysis module implementation for BlinkLinMulT TorchScript model
 * @details Processes eye patches through a temporal sliding window and runs
 *          the BlinkLinMulT model for blink detection, PERCLOS computation.
 */

#include "blink_analysis.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;
using json = nlohmann::json;

blink_analysis_component::blink_analysis_component() = default;
blink_analysis_component::~blink_analysis_component() = default;

bool blink_analysis_component::init(const json& params) {
    try {
        model_path_ = params.value("model_path", model_path_);
        gpu_id_     = params.value("gpu_id", gpu_id_);
        seq_len_    = params.value("seq_len", seq_len_);
        crop_w_     = params.value("crop_width", crop_w_);
        crop_h_     = params.value("crop_height", crop_h_);
        threshold_  = params.value("threshold", threshold_);

        /* Search for model file if path does not exist */
        if (!fs::exists(model_path_)) {
            std::vector<std::string> candidates = {
                "bin/x86_64/models/blinklinmult-union.torchscript",
                "models/blinklinmult-union.torchscript",
                "/home/iae-vc/dev/flame_osm/bin/x86_64/models/blinklinmult-union.torchscript"
            };
            for (const auto& cand : candidates) {
                if (fs::exists(cand)) {
                    logger::info("[blink_analysis] Model path '{}' not found, using: '{}'", model_path_, cand);
                    model_path_ = cand;
                    break;
                }
            }
        }

        if (!fs::exists(model_path_)) {
            logger::error("[blink_analysis] Model file not found: {}", model_path_);
            return false;
        }

        /* Setup device */
        if (torch::cuda::is_available() && gpu_id_ >= 0) {
            device_ = torch::Device(torch::kCUDA, gpu_id_);
            logger::info("[blink_analysis] Using CUDA GPU device: {}", gpu_id_);
        } else {
            device_ = torch::Device(torch::kCPU);
            logger::info("[blink_analysis] Using CPU device");
        }

        /* Load TorchScript model */
        module_ = torch::jit::load(model_path_, device_);
        module_.eval();

        /* Warmup forward pass with dummy data */
        {
            torch::NoGradGuard no_grad;
            auto dummy_low  = torch::zeros({1, seq_len_, 3, crop_h_, crop_w_}, device_);
            auto dummy_high = torch::zeros({1, seq_len_, 160}, device_);
            std::vector<torch::jit::IValue> inputs = {dummy_low, dummy_high};
            module_.forward(inputs);
        }

        model_loaded_ = true;
        logger::info("[blink_analysis] Model loaded and warmed up: {}", model_path_);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[blink_analysis] LibTorch error: {}", e.what());
        model_loaded_ = false;
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[blink_analysis] Init error: {}", e.what());
        model_loaded_ = false;
        return false;
    }
}

/* ================================================================
   Eye ROI Extraction
   ================================================================ */

std::pair<cv::Rect, cv::Rect>
blink_analysis_component::extractEyeROIs(const cv::Rect& face, const cv::Size& img_sz) {
    cv::Rect left_eye, right_eye;
    if (face.width <= 0 || face.height <= 0) return {left_eye, right_eye};

    int eye_w = static_cast<int>(face.width * 0.33f);
    int eye_h = static_cast<int>(face.height * 0.26f);
    int eye_y = face.y + static_cast<int>(face.height * 0.23f);

    int left_x  = face.x + static_cast<int>(face.width * 0.14f);
    int right_x = face.x + static_cast<int>(face.width * 0.53f);

    left_eye  = cv::Rect(left_x,  eye_y, eye_w, eye_h) & cv::Rect(0, 0, img_sz.width, img_sz.height);
    right_eye = cv::Rect(right_x, eye_y, eye_w, eye_h) & cv::Rect(0, 0, img_sz.width, img_sz.height);

    return {left_eye, right_eye};
}

/* ================================================================
   Eye Patch Preprocessing  ->  (3, 64, 64) tensor
   ================================================================ */

torch::Tensor blink_analysis_component::preprocessEyePatch(const cv::Mat& eye_img) {
    cv::Mat resized;
    cv::resize(eye_img, resized, cv::Size(crop_w_, crop_h_), 0, 0, cv::INTER_CUBIC);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // ImageNet standardization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    cv::merge(channels, float_img);

    auto tensor = torch::from_blob(float_img.data,
                                   {crop_h_, crop_w_, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).clone();  // (3, H, W)
    return tensor;
}

/* ================================================================
   Build High-Level Feature Vector (160 dims)
   ================================================================ */

torch::Tensor blink_analysis_component::buildHighFeature(
    const std::array<float, 3>& euler, float ear)
{
    auto feat = torch::zeros({160}, torch::kFloat32);
    auto acc = feat.accessor<float, 1>();
    acc[0] = euler[0];  // pitch
    acc[1] = euler[1];  // yaw
    acc[2] = euler[2];  // roll
    acc[159] = ear;     // eye aspect ratio
    return feat;
}

/* ================================================================
   Main Processing  (called per frame)
   ================================================================ */

blink_analysis::DetectionResult
blink_analysis_component::process(
    const cv::Mat& image,
    const cv::Rect& face_bbox,
    const std::array<float, 3>& head_pose_euler,
    float ear,
    int64_t timestamp)
{
    blink_analysis::DetectionResult result;
    result.timestamp = timestamp;

    if (!model_loaded_ || image.empty()) return result;

    /* 1. Extract eye ROIs from face bounding box */
    auto [left_eye, right_eye] = extractEyeROIs(face_bbox, image.size());

    /* Choose left eye as primary (fallback to right if left is invalid) */
    cv::Rect active_eye = left_eye;
    if (active_eye.area() <= 0 && right_eye.area() > 0) {
        active_eye = right_eye;
    }

    /* Build eye patch tensor */
    torch::Tensor low_feat;
    if (active_eye.area() > 0) {
        cv::Mat eye_patch = image(active_eye).clone();
        low_feat = preprocessEyePatch(eye_patch);
    } else {
        low_feat = torch::zeros({3, crop_h_, crop_w_}, torch::kFloat32);
    }

    /* Build high-level feature tensor */
    torch::Tensor high_feat = buildHighFeature(head_pose_euler, ear);

    /* 2. Push into sliding window buffers */
    low_feat_buffer_.push_back(low_feat);
    high_feat_buffer_.push_back(high_feat);
    if (low_feat_buffer_.size() > static_cast<size_t>(seq_len_)) {
        low_feat_buffer_.pop_front();
        high_feat_buffer_.pop_front();
    }

    /* 3. Run inference when sequence buffer is full */
    if (low_feat_buffer_.size() == static_cast<size_t>(seq_len_)) {
        result.buffer_full = true;

        // Stack: input_low (1, seq_len, 3, 64, 64), input_high (1, seq_len, 160)
        std::vector<torch::Tensor> low_list(low_feat_buffer_.begin(), low_feat_buffer_.end());
        std::vector<torch::Tensor> high_list(high_feat_buffer_.begin(), high_feat_buffer_.end());

        auto input_low  = torch::stack(low_list, 0).unsqueeze(0).to(device_);
        auto input_high = torch::stack(high_list, 0).unsqueeze(0).to(device_);

        float cls_prob = 0.0f;
        float seq_prob = 0.0f;

        try {
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> model_inputs = {input_low, input_high};
            auto output = module_.forward(model_inputs);

            if (output.isTuple()) {
                auto elements = output.toTuple()->elements();
                if (elements.size() >= 1 && elements[0].isTensor()) {
                    cls_prob = torch::sigmoid(elements[0].toTensor()).item<float>();
                }
                if (elements.size() >= 2 && elements[1].isTensor()) {
                    auto y_seq = elements[1].toTensor();  // (1, seq_len, 1)
                    int64_t last_idx = y_seq.size(1) - 1;
                    seq_prob = torch::sigmoid(y_seq[0][last_idx][0]).item<float>();
                }
            } else if (output.isTensor()) {
                cls_prob = torch::sigmoid(output.toTensor()).item<float>();
                seq_prob = cls_prob;
            }

            // Weighted combination (same as blink.detection.inference)
            current_blink_prob_ = 0.7f * cls_prob + 0.3f * seq_prob;
        }
        catch (const std::exception& e) {
            logger::error("[blink_analysis] Inference error: {}", e.what());
        }
        bool is_blinking = (current_blink_prob_ >= threshold_);

        // State transition & blink count
        if (is_blinking && !prev_blinking_) {
            blink_count_++;
        }
        prev_blinking_ = is_blinking;

        // PERCLOS calculation
        blink_history_.push_back(is_blinking);
        if (blink_history_.size() > perclos_history_size_) {
            blink_history_.pop_front();
        }
        int closed_count = std::count(blink_history_.begin(), blink_history_.end(), true);
        current_perclos_ = blink_history_.empty()
            ? 0.0f
            : static_cast<float>(closed_count) / blink_history_.size();

        result.cls_prob = cls_prob;
        result.seq_prob = seq_prob;
    }

    /* Fill result with current state */
    result.blink_prob = current_blink_prob_;
    result.is_blinking = prev_blinking_;
    result.blink_count = blink_count_;
    result.perclos = current_perclos_;

    return result;
}

/* ================================================================
   Visualization
   ================================================================ */

void blink_analysis_component::drawResult(
    cv::Mat& image,
    const cv::Rect& face_bbox,
    const blink_analysis::DetectionResult& result)
{
    /* Draw eye ROIs */
    auto [left_eye, right_eye] = extractEyeROIs(face_bbox, image.size());

    cv::Scalar eye_color = result.is_blinking
        ? cv::Scalar(0, 0, 255)    // Red when blinking
        : cv::Scalar(0, 255, 0);   // Green when open

    if (left_eye.area() > 0)
        cv::rectangle(image, left_eye, eye_color, 2);
    if (right_eye.area() > 0)
        cv::rectangle(image, right_eye, eye_color, 2);

    /* Draw blink info panel (top-right area) */
    int panel_w = 220;
    int panel_h = 85;
    int panel_x = image.cols - panel_w - 10;
    int panel_y = 10;

    // Semi-transparent background
    cv::Rect panel_rect(panel_x, panel_y, panel_w, panel_h);
    panel_rect &= cv::Rect(0, 0, image.cols, image.rows);
    if (panel_rect.area() > 0) {
        cv::Mat overlay;
        image.copyTo(overlay);
        cv::rectangle(overlay, panel_rect, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(overlay, 0.5, image, 0.5, 0, image);
        cv::rectangle(image, panel_rect, cv::Scalar(255, 255, 255), 1);
    }

    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fs = 0.45;
    int th = 1;
    int tx = panel_x + 8;
    int ty = panel_y + 18;

    // Status line
    std::string status = result.is_blinking ? "BLINK" : "OPEN";
    cv::Scalar status_color = result.is_blinking
        ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(image, "Eye: " + status, cv::Point(tx, ty),
                font, fs, status_color, th, cv::LINE_AA);

    // Probability
    char prob_str[64];
    snprintf(prob_str, sizeof(prob_str), "Prob: %.3f", result.blink_prob);
    cv::putText(image, prob_str, cv::Point(tx, ty + 18),
                font, fs, cv::Scalar(0, 255, 255), th, cv::LINE_AA);

    // Blink count
    char count_str[64];
    snprintf(count_str, sizeof(count_str), "Blinks: %d", result.blink_count);
    cv::putText(image, count_str, cv::Point(tx, ty + 36),
                font, fs, cv::Scalar(255, 255, 255), th, cv::LINE_AA);

    // PERCLOS
    char perclos_str[64];
    snprintf(perclos_str, sizeof(perclos_str), "PERCLOS: %.1f%%", result.perclos * 100.0f);
    cv::Scalar perclos_color = (result.perclos > 0.3f)
        ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
    cv::putText(image, perclos_str, cv::Point(tx, ty + 54),
                font, fs, perclos_color, th, cv::LINE_AA);
}
