/**
 * @file blink_analysis.cc
 * @brief Blink Analysis module implementation for BlinkLinMulT and OCEC TorchScript models
 * @details Supports BlinkLinMulT sequence model and OCEC eye open/close classifier.
 *          For OCEC, extracts a horizontally aligned square eye ROI using 68 landmarks.
 */

#include "blink_analysis.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <chrono>
#include <cmath>

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
        target_eye_ = params.value("target_eye", "left"); // Default: left eye
        bbox_scale_ = params.value("bbox_scale", params.value("scale_ratio", bbox_scale_));
        threshold_open_ = params.value("threshold_open", params.value("open_threshold", threshold_open_));
        threshold_close_ = params.value("threshold_close", params.value("close_threshold", threshold_close_));

        model_type_ = params.value("model_type", "");
        if (model_type_.empty()) {
            if (model_path_.find("ocec") != std::string::npos) {
                model_type_ = "ocec";
            } else {
                model_type_ = "blinklinmult";
            }
        }

        logger::info("[blink_analysis] Initializing with model_type='{}', target_eye='{}', model_path='{}', gpu_id={}",
                     model_type_, target_eye_, model_path_, gpu_id_);

        /* Search for model file if path does not exist */
        if (!fs::exists(model_path_)) {
            std::vector<std::string> candidates = {
                model_path_,
                "bin/x86_64/models/ocec_l.torchscript",
                "/home/iae-vc/dev/flame_osm/bin/x86_64/models/ocec_l.torchscript",
                "bin/x86_64/models/blinklinmult-union.torchscript",
                "/home/iae-vc/dev/flame_osm/bin/x86_64/models/blinklinmult-union.torchscript"
            };
            for (const auto& cand : candidates) {
                if (fs::exists(cand)) {
                    logger::info("[blink_analysis] Model path not directly found, matched candidate: '{}'", cand);
                    model_path_ = cand;
                    break;
                }
            }
        }

        if (!fs::exists(model_path_)) {
            logger::error("[blink_analysis] Model file not found: {}", model_path_);
            return false;
        }

        /* Setup device (multi-GPU portable) */
        if (torch::cuda::is_available() && gpu_id_ >= 0) {
            device_ = torch::Device(torch::kCUDA, gpu_id_);
            logger::info("[blink_analysis] Using CUDA GPU device: {}", gpu_id_);
        } else {
            device_ = torch::Device(torch::kCPU);
            logger::info("[blink_analysis] Using CPU device");
        }

        /* Load TorchScript model
         * Try loading with device map first; fallback to CPU load and .to(device_) */
        try {
            module_ = torch::jit::load(model_path_, device_);
        }
        catch (const std::exception& e) {
            logger::warn("[blink_analysis] Direct device load notice: {}, falling back to CPU load + transfer", e.what());
            module_ = torch::jit::load(model_path_, torch::kCPU);
        }
        module_.to(device_);
        module_.eval();

        /* Warmup forward pass */
        {
            torch::NoGradGuard no_grad;
            if (model_type_ == "ocec") {
                auto dummy = torch::zeros({1, 3, 24, 40}, device_);
                module_.forward({dummy});
                logger::info("[blink_analysis] OCEC model loaded and warmed up: {}", model_path_);
            } else {
                auto dummy_low  = torch::zeros({1, seq_len_, 3, crop_h_, crop_w_}, device_);
                auto dummy_high = torch::zeros({1, seq_len_, 160}, device_);
                std::vector<torch::jit::IValue> inputs = {dummy_low, dummy_high};
                module_.forward(inputs);
                allocateGpuBuffers();
                logger::info("[blink_analysis] BlinkLinMulT model loaded and warmed up: {}", model_path_);
            }
        }

        model_loaded_ = true;
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
   GPU Buffer Pre-allocation (BlinkLinMulT)
   ================================================================ */

void blink_analysis_component::allocateGpuBuffers() {
    try {
        torch::NoGradGuard no_grad;
        gpu_low_buf_  = torch::zeros({1, seq_len_, 3, crop_h_, crop_w_}, device_);
        gpu_high_buf_ = torch::zeros({1, seq_len_, 160}, device_);
        gpu_buf_allocated_ = true;
        ring_idx_ = 0;
        logger::info("[blink_analysis] Pre-allocated GPU buffers on device (low: [{},{},{},{},{}], high: [{},{},{}])",
                     1, seq_len_, 3, crop_h_, crop_w_, 1, seq_len_, 160);
    }
    catch (const std::exception& e) {
        logger::error("[blink_analysis] Failed to allocate GPU buffers: {}", e.what());
        gpu_buf_allocated_ = false;
    }
}

/* ================================================================
   Eye ROI Extraction (BlinkLinMulT fallback)
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

torch::Tensor blink_analysis_component::preprocessEyePatch(const cv::Mat& eye_img) {
    cv::Mat resized;
    cv::resize(eye_img, resized, cv::Size(crop_w_, crop_h_), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar std_dev(0.229, 0.224, 0.225);
    cv::subtract(float_img, mean, float_img);
    cv::divide(float_img, std_dev, float_img);

    auto tensor = torch::from_blob(float_img.data, {crop_h_, crop_w_, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).clone();
    return tensor;
}

torch::Tensor blink_analysis_component::buildHighFeature(
    const std::array<float, 3>& euler, float ear)
{
    auto feat = torch::zeros({160}, torch::kFloat32);
    feat[0] = euler[0] / 90.0f;
    feat[1] = euler[1] / 90.0f;
    feat[2] = euler[2] / 90.0f;
    feat[3] = ear;
    return feat;
}

/* ================================================================
   OCEC: Landmark-based Horizontal Square Eye ROI Extraction
   ================================================================ */

bool blink_analysis_component::extractHorizontalSquareEyeROI(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks_68,
    cv::Mat& out_cropped_square,
    std::vector<cv::Point2f>& out_corners)
{
    if (image.empty() || landmarks_68.size() < 68) {
        return false;
    }

    // 68 facial landmarks:
    // Left eye (subject's left eye): inner corner = 42, outer corner = 45, indices: 42..47
    // Right eye (subject's right eye): outer corner = 36, inner corner = 39, indices: 36..41
    cv::Point2f p_left, p_right;
    cv::Point2f center(0.0f, 0.0f);

    if (target_eye_ == "right") {
        // Right eye (subject's right eye): 36 outer, 39 inner
        p_left = landmarks_68[36];
        p_right = landmarks_68[39];
        // Centroid of all 6 right eye landmarks (36..41)
        for (int i = 36; i <= 41; ++i) {
            center.x += landmarks_68[i].x;
            center.y += landmarks_68[i].y;
        }
        center.x /= 6.0f;
        center.y /= 6.0f;
    } else {
        // Default: Left eye (subject's left eye): 42 inner (image left), 45 outer (image right)
        p_left = landmarks_68[42];
        p_right = landmarks_68[45];
        // Centroid of all 6 left eye landmarks (42..47)
        for (int i = 42; i <= 47; ++i) {
            center.x += landmarks_68[i].x;
            center.y += landmarks_68[i].y;
        }
        center.x /= 6.0f;
        center.y /= 6.0f;
    }

    float dx = p_right.x - p_left.x;
    float dy = p_right.y - p_left.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist < 4.0f) {
        return false;
    }

    // Angle in degrees to make the eye horizontal
    // atan2(dy, dx) is the angle of vector p_left -> p_right
    float angle = std::atan2(dy, dx) * 180.0f / static_cast<float>(CV_PI);

    // Square bounding box: width = height = eye horizontal distance (D = dist * bbox_scale_)
    // The center of the square bounding box matches the eye landmark centroid exactly.
    float D = dist * bbox_scale_;
    float half_d = D * 0.5f;

    // Calculate 4 corners of the rotated square in original image coordinates
    float rad = angle * static_cast<float>(CV_PI) / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    out_corners.resize(4);
    // 0: top-left (-half_d, -half_d)
    out_corners[0] = cv::Point2f(center.x + (-half_d * cos_a - (-half_d) * sin_a),
                                 center.y + (-half_d * sin_a + (-half_d) * cos_a));
    // 1: top-right (half_d, -half_d)
    out_corners[1] = cv::Point2f(center.x + ( half_d * cos_a - (-half_d) * sin_a),
                                 center.y + ( half_d * sin_a + (-half_d) * cos_a));
    // 2: bottom-right (half_d, half_d)
    out_corners[2] = cv::Point2f(center.x + ( half_d * cos_a -   half_d  * sin_a),
                                 center.y + ( half_d * sin_a +   half_d  * cos_a));
    // 3: bottom-left (-half_d, half_d)
    out_corners[3] = cv::Point2f(center.x + (-half_d * cos_a -   half_d  * sin_a),
                                 center.y + (-half_d * sin_a +   half_d  * cos_a));

    // Crop horizontally aligned square via warpAffine
    // Center maps to (D/2, D/2) in destination image
    int crop_size = std::max(4, static_cast<int>(std::round(D)));
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    M.at<double>(0, 2) += (crop_size * 0.5 - center.x);
    M.at<double>(1, 2) += (crop_size * 0.5 - center.y);

    cv::warpAffine(image, out_cropped_square, M, cv::Size(crop_size, crop_size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    return !out_cropped_square.empty();
}

/* ================================================================
   OCEC: Inference Process
   ================================================================ */

blink_analysis::DetectionResult blink_analysis_component::processOCEC(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks_68,
    int64_t timestamp)
{
    blink_analysis::DetectionResult result;
    result.timestamp = timestamp;

    if (!model_loaded_ || image.empty()) {
        return result;
    }

    cv::Mat cropped_square;
    std::vector<cv::Point2f> corners;
    if (!extractHorizontalSquareEyeROI(image, landmarks_68, cropped_square, corners)) {
        return result;
    }

    result.cropped_eye = cropped_square.clone();
    result.roi_corners = corners;
    result.valid = true;

    // OCEC input resolution: width = 40, height = 24
    cv::Mat ocec_input;
    cv::resize(cropped_square, ocec_input, cv::Size(40, 24), 0, 0, cv::INTER_LINEAR);

    // Preprocessing matching PINTO0309/OCEC:
    // BGR image -> resized (40, 24) -> float32 / 255.0 -> [1, 3, 24, 40]
    cv::Mat float_mat;
    ocec_input.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);

    try {
        torch::NoGradGuard no_grad;
        auto input_tensor = torch::from_blob(float_mat.data, {1, 24, 40, 3}, torch::kFloat32);
        input_tensor = input_tensor.permute({0, 3, 1, 2}).to(device_); // [1, 3, 24, 40]

        auto output = module_.forward({input_tensor}).toTensor();
        float prob_open = output.squeeze().item<float>();
        prob_open = std::clamp(prob_open, 0.0f, 1.0f);

        result.prob_open = prob_open;
        result.blink_prob = 1.0f - prob_open; // eye-closure probability

        // Schmitt trigger hysteresis:
        // When closed, switch to open if prob_open >= threshold_open_ (default: 0.3)
        // When open, switch to closed if prob_open < threshold_close_ (default: 0.1)
        if (!current_is_open_) {
            if (prob_open >= threshold_open_) {
                current_is_open_ = true;
            }
        } else {
            if (prob_open < threshold_close_) {
                current_is_open_ = false;
            }
        }
        result.is_blinking = !current_is_open_;

        // Update blink count and PERCLOS
        if (result.is_blinking && !prev_blinking_) {
            blink_count_++;
        }
        prev_blinking_ = result.is_blinking;
        blink_history_.push_back(result.is_blinking);
        while (blink_history_.size() > perclos_history_size_) {
            blink_history_.pop_front();
        }

        int closed_frames = std::count(blink_history_.begin(), blink_history_.end(), true);
        result.perclos = static_cast<float>(closed_frames) / static_cast<float>(blink_history_.size());
        result.blink_count = blink_count_;
    }
    catch (const std::exception& e) {
        logger::error("[blink_analysis] OCEC inference error: {}", e.what());
    }

    return result;
}

/* ================================================================
   Main Process Dispatcher
   ================================================================ */

blink_analysis::DetectionResult
blink_analysis_component::process(
    const cv::Mat& image,
    const cv::Rect& face_bbox,
    const std::array<float, 3>& head_pose_euler,
    float ear,
    int64_t timestamp,
    const std::vector<cv::Point2f>& landmarks_68)
{
    if (model_type_ == "ocec") {
        return processOCEC(image, landmarks_68, timestamp);
    }

    /* ---- BlinkLinMulT Pipeline ---- */
    blink_analysis::DetectionResult result;
    result.timestamp = timestamp;

    if (!model_loaded_ || image.empty()) return result;

    /* 1. Extract eye ROIs from face bounding box */
    auto [left_eye, right_eye] = extractEyeROIs(face_bbox, image.size());

    cv::Rect active_eye = left_eye;
    if (active_eye.area() <= 0 && right_eye.area() > 0) {
        active_eye = right_eye;
    }

    torch::Tensor low_feat;
    if (active_eye.area() > 0) {
        cv::Mat eye_patch = image(active_eye).clone();
        low_feat = preprocessEyePatch(eye_patch);
    } else {
        low_feat = torch::zeros({3, crop_h_, crop_w_}, torch::kFloat32);
    }

    torch::Tensor high_feat = buildHighFeature(head_pose_euler, ear);

    low_feat_buffer_.push_back(low_feat);
    high_feat_buffer_.push_back(high_feat);

    while (static_cast<int>(low_feat_buffer_.size()) > seq_len_) {
        low_feat_buffer_.pop_front();
        high_feat_buffer_.pop_front();
    }

    result.buffer_full = (static_cast<int>(low_feat_buffer_.size()) >= seq_len_);
    if (!result.buffer_full) {
        result.blink_prob = current_blink_prob_;
        result.is_blinking = prev_blinking_;
        result.blink_count = blink_count_;
        result.perclos = current_perclos_;
        return result;
    }

    if (!gpu_buf_allocated_) {
        allocateGpuBuffers();
        if (!gpu_buf_allocated_) return result;
    }

    try {
        torch::NoGradGuard no_grad;
        int buf_len = static_cast<int>(low_feat_buffer_.size());
        auto low_stack = torch::stack(std::vector<torch::Tensor>(low_feat_buffer_.begin(), low_feat_buffer_.end()));
        auto high_stack = torch::stack(std::vector<torch::Tensor>(high_feat_buffer_.begin(), high_feat_buffer_.end()));

        gpu_low_buf_.slice(1, 0, buf_len).copy_(low_stack.to(device_));
        gpu_high_buf_.slice(1, 0, buf_len).copy_(high_stack.to(device_));

        std::vector<torch::jit::IValue> inputs = {gpu_low_buf_, gpu_high_buf_};
        auto output = module_.forward(inputs);

        float cls_p = 0.0f;
        float seq_p = 0.0f;

        if (output.isTuple()) {
            auto tuple_out = output.toTuple();
            auto elems = tuple_out->elements();
            if (elems.size() >= 2) {
                auto t0 = torch::sigmoid(elems[0].toTensor()).cpu();
                auto t1 = torch::sigmoid(elems[1].toTensor()).cpu();
                cls_p = t0.slice(1, -1, t0.size(1)).squeeze().item<float>();
                seq_p = t1.slice(1, -1, t1.size(1)).squeeze().item<float>();
            } else if (elems.size() == 1) {
                auto t0 = torch::sigmoid(elems[0].toTensor()).cpu();
                cls_p = t0.slice(1, -1, t0.size(1)).squeeze().item<float>();
                seq_p = cls_p;
            }
        } else if (output.isTensor()) {
            auto t0 = torch::sigmoid(output.toTensor()).cpu();
            cls_p = t0.slice(1, -1, t0.size(1)).squeeze().item<float>();
            seq_p = cls_p;
        }

        result.cls_prob = cls_p;
        result.seq_prob = seq_p;
        result.blink_prob = 0.5f * (cls_p + seq_p);
        current_blink_prob_ = result.blink_prob;
    }
    catch (const std::exception& e) {
        logger::error("[blink_analysis] Forward pass error: {}", e.what());
        return result;
    }

    result.is_blinking = (current_blink_prob_ >= threshold_);
    if (result.is_blinking && !prev_blinking_) {
        blink_count_++;
    }
    prev_blinking_ = result.is_blinking;

    blink_history_.push_back(result.is_blinking);
    while (blink_history_.size() > perclos_history_size_) {
        blink_history_.pop_front();
    }

    int closed_count = std::count(blink_history_.begin(), blink_history_.end(), true);
    current_perclos_ = static_cast<float>(closed_count) / static_cast<float>(blink_history_.size());

    result.blink_count = blink_count_;
    result.perclos = current_perclos_;
    return result;
}

/* ================================================================
   Visualization
   ================================================================ */

void blink_analysis_component::drawResultOCEC(
    cv::Mat& image,
    const blink_analysis::DetectionResult& result,
    float scale_x,
    float scale_y,
    int graph_x,
    int graph_y,
    int graph_w,
    int graph_h,
    float ui_scale,
    int spacing,
    int margin_x)
{
    if (!result.valid) return;

    // 1. Draw white bounding box of the cropped eye ROI on the output image
    if (result.roi_corners.size() == 4) {
        std::vector<cv::Point> pts(4);
        for (int i = 0; i < 4; ++i) {
            pts[i] = cv::Point(static_cast<int>(std::round(result.roi_corners[i].x * scale_x)),
                               static_cast<int>(std::round(result.roi_corners[i].y * scale_y)));
        }
        for (int i = 0; i < 4; ++i) {
            cv::line(image, pts[i], pts[(i + 1) % 4], cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }

    // 2. Display cropped eye image and OCEC result directly above the Readiness Score graph box on the right side
    if (!result.cropped_eye.empty()) {
        int panel_right = (graph_x >= 0 && graph_w > 0) ? (graph_x + graph_w) : (image.cols - margin_x);
        int y_bottom = (graph_y >= 0) ? (graph_y - spacing) : (image.rows - static_cast<int>(std::round(75.0f * ui_scale)) - spacing);

        // Eye image size matching UI scale
        int eye_sz = std::max(20, static_cast<int>(std::round(48.0f * ui_scale)));

        // Match font size and thickness to other panels (Gaze, EAR, Head Pose: 0.40 * ui_scale)
        float font_scale = 0.40f * ui_scale;
        int font_thick = std::max(1, static_cast<int>(std::round(ui_scale)));
        int font = cv::FONT_HERSHEY_SIMPLEX;

        bool is_open = !result.is_blinking;
        std::string text = cv::format("Eye: %s (%.2f)", is_open ? "OPEN" : "CLOSE", result.prob_open);
        cv::Scalar text_color = is_open ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

        int baseline = 0;
        cv::Size text_sz = cv::getTextSize(text, font, font_scale, font_thick, &baseline);

        int pad_x = std::max(4, static_cast<int>(std::round(6.0f * ui_scale)));
        int pad_y = std::max(3, static_cast<int>(std::round(4.0f * ui_scale)));
        int panel_h = eye_sz + pad_y * 2;
        int panel_w = pad_x + eye_sz + pad_x + text_sz.width + pad_x;

        int panel_x = panel_right - panel_w;
        int panel_y = y_bottom - panel_h;

        // Clip to image boundary
        if (panel_x >= 0 && panel_y >= 0 && panel_x + panel_w <= image.cols && panel_y + panel_h <= image.rows) {
            cv::Rect panel_roi(panel_x, panel_y, panel_w, panel_h);
            cv::Mat overlay;
            image.copyTo(overlay);
            cv::rectangle(overlay, panel_roi, cv::Scalar(20, 20, 20), cv::FILLED);
            cv::addWeighted(overlay, 0.6, image, 0.4, 0, image);

            int border_thick = std::max(1, static_cast<int>(std::round(1.0f * ui_scale)));
            cv::rectangle(image, panel_roi, cv::Scalar(80, 80, 80), border_thick);

            // Copy eye image
            int eye_x = panel_x + pad_x;
            int eye_y = panel_y + pad_y;
            cv::Rect eye_rect(eye_x, eye_y, eye_sz, eye_sz);
            cv::Mat disp_eye;
            cv::resize(result.cropped_eye, disp_eye, cv::Size(eye_sz, eye_sz), 0, 0, cv::INTER_LINEAR);
            disp_eye.copyTo(image(eye_rect));

            // White border around eye image
            cv::rectangle(image, eye_rect, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            // Draw text vertically centered next to the eye image
            int text_x = eye_x + eye_sz + pad_x;
            int text_y = panel_y + (panel_h + text_sz.height) / 2;
            cv::putText(image, text, cv::Point(text_x, text_y), font, font_scale, text_color, font_thick, cv::LINE_AA);
        }
    }
}

void blink_analysis_component::drawResult(
    cv::Mat& image,
    const cv::Rect& face_bbox,
    const blink_analysis::DetectionResult& result,
    float scale_x,
    float scale_y,
    int graph_x,
    int graph_y,
    int graph_w,
    int graph_h,
    float ui_scale,
    int spacing,
    int margin_x)
{
    if (model_type_ == "ocec") {
        drawResultOCEC(image, result, scale_x, scale_y, graph_x, graph_y, graph_w, graph_h, ui_scale, spacing, margin_x);
        return;
    }

    /* ---- BlinkLinMulT Visualization ---- */
    auto [left_eye, right_eye] = extractEyeROIs(face_bbox, image.size());

    cv::Scalar eye_color = result.is_blinking
        ? cv::Scalar(0, 0, 255)
        : cv::Scalar(0, 255, 0);

    if (left_eye.area() > 0)
        cv::rectangle(image, left_eye, eye_color, 2);
    if (right_eye.area() > 0)
        cv::rectangle(image, right_eye, eye_color, 2);

    int panel_w = 220;
    int panel_h = 85;
    int panel_x = image.cols - panel_w - 10;
    int panel_y = 10;

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

    std::string status = result.is_blinking ? "BLINK" : "OPEN";
    cv::Scalar status_color = result.is_blinking
        ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(image, "Eye: " + status, cv::Point(tx, ty),
                font, fs, status_color, th, cv::LINE_AA);

    char prob_str[64];
    snprintf(prob_str, sizeof(prob_str), "Prob: %.3f", result.blink_prob);
    cv::putText(image, prob_str, cv::Point(tx, ty + 18),
                font, fs, cv::Scalar(0, 255, 255), th, cv::LINE_AA);

    char count_str[64];
    snprintf(count_str, sizeof(count_str), "Blinks: %d", result.blink_count);
    cv::putText(image, count_str, cv::Point(tx, ty + 36),
                font, fs, cv::Scalar(255, 255, 255), th, cv::LINE_AA);

    char perclos_str[64];
    snprintf(perclos_str, sizeof(perclos_str), "PERCLOS: %.1f%%", result.perclos * 100.0f);
    cv::Scalar perclos_color = (result.perclos > 0.3f)
        ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
    cv::putText(image, perclos_str, cv::Point(tx, ty + 54),
                font, fs, perclos_color, th, cv::LINE_AA);
}
