/**
 * @file blink_analysis.hpp
 * @brief Blink Analysis module for BlinkLinMulT TorchScript model
 * @details Standalone blink detection component that can be embedded
 *          into osm.monolithic.inference_v2. Uses a sliding-window approach
 *          with dual-input (low-level eye image + high-level pose features).
 */

#pragma once

#include <dep/json.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <deque>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

namespace blink_analysis {
    /**
     * @brief Result of blink detection for a single frame
     */
    struct DetectionResult {
        float blink_prob = 0.0f;       ///< Combined blink probability [0,1]
        float cls_prob = 0.0f;         ///< Classification branch probability
        float seq_prob = 0.0f;         ///< Sequence branch probability
        bool is_blinking = false;      ///< Whether eyes are currently closed
        int blink_count = 0;           ///< Cumulative blink count
        float perclos = 0.0f;          ///< PERCLOS (% eye closure over window)
        int64_t timestamp = 0;         ///< Frame timestamp (ms)
        bool buffer_full = false;      ///< True if sequence buffer has enough frames
    };
}

/**
 * @class blink_analysis_component
 * @brief Encapsulates BlinkLinMulT TorchScript inference logic
 *
 * Pipeline:
 *   face bbox -> eye ROI extraction -> preprocessing -> sequence buffer -> model inference
 *
 * The model expects two inputs:
 *   - input_low:  (1, seq_len, 3, 64, 64) — eye patch image features
 *   - input_high: (1, seq_len, 160)        — high-level features (head pose, EAR, etc.)
 */
class blink_analysis_component {
public:
    blink_analysis_component();
    ~blink_analysis_component();

    /**
     * @brief Initialize model and parameters from JSON
     * @param params JSON parameters for blink_detection section
     * @return true on success
     */
    bool init(const nlohmann::json& params);

    /**
     * @brief Process a single frame and return blink detection result
     * @param image Input BGR image
     * @param face_bbox Face bounding box from YOLO detector
     * @param head_pose_euler Head pose euler angles [pitch, yaw, roll] (optional, zeros if unavailable)
     * @param ear Eye Aspect Ratio (optional, 0 if unavailable)
     * @param timestamp Frame timestamp in milliseconds
     * @return DetectionResult with blink probability, count, PERCLOS, etc.
     */
    blink_analysis::DetectionResult process(
        const cv::Mat& image,
        const cv::Rect& face_bbox,
        const std::array<float, 3>& head_pose_euler,
        float ear,
        int64_t timestamp);

    /**
     * @brief Draw blink detection visualization on output image
     * @param image Output image to draw on
     * @param face_bbox Face bounding box (in output image coordinates)
     * @param result Detection result
     */
    void drawResult(cv::Mat& image,
                    const cv::Rect& face_bbox,
                    const blink_analysis::DetectionResult& result);

    /** @brief Check if model is loaded and ready */
    bool isLoaded() const { return model_loaded_; }

private:
    /* ---- Model & Device ---- */
    torch::jit::script::Module module_;
    torch::Device device_ = torch::Device(torch::kCPU);
    bool model_loaded_ = false;

    /* ---- Parameters ---- */
    std::string model_path_ = "bin/x86_64/models/blinklinmult-union.torchscript";
    int gpu_id_ = 0;
    int seq_len_ = 15;
    int crop_w_ = 64;
    int crop_h_ = 64;
    float threshold_ = 0.5f;

    /* ---- Sliding Window Buffers ---- */
    std::deque<torch::Tensor> low_feat_buffer_;   // each: (3, 64, 64)
    std::deque<torch::Tensor> high_feat_buffer_;  // each: (160,)
    std::deque<bool> blink_history_;               // for PERCLOS
    const size_t perclos_history_size_ = 90;

    /* ---- State Tracking ---- */
    bool prev_blinking_ = false;
    int blink_count_ = 0;
    float current_blink_prob_ = 0.0f;
    float current_perclos_ = 0.0f;

    /* ---- Internal Helpers ---- */
    std::pair<cv::Rect, cv::Rect> extractEyeROIs(const cv::Rect& face, const cv::Size& img_sz);
    torch::Tensor preprocessEyePatch(const cv::Mat& eye_img);
    torch::Tensor buildHighFeature(const std::array<float, 3>& euler, float ear);
};
