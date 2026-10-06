/**
 * @file blink_analysis.hpp
 * @brief Blink Analysis module for BlinkLinMulT and OCEC TorchScript models
 * @details Standalone blink detection component that can be embedded
 *          into osm.monolithic.inference_v2. Supports BlinkLinMulT (sequence)
 *          and OCEC (open/closed eye classifier with landmark-based horizontal square ROI).
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

        // OCEC / Landmark-based eye fields
        float prob_open = 0.0f;        ///< OCEC prob_open [0.0, 1.0]
        cv::Mat cropped_eye;           ///< Rotated & cropped square eye ROI (horizontal)
        std::vector<cv::Point2f> roi_corners; ///< 4 corners in original image coords
        bool valid = false;            ///< True if eye ROI and detection are valid
    };
}

/**
 * @class blink_analysis_component
 * @brief Encapsulates BlinkLinMulT / OCEC TorchScript inference logic
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
     * @param head_pose_euler Head pose euler angles [pitch, yaw, roll]
     * @param ear Eye Aspect Ratio
     * @param timestamp Frame timestamp in milliseconds
     * @param landmarks_68 68 facial landmarks from face analysis (optional)
     * @return DetectionResult with blink probability, count, PERCLOS, etc.
     */
    blink_analysis::DetectionResult process(
        const cv::Mat& image,
        const cv::Rect& face_bbox,
        const std::array<float, 3>& head_pose_euler,
        float ear,
        int64_t timestamp,
        const std::vector<cv::Point2f>& landmarks_68 = {});

    /**
     * @brief Draw blink detection visualization on output image
     * @param image Output image to draw on
     * @param face_bbox Face bounding box (in output image coordinates)
     * @param result Detection result
     * @param scale_x Scaling factor from original image width to output image width
     * @param scale_y Scaling factor from original image height to output image height
     * @param graph_x X position of the readiness score graph box (-1 for default)
     * @param graph_y Y position of the readiness score graph box (-1 for default)
     * @param graph_w Width of the readiness score graph box (-1 for default)
     * @param graph_h Height of the readiness score graph box (-1 for default)
     * @param ui_scale Scaling factor for UI elements
     * @param spacing Spacing between UI panels
     * @param margin_x Margin from image edge
     */
    void drawResult(cv::Mat& image,
                    const cv::Rect& face_bbox,
                    const blink_analysis::DetectionResult& result,
                    float scale_x = 1.0f,
                    float scale_y = 1.0f,
                    int graph_x = -1,
                    int graph_y = -1,
                    int graph_w = -1,
                    int graph_h = -1,
                    float ui_scale = 1.0f,
                    int spacing = 10,
                    int margin_x = 10);

    /** @brief Check if model is loaded and ready */
    bool isLoaded() const { return model_loaded_; }

    /** @brief Get model type ("ocec" or "blinklinmult") */
    const std::string& getModelType() const { return model_type_; }

private:
    /* ---- Model & Device ---- */
    torch::jit::script::Module module_;
    torch::Device device_ = torch::Device(torch::kCPU);
    bool model_loaded_ = false;

    /* ---- Parameters ---- */
    std::string model_type_ = "ocec"; // "ocec" or "blinklinmult"
    std::string model_path_ = "bin/x86_64/models/ocec_l.torchscript";
    std::string target_eye_ = "left"; // "left" (default) or "right"
    int gpu_id_ = 0;
    int seq_len_ = 15;
    int crop_w_ = 40;
    int crop_h_ = 24;
    float threshold_ = 0.5f;
    float bbox_scale_ = 1.10f;        // 10% bounding box expansion (default: 1.10)
    float threshold_open_ = 0.3f;     // Schmitt trigger threshold for opening (default: 0.3)
    float threshold_close_ = 0.1f;    // Schmitt trigger threshold for closing (default: 0.1)

    /* ---- Sliding Window Buffers (for BlinkLinMulT) ---- */
    std::deque<torch::Tensor> low_feat_buffer_;   // each: (3, 64, 64)
    std::deque<torch::Tensor> high_feat_buffer_;  // each: (160,)
    std::deque<bool> blink_history_;               // for PERCLOS
    const size_t perclos_history_size_ = 90;

    /* ---- Pre-allocated GPU Buffers (for BlinkLinMulT) ---- */
    torch::Tensor gpu_low_buf_;    // (1, seq_len, 3, crop_h, crop_w) on device_
    torch::Tensor gpu_high_buf_;   // (1, seq_len, 160) on device_
    bool gpu_buf_allocated_ = false;
    int ring_idx_ = 0;             // current write position in ring buffer

    /* ---- State Tracking ---- */
    bool prev_blinking_ = false;
    bool current_is_open_ = false; // Schmitt trigger state
    int blink_count_ = 0;
    float current_blink_prob_ = 0.0f;
    float current_perclos_ = 0.0f;

    /* ---- Internal Helpers for BlinkLinMulT ---- */
    void allocateGpuBuffers();
    std::pair<cv::Rect, cv::Rect> extractEyeROIs(const cv::Rect& face, const cv::Size& img_sz);
    torch::Tensor preprocessEyePatch(const cv::Mat& eye_img);
    torch::Tensor buildHighFeature(const std::array<float, 3>& euler, float ear);

    /* ---- Internal Helpers for OCEC ---- */
    bool extractHorizontalSquareEyeROI(
        const cv::Mat& image,
        const std::vector<cv::Point2f>& landmarks_68,
        cv::Mat& out_cropped_square,
        std::vector<cv::Point2f>& out_corners);

    blink_analysis::DetectionResult processOCEC(
        const cv::Mat& image,
        const std::vector<cv::Point2f>& landmarks_68,
        int64_t timestamp);

    void drawResultOCEC(
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
        int margin_x);
};
