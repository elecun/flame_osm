/**
 * @file blink.detection.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Blink Detection Inference Component using BlinkLinMulT with LibTorch
 * @version 0.1
 * @date 2026-09-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef FLAME_BLINK_DETECTION_INFERENCE_HPP_INCLUDED
#define FLAME_BLINK_DETECTION_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <dep/json.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <queue>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

namespace blink_detection {
    struct QueuedFrame {
        cv::Mat image;
        nlohmann::json tag;
        int64_t timestamp = 0;
    };

    struct EyeFrameFeature {
        torch::Tensor low_feature;   // (3, 64, 64)
        torch::Tensor high_feature;  // (160,)
        int64_t timestamp = 0;
        cv::Rect left_eye_rect;
        cv::Rect right_eye_rect;
        cv::Rect face_rect;
        bool has_eye = false;
    };

    struct DetectionResult {
        float blink_prob_cls = 0.0f;
        float blink_prob_seq = 0.0f;
        float blink_prob = 0.0f;
        bool is_blinking = false;
        int blink_count = 0;
        float perclos = 0.0f;
        int64_t timestamp = 0;
    };
}

class blink_detection_inference : public flame::component::Object {
public:
    blink_detection_inference();
    virtual ~blink_detection_inference() = default;

    /* default flame component interface functions */
    bool onInit() override;
    void onLoop() override;
    void onClose() override;
    void onData(flame::component::ZData& data) override;

private:
    /* Inference worker thread */
    void _inference_process();

    /* Model loader */
    bool _load_model(const std::string& model_path, int gpu_id);

    /* Preprocessing */
    cv::Rect _detect_face_or_estimate_roi(const cv::Mat& image);
    std::pair<cv::Rect, cv::Rect> _extract_eye_rois(const cv::Rect& face_box, const cv::Size& img_size);
    torch::Tensor _preprocess_eye_patch(const cv::Mat& eye_img);

    /* Internal State */
    std::atomic<bool> _worker_stop{false};
    std::thread _inference_worker;

    /* Queue for incoming data */
    std::queue<blink_detection::QueuedFrame> _data_queue;
    std::mutex _queue_mtx;
    std::condition_variable _queue_cv;
    const size_t _max_queue_size = 10;

    /* LibTorch module and device */
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    bool _is_model_loaded = false;

    /* Model & Inference parameters */
    std::string _model_path = "bin/x86_64/models/blinklinmult-union.torchscript";
    int _gpu_id = 0;
    int _seq_len = 15;
    int _crop_width = 64;
    int _crop_height = 64;
    float _threshold = 0.5f;
    std::string _eye_selection = "left"; // "left", "right", "both"
    bool _show_info = true;
    bool _visualize = true;

    /* Temporal sliding window */
    std::deque<blink_detection::EyeFrameFeature> _sequence_buffer;
    std::deque<bool> _blink_history; // for PERCLOS calculation (last 60-90 frames)
    const size_t _perclos_history_size = 90;

    /* Tracking state */
    bool _prev_is_blinking = false;
    int _blink_count = 0;
    float _current_blink_prob = 0.0f;
    float _current_perclos = 0.0f;
    std::chrono::steady_clock::time_point _last_blink_time;

    /* OpenCV Cascade classifier for fallback face/eye detection */
    cv::CascadeClassifier _face_cascade;
    cv::CascadeClassifier _eye_cascade;
    bool _has_cascade = false;

    /* Output monitor parameters */
    int _target_width = 800;
    int _target_height = 450;
    std::string _input_port = "image_stream_1";
    std::string _output_data_port = "blink_result";
    std::string _output_monitor_port = "image_stream_1_processed_monitor";
};

EXPORT_COMPONENT_API

#endif // FLAME_BLINK_DETECTION_INFERENCE_HPP_INCLUDED
