/**
 * @file body.kps.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Body Keypoint Inference Component using YOLO11x-pose with TensorRT
 * @version 0.1
 * @date 2025-07-31
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_BODY_KPS_INFERENCE_COMPONENT_HPP_INCLUDED
#define FLAME_BODY_KPS_INFERENCE_COMPONENT_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

struct KeyPoint {
    float x, y;
    float confidence;
};

struct PoseResult {
    std::vector<KeyPoint> keypoints;
    float bbox_confidence;
    cv::Rect bbox;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class body_kps_inference : public flame::component::object {
    public:
        body_kps_inference();
        virtual ~body_kps_inference() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;

    private:
        std::atomic<bool> _worker_stop { false };
        std::thread _inference_worker;

        /* TensorRT related */
        std::unique_ptr<nvinfer1::IRuntime> _runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> _engine;
        std::unique_ptr<nvinfer1::IExecutionContext> _context;
        Logger _logger;

        /* CUDA memory */
        void* _gpu_input_buffer = nullptr;
        void* _gpu_output_buffer = nullptr;
        float* _cpu_input_buffer = nullptr;
        float* _cpu_output_buffer = nullptr;

        /* Model parameters */
        std::string _engine_path;
        int _input_width = 640;
        int _input_height = 640;
        int _num_keypoints = 17;
        size_t _input_size;
        size_t _output_size;

        /* Processing functions */
        void _inference_process();
        bool _load_engine(const std::string& engine_path);
        void _allocate_buffers();
        void _free_buffers();
        cv::Mat _preprocess_image(const cv::Mat& image);
        std::vector<PoseResult> _postprocess_output(float* output, int batch_size);
        void _draw_keypoints(cv::Mat& image, const std::vector<PoseResult>& results);
};

EXPORT_COMPONENT_API

#endif