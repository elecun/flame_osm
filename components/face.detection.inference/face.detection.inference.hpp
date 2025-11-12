/**
 * @file face.detection.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Face Detection Inference Component using YOLO11 with TensorRT
 * @version 0.1
 * @date 2025-11-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef FLAME_FACE_DETECTION_COMPONENT_HPP_INCLUDED
#define FLAME_FACE_DETECTION_COMPONENT_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace face_detection {
    struct FaceResult {
        cv::Rect bbox;
        float confidence;
    };
}

using namespace std;
using namespace cv;
using namespace flame::component;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class face_detection_inference : public flame::component::object {
    public:
        face_detection_inference();
        virtual ~face_detection_inference() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message(const message_t& msg) override;

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
        cudaStream_t _cuda_stream = nullptr;  // CUDA stream for async execution

        /* Model parameters */
        std::string _model_path;
        int _input_width = 640;
        int _input_height = 640;
        int _gpu_id = 0;  // GPU device ID
        size_t _input_size;
        size_t _output_size;

        /* Letterbox padding info (updated during preprocessing) */
        float _letterbox_scale = 1.0f;
        int _letterbox_pad_left = 0;
        int _letterbox_pad_top = 0;

        /* Processing functions */
        void _inference_process();
        bool _load_engine(const std::string& engine_path);
        void _allocate_buffers();
        void _free_buffers();
        cv::Mat _preprocess_image(const cv::Mat& image);
        std::vector<face_detection::FaceResult> _postprocess_output(float* output, int batch_size, int img_width, int img_height);
};

EXPORT_COMPONENT_API

#endif
