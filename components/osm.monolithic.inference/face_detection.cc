#include "face_detection.hpp"
#include <flame/log.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

void FaceTRTLogger::log(Severity severity, const char* msg) noexcept {
    switch(severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            logger::error("[FaceTRT] {}", msg);
            break;
        case Severity::kWARNING:
            logger::warn("[FaceTRT] {}", msg);
            break;
        case Severity::kINFO:
            logger::info("[FaceTRT] {}", msg);
            break;
        case Severity::kVERBOSE:
            logger::debug("[FaceTRT] {}", msg);
            break;
    }
}

face_detection::face_detection() {
}

face_detection::~face_detection() {
    freeBuffers();
    _context.reset();
    _engine.reset();
    _runtime.reset();
}

bool face_detection::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;

    try {
        if (!fs::exists(model_path)) {
            logger::error("[FaceDetection] Model file not found: {}", model_path);
            return false;
        }

        // Set CUDA device
        cudaError_t cuda_status = cudaSetDevice(_gpu_id);
        if (cuda_status != cudaSuccess) {
            logger::error("[FaceDetection] Failed to set CUDA device {}: {}", _gpu_id, cudaGetErrorString(cuda_status));
            return false;
        }

        // Read engine file
        std::ifstream engine_file(model_path, std::ios::binary);
        if (!engine_file.good()) {
            logger::error("[FaceDetection] Failed to open engine file: {}", model_path);
            return false;
        }

        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);
        engine_file.close();

        // Create TensorRT runtime and deserialize engine
        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if (!_runtime) {
            logger::error("[FaceDetection] Failed to create TensorRT runtime");
            return false;
        }

        _engine.reset(_runtime->deserializeCudaEngine(engine_data.data(), engine_size));
        if (!_engine) {
            logger::error("[FaceDetection] Failed to deserialize CUDA engine");
            return false;
        }

        _context.reset(_engine->createExecutionContext());
        if (!_context) {
            logger::error("[FaceDetection] Failed to create execution context");
            return false;
        }

        // Create CUDA stream
        cudaError_t stream_status = cudaStreamCreate(&_cuda_stream);
        if (stream_status != cudaSuccess) {
            logger::error("[FaceDetection] Failed to create CUDA stream: {}", cudaGetErrorString(stream_status));
            return false;
        }

        // Allocate buffers
        allocateBuffers();

        logger::info("[FaceDetection] Loaded TensorRT engine successfully from {}", model_path);
        return true;
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during model load: {}", e.what());
        return false;
    }
}

void face_detection::allocateBuffers() {
    _input_size = 1 * 3 * _input_height * _input_width * sizeof(float);
    
    // YOLO11 detection output: [batch, num_boxes, channels] (num_boxes=8400, channels=6: 4(bbox) + 1(conf) + 1(class))
    const int num_boxes = 8400;
    const int num_channels = 6;
    _output_size = 1 * num_boxes * num_channels * sizeof(float);

    _cpu_input_buffer = new float[_input_size / sizeof(float)];
    _cpu_output_buffer = new float[_output_size / sizeof(float)];

    cudaMalloc(&_gpu_input_buffer, _input_size);
    cudaMalloc(&_gpu_output_buffer, _output_size);
}

void face_detection::freeBuffers() {
    if (_cuda_stream) {
        // Wait for any remaining operations on the CUDA stream to finish
        cudaStreamSynchronize(_cuda_stream);
        cudaStreamDestroy(_cuda_stream);
        _cuda_stream = nullptr;
    }
    if (_cpu_input_buffer) {
        delete[] _cpu_input_buffer;
        _cpu_input_buffer = nullptr;
    }
    if (_cpu_output_buffer) {
        delete[] _cpu_output_buffer;
        _cpu_output_buffer = nullptr;
    }
    if (_gpu_input_buffer) {
        cudaFree(_gpu_input_buffer);
        _gpu_input_buffer = nullptr;
    }
    if (_gpu_output_buffer) {
        cudaFree(_gpu_output_buffer);
        _gpu_output_buffer = nullptr;
    }
}

cv::Mat face_detection::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;

    int orig_width = image.cols;
    int orig_height = image.rows;

    _letterbox_scale = std::min(
        (float)_input_width / orig_width,
        (float)_input_height / orig_height
    );

    int new_width = (int)(orig_width * _letterbox_scale);
    int new_height = (int)(orig_height * _letterbox_scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    processed = cv::Mat::zeros(cv::Size(_input_width, _input_height), CV_8UC3);
    _letterbox_pad_top = (_input_height - new_height) / 2;
    _letterbox_pad_left = (_input_width - new_width) / 2;

    resized.copyTo(processed(cv::Rect(_letterbox_pad_left, _letterbox_pad_top, new_width, new_height)));

    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    processed.convertTo(processed, CV_32F, 1.0/255.0);

    return processed;
}

std::vector<cv::Rect> face_detection::postprocessOutput(float* output, int img_width, int img_height) {
    std::vector<cv::Rect> bboxes;

    const int num_boxes = 8400;
    const int num_channels = 6;
    const float conf_threshold = 0.5f;

    float best_score = 0.0f;
    int best_index = -1;

    for (int i = 0; i < num_boxes; i++) {
        float* box_data = output + i * num_channels;
        float obj_conf = box_data[4];

        if (obj_conf > conf_threshold) {
            float x1 = box_data[0];
            float y1 = box_data[1];
            float x2 = box_data[2];
            float y2 = box_data[3];
            float area = (x2 - x1) * (y2 - y1);
            float score = obj_conf * sqrt(area);

            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }
    }

    if (best_index >= 0) {
        float* box_data = output + best_index * num_channels;
        float x1 = box_data[0];
        float y1 = box_data[1];
        float x2 = box_data[2];
        float y2 = box_data[3];

        float x1_unpadded = (x1 - _letterbox_pad_left) / _letterbox_scale;
        float y1_unpadded = (y1 - _letterbox_pad_top) / _letterbox_scale;
        float x2_unpadded = (x2 - _letterbox_pad_left) / _letterbox_scale;
        float y2_unpadded = (y2 - _letterbox_pad_top) / _letterbox_scale;

        cv::Rect bbox(
            std::max(0.0f, x1_unpadded),
            std::max(0.0f, y1_unpadded),
            std::max(0.0f, x2_unpadded - x1_unpadded),
            std::max(0.0f, y2_unpadded - y1_unpadded)
        );

        // Crop inside bounds
        bbox = bbox & cv::Rect(0, 0, img_width, img_height);
        if (bbox.width > 0 && bbox.height > 0) {
            bboxes.push_back(bbox);
        }
    }

    return bboxes;
}

std::vector<cv::Rect> face_detection::process(const cv::Mat& image) {
    std::vector<cv::Rect> bboxes;
    if (image.empty()) {
        logger::warn("[FaceDetection] Input image is empty");
        return bboxes;
    }

    try {
        cudaError_t cuda_status = cudaSetDevice(_gpu_id);
        if (cuda_status != cudaSuccess) {
            logger::error("[FaceDetection] Failed to set CUDA device {}: {}", _gpu_id, cudaGetErrorString(cuda_status));
            return bboxes;
        }

        // Preprocess
        cv::Mat processed_image = preprocessImage(image);

        // HWC to CHW
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < _input_height; h++) {
                for (int w = 0; w < _input_width; w++) {
                    _cpu_input_buffer[c * _input_height * _input_width + h * _input_width + w] =
                        processed_image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // Copy input to GPU
        cudaMemcpyAsync(_gpu_input_buffer, _cpu_input_buffer, _input_size, cudaMemcpyHostToDevice, _cuda_stream);

        // Set input shape
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 4;
        input_dims.d[0] = 1;
        input_dims.d[1] = 3;
        input_dims.d[2] = _input_height;
        input_dims.d[3] = _input_width;

        if (!_context->setInputShape("images", input_dims)) {
            logger::error("[FaceDetection] Failed to set input shape");
            return bboxes;
        }

        _context->setTensorAddress("images", _gpu_input_buffer);
        _context->setTensorAddress("output0", _gpu_output_buffer);

        // Inference
        bool success = _context->enqueueV3(_cuda_stream);
        if (success) {
            cudaMemcpyAsync(_cpu_output_buffer, _gpu_output_buffer, _output_size, cudaMemcpyDeviceToHost, _cuda_stream);
            cudaStreamSynchronize(_cuda_stream);

            // Postprocess
            bboxes = postprocessOutput(_cpu_output_buffer, image.cols, image.rows);
        } else {
            logger::error("[FaceDetection] Inference failed");
        }
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during process: {}", e.what());
    }

    return bboxes;
}
