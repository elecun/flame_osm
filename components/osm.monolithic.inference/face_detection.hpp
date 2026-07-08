#ifndef OSM_MONOLITHIC_INFERENCE_FACE_DETECTION_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_FACE_DETECTION_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

class FaceTRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class face_detection {
public:
    face_detection();
    ~face_detection();

    // Load the TensorRT engine
    bool loadModel(const std::string& model_path, int gpu_id = 0);

    // Process single image, return bounding boxes of detected faces
    std::vector<cv::Rect> process(const cv::Mat& image);

private:
    void allocateBuffers();
    void freeBuffers();
    cv::Mat preprocessImage(const cv::Mat& image);
    std::vector<cv::Rect> postprocessOutput(float* output, int img_width, int img_height);

private:
    /* TensorRT related */
    std::unique_ptr<nvinfer1::IRuntime> _runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> _engine;
    std::unique_ptr<nvinfer1::IExecutionContext> _context;
    FaceTRTLogger _logger;

    /* CUDA memory */
    void* _gpu_input_buffer = nullptr;
    void* _gpu_output_buffer = nullptr;
    float* _cpu_input_buffer = nullptr;
    float* _cpu_output_buffer = nullptr;
    cudaStream_t _cuda_stream = nullptr;

    /* Model parameters */
    int _input_width = 640;
    int _input_height = 640;
    size_t _input_size = 0;
    size_t _output_size = 0;
    int _gpu_id = 0;

    /* Letterbox padding info (updated during preprocessing) */
    float _letterbox_scale = 1.0f;
    int _letterbox_pad_left = 0;
    int _letterbox_pad_top = 0;
};

#endif
