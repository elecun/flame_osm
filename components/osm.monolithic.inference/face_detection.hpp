#ifndef OSM_MONOLITHIC_INFERENCE_FACE_DETECTION_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_FACE_DETECTION_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

class face_detection {
public:
    face_detection();
    ~face_detection();

    // Load the model
    bool loadModel(const std::string& model_path, int gpu_id = 0);

    // Process image and return bounding boxes
    std::vector<cv::Rect> process(const cv::Mat& image, float nms_threshold = 0.45f);

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _input_width = 640;
    int _input_height = 640;
    int _gpu_id = 0;
};

#endif
