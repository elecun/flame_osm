#ifndef OSM_MONOLITHIC_INFERENCE_FACE_LANDMARK_2D_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_FACE_LANDMARK_2D_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

namespace face_landmark {
    struct LandmarkResult {
        std::vector<cv::Point2f> points;
        cv::Rect crop_bbox; // The actual square region used by FAN in original image space
    };
}

class face_landmark_2d {
public:
    face_landmark_2d();
    ~face_landmark_2d();

    // Load the FAN TorchScript model
    bool loadModel(const std::string& model_path, int gpu_id = 0);

    // Process face bounding boxes on original image and return 2D landmarks (68 points for each face)
    std::vector<face_landmark::LandmarkResult> process(const cv::Mat& image, const std::vector<cv::Rect>& bboxes);

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _input_width = 256;
    int _input_height = 256;
    int _gpu_id = 0;
};

#endif
