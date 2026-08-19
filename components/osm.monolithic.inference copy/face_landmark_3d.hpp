#ifndef OSM_MONOLITHIC_INFERENCE_FACE_LANDMARK_3D_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_FACE_LANDMARK_3D_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

namespace face_landmark_3d_ns {
    struct Landmark3DResult {
        std::vector<cv::Point3f> points_3d;
        std::vector<cv::Point2f> points_2d;
        cv::Rect crop_bbox;
    };
}

class face_landmark_3d {
public:
    face_landmark_3d();
    ~face_landmark_3d();

    // Load the FAN 3D TorchScript model
    bool loadModel(const std::string& model_path, int gpu_id = 0);

    // Process face bounding boxes on original image and return 3D landmarks (68 points for each face)
    std::vector<face_landmark_3d_ns::Landmark3DResult> process(const cv::Mat& image, const std::vector<cv::Rect>& bboxes);

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _input_width = 256;
    int _input_height = 256;
    int _gpu_id = 0;
};

#endif
