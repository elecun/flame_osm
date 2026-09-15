/**
 * @file face_detection.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief 
 * @version 0.1
 * @date 2026-09-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef OSM_MONOLITHIC_INFERENCE_FACE_DETECTION_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_FACE_DETECTION_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

struct FaceBox {
    cv::Rect bbox;          // 1:1 square crop box with padding_scale (clamped to image bounds)
    float score = 0.0f;     // Detection confidence
    cv::Rect raw_bbox;      // Raw bounding box before 1:1 padding
};

class face_detection {
public:
    face_detection();
    ~face_detection();

    // Load the model
    bool loadModel(const std::string& model_path, int gpu_id = 0);

    // Detect faces matching demo_e2e_test.py (letterbox, aspect-ratio preserved, 1:1 padding_scale square crop)
    std::vector<FaceBox> detect(const cv::Mat& image, float conf_thresh = 0.7f, float nms_thresh = 0.45f, float padding_scale = 1.25f);

    // Letterbox preprocessing
    static cv::Mat letterbox(const cv::Mat& img, int new_shape, float& out_ratio, float& out_dw, float& out_dh);

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _input_width = 640;
    int _input_height = 640;
    int _gpu_id = 0;
};

#endif
