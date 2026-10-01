#ifndef OSM_MONOLITHIC_INFERENCE_V2_GAZE_FOLLOWING_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_V2_GAZE_FOLLOWING_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>

namespace gaze_following {
struct PersonResult {
    cv::Rect face_bbox;
    cv::Mat heatmap;                 // 64x64 gaze likelihood map
    float inout_score = 0.0f;        // probability that gaze target is in frame
    cv::Point2f target_normalized;   // heatmap peak in [0, 1] image coordinates
    bool is_in_frame = false;
};

struct Result {
    std::vector<PersonResult> people;
    bool valid = false;
};
}

class gaze_following_model {
public:
    bool loadModel(const std::string& model_path, int gpu_id = 0);
    gaze_following::Result process(const cv::Mat& image, const std::vector<cv::Rect>& face_bboxes, float inout_threshold);
    static void drawResult(cv::Mat& image, const gaze_following::Result& result);

private:
    torch::jit::script::Module _module;
    torch::Device _device{torch::kCPU};
    bool _is_loaded{false};
    static constexpr int MODEL_INPUT_SIZE = 448;
};

#endif
