#include "gaze_following.hpp"
#include <flame/log.hpp>
#include <algorithm>
#include <limits>

bool gaze_following_model::loadModel(const std::string& model_path, int gpu_id) {
    try {
        _device = (torch::cuda::is_available() && gpu_id >= 0)
            ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU);
        _module = torch::jit::load(model_path, _device);
        _module.eval();
        _is_loaded = true;
        logger::info("[gaze_following] Loaded Gazelle model from {} on {}", model_path, _device.str());
        return true;
    } catch (const std::exception& e) {
        logger::error("[gaze_following] Failed to load model from {}: {}", model_path, e.what());
        _is_loaded = false;
        return false;
    }
}

gaze_following::Result gaze_following_model::process(const cv::Mat& image, const std::vector<cv::Rect>& face_bboxes, float inout_threshold) {
    gaze_following::Result result;
    if (!_is_loaded || image.empty() || face_bboxes.empty()) return result;

    try {
        cv::Mat resized, rgb, input_mat;
        cv::resize(image, resized, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(input_mat, CV_32FC3, 1.0 / 255.0);
        cv::subtract(input_mat, cv::Scalar(0.485f, 0.456f, 0.406f), input_mat);
        cv::divide(input_mat, cv::Scalar(0.229f, 0.224f, 0.225f), input_mat);

        auto image_tensor = torch::from_blob(input_mat.data, {1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3}, torch::kFloat32)
            .permute({0, 3, 1, 2}).to(_device);
        std::vector<float> bbox_data;
        bbox_data.reserve(face_bboxes.size() * 4);
        for (const auto& bbox : face_bboxes) {
            const float x1 = std::clamp(static_cast<float>(bbox.x) / image.cols, 0.0f, 1.0f);
            const float y1 = std::clamp(static_cast<float>(bbox.y) / image.rows, 0.0f, 1.0f);
            const float x2 = std::clamp(static_cast<float>(bbox.x + bbox.width) / image.cols, 0.0f, 1.0f);
            const float y2 = std::clamp(static_cast<float>(bbox.y + bbox.height) / image.rows, 0.0f, 1.0f);
            bbox_data.insert(bbox_data.end(), {x1, y1, x2, y2});
        }
        auto bbox_tensor = torch::from_blob(bbox_data.data(), {static_cast<long>(face_bboxes.size()), 4}, torch::kFloat32).to(_device);

        torch::NoGradGuard no_grad;
        auto output = _module.forward({image_tensor, bbox_tensor}).toTuple()->elements();
        auto heatmaps = output[0].toTensor().to(torch::kCPU).contiguous();
        auto inouts = output[1].toTensor().to(torch::kCPU).contiguous();
        auto heatmap_acc = heatmaps.accessor<float, 3>();
        const float* inout_ptr = inouts.data_ptr<float>();

        for (size_t i = 0; i < face_bboxes.size(); ++i) {
            gaze_following::PersonResult person;
            person.face_bbox = face_bboxes[i];
            person.heatmap = cv::Mat(heatmaps.size(1), heatmaps.size(2), CV_32F);
            float max_value = -std::numeric_limits<float>::infinity();
            cv::Point max_point;
            for (int y = 0; y < person.heatmap.rows; ++y) {
                for (int x = 0; x < person.heatmap.cols; ++x) {
                    const float value = heatmap_acc[i][y][x];
                    person.heatmap.at<float>(y, x) = value;
                    if (value > max_value) { max_value = value; max_point = cv::Point(x, y); }
                }
            }
            person.target_normalized = cv::Point2f(
                (max_point.x + 0.5f) / person.heatmap.cols,
                (max_point.y + 0.5f) / person.heatmap.rows);
            person.inout_score = inout_ptr[i];
            person.is_in_frame = person.inout_score >= inout_threshold;
            result.people.push_back(std::move(person));
        }
        result.valid = true;
    } catch (const std::exception& e) {
        logger::error("[gaze_following] Inference failed: {}", e.what());
    }
    return result;
}

void gaze_following_model::drawResult(cv::Mat& image, const gaze_following::Result& result) {
    if (!result.valid || image.empty()) return;
    cv::Mat combined = cv::Mat::zeros(image.size(), CV_32F);
    for (const auto& person : result.people) {
        if (!person.is_in_frame) continue;
        cv::Mat resized;
        cv::resize(person.heatmap, resized, image.size(), 0, 0, cv::INTER_LINEAR);
        cv::max(combined, resized, combined);
    }
    double max_value = 0.0;
    cv::minMaxLoc(combined, nullptr, &max_value);
    if (max_value > 0.0) {
        cv::Mat normalized, color, mask;
        combined.convertTo(normalized, CV_8U, 255.0 / max_value);
        cv::applyColorMap(normalized, color, cv::COLORMAP_JET);
        cv::threshold(normalized, mask, 30, 255, cv::THRESH_BINARY);
        cv::Mat blended;
        cv::addWeighted(image, 0.55, color, 0.45, 0.0, blended);
        blended.copyTo(image, mask);
    }
    for (size_t i = 0; i < result.people.size(); ++i) {
        const auto& person = result.people[i];
        const cv::Scalar color = (i % 2 == 0) ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
        const cv::Point eye(person.face_bbox.x + person.face_bbox.width / 2,
                            person.face_bbox.y + static_cast<int>(person.face_bbox.height * 0.35f));
        const cv::Point target(static_cast<int>(person.target_normalized.x * image.cols),
                               static_cast<int>(person.target_normalized.y * image.rows));
        if (person.is_in_frame) {
            cv::arrowedLine(image, eye, target, color, 2, cv::LINE_AA, 0, 0.03);
            cv::circle(image, target, 14, color, 2, cv::LINE_AA);
            cv::drawMarker(image, target, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 14, 2, cv::LINE_AA);
        }
    }
}
