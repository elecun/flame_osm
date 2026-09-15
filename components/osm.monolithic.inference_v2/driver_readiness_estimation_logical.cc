#include "driver_readiness_estimation_logical.hpp"
#include <flame/log.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <numeric>

driver_readiness_estimation_logical::driver_readiness_estimation_logical() {
}

driver_readiness_estimation_logical::~driver_readiness_estimation_logical() {
}

void driver_readiness_estimation_logical::setParameters(
    const cv::Point2f& steer_ref,
    const driver_readiness_logical::GaussianParam& g_yaw,
    const driver_readiness_logical::GaussianParam& g_pitch,
    const driver_readiness_logical::GaussianParam& g_steer_lw,
    const driver_readiness_logical::GaussianParam& g_steer_rw,
    const driver_readiness_logical::GaussianParam& g_lw_rw,
    size_t window_size,
    double readiness_low,
    double readiness_high
) {
    _steer_ref = steer_ref;
    _g_yaw = g_yaw;
    _g_pitch = g_pitch;
    _g_steer_lw = g_steer_lw;
    _g_steer_rw = g_steer_rw;
    _g_lw_rw = g_lw_rw;
    _window_size = (window_size > 0) ? window_size : 30;
    _readiness_low = readiness_low;
    _readiness_high = readiness_high;

    _score_window.clear();

    logger::info("[driver_readiness_logical] Parameters configured: steer_ref=({:.1f}, {:.1f}), "
                 "yaw[mean={:.1f}, var={:.1f}], pitch[mean={:.1f}, var={:.1f}], "
                 "steer_lw[mean={:.1f}, var={:.1f}], steer_rw[mean={:.1f}, var={:.1f}], lw_rw[mean={:.1f}, var={:.1f}], "
                 "window_size={}, thresholds=[low:{:.2f}, high:{:.2f}]",
                 _steer_ref.x, _steer_ref.y,
                 _g_yaw.mean, _g_yaw.var, _g_pitch.mean, _g_pitch.var,
                 _g_steer_lw.mean, _g_steer_lw.var, _g_steer_rw.mean, _g_steer_rw.var, _g_lw_rw.mean, _g_lw_rw.var,
                 _window_size, _readiness_low, _readiness_high);
}

double driver_readiness_estimation_logical::computeGaussian(double x, double mean, double var) {
    if (var <= 1e-6) return 0.0;
    double diff = x - mean;
    return std::exp(-(diff * diff) / (2.0 * var));
}

double driver_readiness_estimation_logical::computeAngleGaussian(double angle, double mean, double var) {
    if (var <= 1e-6) return 0.0;
    double diff = angle - mean;
    // Wrap around [-180, 180] degrees
    while (diff > 180.0) diff -= 360.0;
    while (diff < -180.0) diff += 360.0;
    return std::exp(-(diff * diff) / (2.0 * var));
}

driver_readiness_logical::LogicalReadinessResult driver_readiness_estimation_logical::process(
    const head_pose::PoseResult& pose_res,
    bool has_pose,
    const std::vector<body_pose::PoseResult>& body_poses
) {
    driver_readiness_logical::LogicalReadinessResult result;

    // 1. Head Pose components (yaw & pitch)
    if (has_pose && pose_res.success) {
        double current_pitch = pose_res.euler[0];
        double current_yaw = pose_res.euler[1];

        result.score_yaw = computeAngleGaussian(current_yaw, _g_yaw.mean, _g_yaw.var);
        result.score_pitch = computeAngleGaussian(current_pitch, _g_pitch.mean, _g_pitch.var);
    } else {
        result.score_yaw = 0.0;
        result.score_pitch = 0.0;
    }

    // 2. Body Pose components (wrist distances)
    // COCO Keypoints: index 9 = left_wrist (lw), index 10 = right_wrist (rw)
    bool has_lw = false;
    bool has_rw = false;
    cv::Point2f pt_lw(0.0f, 0.0f);
    cv::Point2f pt_rw(0.0f, 0.0f);

    if (!body_poses.empty()) {
        const auto& pose = body_poses[0];
        if (pose.keypoints.size() > 9 && pose.keypoints[9].confidence > 0.2f) {
            pt_lw = cv::Point2f(pose.keypoints[9].x, pose.keypoints[9].y);
            has_lw = true;
        }
        if (pose.keypoints.size() > 10 && pose.keypoints[10].confidence > 0.2f) {
            pt_rw = cv::Point2f(pose.keypoints[10].x, pose.keypoints[10].y);
            has_rw = true;
        }
    }

    // Distance between left wrist and steer_ref
    if (has_lw) {
        double dist_steer_lw = cv::norm(pt_lw - _steer_ref);
        result.score_steer_lw = computeGaussian(dist_steer_lw, _g_steer_lw.mean, _g_steer_lw.var);
    } else {
        result.score_steer_lw = 0.0;
    }

    // Distance between right wrist and steer_ref
    if (has_rw) {
        double dist_steer_rw = cv::norm(pt_rw - _steer_ref);
        result.score_steer_rw = computeGaussian(dist_steer_rw, _g_steer_rw.mean, _g_steer_rw.var);
    } else {
        result.score_steer_rw = 0.0;
    }

    // Distance between left wrist and right wrist
    if (has_lw && has_rw) {
        double dist_lw_rw = cv::norm(pt_lw - pt_rw);
        result.score_lw_rw = computeGaussian(dist_lw_rw, _g_lw_rw.mean, _g_lw_rw.var);
    } else {
        result.score_lw_rw = 0.0;
    }

    // 3. Raw readiness score: average of 5 unnormalized Gaussian scores [0.0 ~ 1.0]
    result.raw_score = (result.score_yaw + result.score_pitch +
                        result.score_steer_lw + result.score_steer_rw + result.score_lw_rw) / 5.0;

    // 4. Moving window average
    _score_window.push_back(result.raw_score);
    if (_score_window.size() > _window_size) {
        _score_window.pop_front();
    }

    double sum = std::accumulate(_score_window.begin(), _score_window.end(), 0.0);
    result.readiness_score = _score_window.empty() ? 0.0 : (sum / _score_window.size());
    result.valid = true;

    // 5. Categorization based on window-averaged score
    // < readiness_low -> "low"
    // [readiness_low, readiness_high] -> "moderate"
    // > readiness_high -> "high"
    if (result.readiness_score > _readiness_high) {
        result.category = "high";
    } else if (result.readiness_score >= _readiness_low) {
        result.category = "moderate";
    } else {
        result.category = "low";
    }

    return result;
}

void driver_readiness_estimation_logical::drawResult(
    cv::Mat& image,
    const driver_readiness_logical::LogicalReadinessResult& result
) {
    if (!result.valid) {
        return;
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Readiness : " << result.category 
       << " (Score: " << result.readiness_score << " [raw: " << result.raw_score << "])";

    std::string text = ss.str();
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

    int margin_x = 20;
    int margin_y = 20;
    int pos_x = image.cols - text_size.width - margin_x;
    int pos_y = image.rows - margin_y;

    if (pos_x < 10) pos_x = 10;

    cv::Scalar color(0, 0, 255); // Red for low
    if (result.category == "high") {
        color = cv::Scalar(0, 255, 0); // Green for high
    } else if (result.category == "moderate") {
        color = cv::Scalar(0, 255, 255); // Yellow for moderate
    }

    cv::Rect box(pos_x - 5, pos_y - text_size.height - 5, text_size.width + 10, text_size.height + baseline + 10);
    cv::Mat overlay;
    image.copyTo(overlay);
    cv::rectangle(overlay, box, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.5, image, 0.5, 0, image);

    cv::putText(image, text, cv::Point(pos_x, pos_y), font_face, font_scale, color, thickness, cv::LINE_AA);
}

