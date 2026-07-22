#include "driver_readiness_estimation_logical.hpp"
#include <flame/log.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>

driver_readiness_estimation_logical::driver_readiness_estimation_logical() {
}

driver_readiness_estimation_logical::~driver_readiness_estimation_logical() {
}

void driver_readiness_estimation_logical::setParameters(
    double ref_yaw, double ref_pitch,
    double sigma_yaw, double sigma_pitch, double t_window,
    double readiness_low, double readiness_moderate, double readiness_high
) {
    _ref_yaw = ref_yaw;
    _ref_pitch = ref_pitch;
    _sigma_yaw = (sigma_yaw > 0.0) ? sigma_yaw : 15.0;
    _sigma_pitch = (sigma_pitch > 0.0) ? sigma_pitch : 10.0;
    _t_window = (t_window > 0.0) ? t_window : 2.0;
    _readiness_low = readiness_low;
    _readiness_moderate = readiness_moderate;
    _readiness_high = readiness_high;

    logger::info("[driver_readiness_logical] Parameters set: ref_yaw={:.1f}deg, ref_pitch={:.1f}deg, sigma_yaw={:.1f}deg, sigma_pitch={:.1f}deg, t_window={:.2f}s, thresholds=[low:{:.2f}, mod:{:.2f}, high:{:.2f}]",
                 _ref_yaw, _ref_pitch, _sigma_yaw, _sigma_pitch, _t_window,
                 _readiness_low, _readiness_moderate, _readiness_high);
}

driver_readiness_logical::LogicalReadinessResult driver_readiness_estimation_logical::process(
    const head_pose::PoseResult& pose_res,
    bool has_pose,
    std::chrono::steady_clock::time_point now
) {
    driver_readiness_logical::LogicalReadinessResult result;
    result.t_window = _t_window;

    double current_yaw = (has_pose && pose_res.success) ? pose_res.euler[1] : 0.0;
    double current_pitch = (has_pose && pose_res.success) ? pose_res.euler[0] : 0.0;
    double theta_yaw = current_yaw - _ref_yaw;
    double theta_pitch = current_pitch - _ref_pitch;

    // 1. Check if current head pose is within the allowable threshold range (+-sigma)
    bool is_within_threshold = (has_pose && pose_res.success &&
                                std::abs(theta_yaw) <= _sigma_yaw &&
                                std::abs(theta_pitch) <= _sigma_pitch);

    // 2. Only push sample into ring buffer if it is within allowable threshold range
    if (is_within_threshold) {
        driver_readiness_logical::HeadPoseData current_data;
        current_data.timestamp = now;
        current_data.has_pose = true;
        current_data.pitch = pose_res.euler[0];
        current_data.yaw = pose_res.euler[1];
        current_data.roll = pose_res.euler[2];

        _ring_buffer.push_back(current_data);
        if (_ring_buffer.size() > MAX_RING_BUFFER_CAPACITY) {
            _ring_buffer.pop_front();
        }
    }

    // 3. Evict samples older than t_window (time diff between oldest sample in buffer and current time > t_window)
    while (!_ring_buffer.empty()) {
        double time_span = std::chrono::duration<double>(now - _ring_buffer.front().timestamp).count();
        if (time_span > _t_window) {
            _ring_buffer.pop_front();
        } else {
            break;
        }
    }

    // 4. Calculate Readiness Score
    // Exponential term based on current relative rotation angle
    double yaw_term = theta_yaw / _sigma_yaw;
    double pitch_term = theta_pitch / _sigma_pitch;
    double exp_val = std::exp(-(yaw_term * yaw_term + pitch_term * pitch_term));
    result.exp_component = exp_val;

    // Calculate t_dwell based on valid buffer count within t_window: count * 0.033s (33ms per frame)
    constexpr double FRAME_INTERVAL_SEC = 0.033; // 33ms
    double t_dwell = static_cast<double>(_ring_buffer.size()) * FRAME_INTERVAL_SEC;

    result.t_dwell = t_dwell;
    // Dwell ratio: t_dwell / t_window (bounded in [0.0, 1.0])
    double ratio = (_t_window > 0.0) ? (t_dwell / _t_window) : 0.0;
    if (ratio > 1.0) ratio = 1.0;
    result.dwell_ratio = ratio;

    // Final Readiness Score
    result.readiness_score = exp_val * ratio;
    result.valid = true;

    // Categorization:
    // 0.0 ~ readiness_low -> "low"
    // readiness_low ~ readiness_moderate -> "moderate"
    // readiness_moderate ~ readiness_high -> "high"
    if (result.readiness_score > _readiness_moderate) {
        result.category = "high";
    } else if (result.readiness_score > _readiness_low) {
        result.category = "moderate";
    } else {
        result.category = "low";
    }

    // Log readiness score and category
    logger::info("[driver_readiness_logical] Readiness Score: {:.4f} [{}] (exp={:.4f}, dwell_ratio={:.4f}, t_dwell={:.3f}s/{:.2f}s, buffer_cnt={}, theta_yaw={:.1f}, theta_pitch={:.1f})",
                 result.readiness_score, result.category, result.exp_component, result.dwell_ratio,
                 result.t_dwell, result.t_window, _ring_buffer.size(), theta_yaw, theta_pitch);

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
    ss << "Driver Readiness (Logical): " << result.category 
       << " (Score: " << result.readiness_score << ", Dwell: " << result.t_dwell << "s/" << result.t_window << "s)";

    std::string text = ss.str();
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
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

    // Semi-transparent background box for legibility at bottom-right
    cv::Rect box(pos_x - 5, pos_y - text_size.height - 5, text_size.width + 10, text_size.height + baseline + 10);
    cv::Mat overlay;
    image.copyTo(overlay);
    cv::rectangle(overlay, box, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.5, image, 0.5, 0, image);

    cv::putText(image, text, cv::Point(pos_x, pos_y), font_face, font_scale, color, thickness, cv::LINE_AA);
}
