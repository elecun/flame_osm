#ifndef OSM_MONOLITHIC_INFERENCE_DRIVER_READINESS_ESTIMATION_LOGICAL_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_DRIVER_READINESS_ESTIMATION_LOGICAL_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <deque>
#include "head_pose_estimation_from_2d.hpp"

namespace driver_readiness_logical {
    struct HeadPoseData {
        double pitch{0.0};
        double yaw{0.0};
        double roll{0.0};
        std::chrono::steady_clock::time_point timestamp;
        bool has_pose{false};
    };

    struct LogicalReadinessResult {
        double readiness_score{0.0}; // Gaussian decay-based readiness score [0.0 ~ 1.0]
        double exp_component{0.0};   // exp(-((yaw/sigma_yaw)^2 + (pitch/sigma_pitch)^2))
        double dwell_ratio{0.0};      // t_dwell / t_window
        double t_dwell{0.0};          // Cumulative time maintaining inside threshold (seconds)
        double t_window{2.0};         // Observed time window (seconds)
        std::string category{"none"}; // "low", "moderate", "high", or "none"
        bool valid{false};
    };
}

class driver_readiness_estimation_logical {
public:
    driver_readiness_estimation_logical();
    ~driver_readiness_estimation_logical();

    void setParameters(
        double ref_yaw, double ref_pitch,
        double sigma_yaw, double sigma_pitch, double t_window,
        double readiness_low = 0.2, double readiness_moderate = 0.6, double readiness_high = 1.0
    );

    // Process new head pose data with current timestamp
    driver_readiness_logical::LogicalReadinessResult process(
        const head_pose::PoseResult& pose_res,
        bool has_pose,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    // Draw logical readiness score on image
    void drawResult(cv::Mat& image, const driver_readiness_logical::LogicalReadinessResult& result);

private:
    double _ref_yaw{0.0};            // Reference forward yaw angle in degrees
    double _ref_pitch{0.0};          // Reference forward pitch angle in degrees
    double _sigma_yaw{15.0};         // Allowable yaw threshold in degrees (+-15.0)
    double _sigma_pitch{10.0};       // Allowable pitch threshold in degrees (+-10.0)
    double _t_window{2.0};           // Time window in seconds (2.0s)
    double _readiness_low{0.2};      // Threshold for low category (0.2)
    double _readiness_moderate{0.6}; // Threshold for moderate category (0.6)
    double _readiness_high{1.0};     // Threshold for high category (1.0)
    
    static constexpr size_t MAX_RING_BUFFER_CAPACITY = 50; // Buffer capacity for 30fps stream
    std::deque<driver_readiness_logical::HeadPoseData> _ring_buffer;
};

#endif
