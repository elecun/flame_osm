#ifndef OSM_MONOLITHIC_INFERENCE_DRIVER_READINESS_ESTIMATION_LOGICAL_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_DRIVER_READINESS_ESTIMATION_LOGICAL_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <deque>
#include <string>
#include "face_analysis_e2e.hpp"
#include "body_pose_estimation.hpp"

namespace driver_readiness_logical {
    struct GaussianParam {
        double mean{0.0};
        double var{1.0}; // variance (sigma^2)
    };

    struct LogicalReadinessResult {
        // Individual unnormalized Gaussian component scores [0.0 ~ 1.0]
        double score_yaw{0.0};
        double score_pitch{0.0};
        double score_steer_lw{0.0};
        double score_steer_rw{0.0};
        double score_lw_rw{0.0};

        double raw_score{0.0};        // Instantaneous average of 5 scores [0.0 ~ 1.0]
        double readiness_score{0.0};   // Moving-averaged readiness score [0.0 ~ 1.0]
        std::string category{"low"};  // "low", "moderate", "high"
        bool valid{false};
    };
}

class driver_readiness_estimation_logical {
public:
    driver_readiness_estimation_logical();
    ~driver_readiness_estimation_logical();

    void setParameters(
        const cv::Point2f& steer_ref,
        const driver_readiness_logical::GaussianParam& g_yaw,
        const driver_readiness_logical::GaussianParam& g_pitch,
        const driver_readiness_logical::GaussianParam& g_steer_lw,
        const driver_readiness_logical::GaussianParam& g_steer_rw,
        const driver_readiness_logical::GaussianParam& g_lw_rw,
        size_t window_size = 30,
        double readiness_low = 0.2,
        double readiness_high = 0.6
    );

    // Process head pose and body pose keypoints
    driver_readiness_logical::LogicalReadinessResult process(
        const head_pose::PoseResult& pose_res,
        bool has_pose,
        const std::vector<body_pose::PoseResult>& body_poses
    );

    // Draw logical readiness score panel on image
    void drawResult(cv::Mat& image, const driver_readiness_logical::LogicalReadinessResult& result);

    const cv::Point2f& getSteerRef() const { return _steer_ref; }
    double getReadinessLow() const { return _readiness_low; }
    double getReadinessHigh() const { return _readiness_high; }

private:
    static double computeGaussian(double x, double mean, double var);
    static double computeAngleGaussian(double angle, double mean, double var);

    cv::Point2f _steer_ref{640.0f, 800.0f};
    driver_readiness_logical::GaussianParam _g_yaw{0.0, 225.0};
    driver_readiness_logical::GaussianParam _g_pitch{0.0, 100.0};
    driver_readiness_logical::GaussianParam _g_steer_lw{200.0, 10000.0};
    driver_readiness_logical::GaussianParam _g_steer_rw{200.0, 10000.0};
    driver_readiness_logical::GaussianParam _g_lw_rw{300.0, 10000.0};

    size_t _window_size{30};
    double _readiness_low{0.2};
    double _readiness_high{0.6};

    std::deque<double> _score_window;
};

#endif
