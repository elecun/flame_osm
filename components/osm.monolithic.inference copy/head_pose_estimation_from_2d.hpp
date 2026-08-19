#ifndef OSM_MONOLITHIC_INFERENCE_HEAD_POSE_ESTIMATION_FROM_2D_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_HEAD_POSE_ESTIMATION_FROM_2D_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace head_pose {
    struct PoseResult {
        cv::Mat rvec;                        // 3x1 double rotation vector
        cv::Mat tvec;                        // 3x1 double translation vector
        cv::Vec3d euler;                     // (pitch, yaw, roll) in degrees
        cv::Point2f nose_tip_2d{0.0f, 0.0f}; // Anchor coordinate at 2D nose tip
        bool success = false;
    };
}

class head_pose_estimation_from_2d {
public:
    head_pose_estimation_from_2d();
    ~head_pose_estimation_from_2d();

    // Load camera calibration JSON (camera_matrix, distortion_coefficients)
    bool loadCalibration(const std::string& json_path);

    // Estimate head pose from 68 2D facial landmarks and image dimensions
    head_pose::PoseResult estimate(const std::vector<cv::Point2f>& landmarks_68, const cv::Size& image_size);

    // Draw 3D coordinate axes on target image anchored at nose tip
    void drawPoseAxes(cv::Mat& image, const head_pose::PoseResult& result, const cv::Size& orig_image_size, float scale_x = 1.0f, float scale_y = 1.0f);

private:
    std::vector<cv::Point3d> _model_points;
    cv::Mat _camera_matrix;
    cv::Mat _dist_coeffs;
    bool _has_calibration{false};
};

#endif
