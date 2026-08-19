#ifndef OSM_MONOLITHIC_INFERENCE_HEAD_POSE_ESTIMATION_FROM_3D_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_HEAD_POSE_ESTIMATION_FROM_3D_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "head_pose_estimation_from_2d.hpp"

class head_pose_estimation_from_3d {
public:
    head_pose_estimation_from_3d();
    ~head_pose_estimation_from_3d();

    // Load camera calibration JSON
    bool loadCalibration(const std::string& json_path);

    // Estimate head pose from 68 3D facial landmarks (Point3f) and image dimensions
    head_pose::PoseResult estimate(const std::vector<cv::Point3f>& landmarks_68_3d, const cv::Size& image_size);

    // Draw 3D coordinate axes on target image
    void drawPoseAxes(cv::Mat& image, const head_pose::PoseResult& result, const cv::Size& orig_image_size, float scale_x = 1.0f, float scale_y = 1.0f);

private:
    std::vector<cv::Point3d> _model_points;
    cv::Mat _camera_matrix;
    cv::Mat _dist_coeffs;
    bool _has_calibration{false};
};

#endif
