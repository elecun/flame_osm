#ifndef OSM_MONOLITHIC_INFERENCE_HEAD_POSE_ESTIMATION_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_HEAD_POSE_ESTIMATION_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <vector>

namespace head_pose {
    struct PoseResult {
        cv::Mat rvec;        // 3x1 double rotation vector
        cv::Mat tvec;        // 3x1 double translation vector
        cv::Vec3d euler;     // (pitch, yaw, roll) in degrees
        bool success = false;
    };
}

class head_pose_estimation {
public:
    head_pose_estimation();
    ~head_pose_estimation();

    // Estimate head pose from 68 2D facial landmarks and image dimensions
    head_pose::PoseResult estimate(const std::vector<cv::Point2f>& landmarks_68, const cv::Size& image_size);

    // Draw 3D coordinate axes on target image
    void drawPoseAxes(cv::Mat& image, const head_pose::PoseResult& result, const cv::Size& orig_image_size, float scale_x = 1.0f, float scale_y = 1.0f);

private:
    std::vector<cv::Point3d> _model_points;
};

#endif
