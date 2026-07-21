#include "head_pose_estimation.hpp"
#include <flame/log.hpp>
#include <cmath>

head_pose_estimation::head_pose_estimation() {
    // Standard 3D facial model points (in mm) based on 68-point model
    _model_points = {
        cv::Point3d(0.0, 0.0, 0.0),             // Nose tip (30)
        cv::Point3d(0.0, -330.0, -65.0),        // Chin (8)
        cv::Point3d(-225.0, 170.0, -135.0),     // Left eye left corner (36)
        cv::Point3d(225.0, 170.0, -135.0),      // Right eye right corner (45)
        cv::Point3d(-150.0, -150.0, -125.0),    // Left mouth corner (48)
        cv::Point3d(150.0, -150.0, -125.0)      // Right mouth corner (54)
    };
}

head_pose_estimation::~head_pose_estimation() {}

head_pose::PoseResult head_pose_estimation::estimate(const std::vector<cv::Point2f>& landmarks_68, const cv::Size& image_size) {
    head_pose::PoseResult res;
    if (landmarks_68.size() < 68 || image_size.width <= 0 || image_size.height <= 0) {
        return res;
    }

    try {
        // Extract 6 key landmark points: nose tip (30), chin (8), left eye corner (36), right eye corner (45), left mouth (48), right mouth (54)
        std::vector<cv::Point2d> image_points = {
            cv::Point2d(landmarks_68[30].x, landmarks_68[30].y),
            cv::Point2d(landmarks_68[8].x, landmarks_68[8].y),
            cv::Point2d(landmarks_68[36].x, landmarks_68[36].y),
            cv::Point2d(landmarks_68[45].x, landmarks_68[45].y),
            cv::Point2d(landmarks_68[48].x, landmarks_68[48].y),
            cv::Point2d(landmarks_68[54].x, landmarks_68[54].y)
        };

        // Estimate camera intrinsic matrix
        double focal_length = (double)image_size.width;
        cv::Point2d center((double)image_size.width / 2.0, (double)image_size.height / 2.0);
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
            focal_length, 0.0, center.x,
            0.0, focal_length, center.y,
            0.0, 0.0, 1.0);
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

        // Solve PnP with RANSAC
        bool success = cv::solvePnPRansac(
            _model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            res.rvec,
            res.tvec,
            false,
            100,
            8.0,
            0.99,
            cv::noArray(),
            cv::SOLVEPNP_ITERATIVE
        );

        if (!success) {
            success = cv::solvePnP(_model_points, image_points, camera_matrix, dist_coeffs, res.rvec, res.tvec, false, cv::SOLVEPNP_ITERATIVE);
        }

        if (success && !res.rvec.empty() && !res.tvec.empty()) {
            res.success = true;

            // Convert rotation vector to 3x3 matrix
            cv::Mat R;
            cv::Rodrigues(res.rvec, R);

            double r00 = R.at<double>(0, 0);
            double r10 = R.at<double>(1, 0);
            double r20 = R.at<double>(2, 0);
            double r21 = R.at<double>(2, 1);
            double r22 = R.at<double>(2, 2);
            double r12 = R.at<double>(1, 2);
            double r11 = R.at<double>(1, 1);

            double sy = std::sqrt(r00 * r00 + r10 * r10);
            bool singular = sy < 1e-6;

            double pitch, yaw, roll;
            if (!singular) {
                pitch = std::atan2(r21, r22);
                yaw   = std::atan2(-r20, sy);
                roll  = std::atan2(r10, r00);
            } else {
                pitch = std::atan2(-r12, r11);
                yaw   = std::atan2(-r20, sy);
                roll  = 0.0;
            }

            // Convert radians to degrees
            res.euler = cv::Vec3d(
                pitch * 180.0 / M_PI,
                yaw   * 180.0 / M_PI,
                roll  * 180.0 / M_PI
            );
        }
    }
    catch (const std::exception& e) {
        logger::error("[HeadPoseEstimation] Exception during pose estimation: {}", e.what());
    }

    return res;
}

void head_pose_estimation::drawPoseAxes(cv::Mat& image, const head_pose::PoseResult& result, const cv::Size& orig_image_size, float scale_x, float scale_y) {
    if (!result.success || result.rvec.empty() || result.tvec.empty()) {
        return;
    }

    try {
        double focal_length = (double)orig_image_size.width;
        cv::Point2d center((double)orig_image_size.width / 2.0, (double)orig_image_size.height / 2.0);
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
            focal_length, 0.0, center.x,
            0.0, focal_length, center.y,
            0.0, 0.0, 1.0);
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

        double axis_length = 300.0;
        std::vector<cv::Point3d> axis_points_3d = {
            cv::Point3d(0.0, 0.0, 0.0),            // Origin (Nose tip)
            cv::Point3d(axis_length, 0.0, 0.0),    // X-axis (Red)
            cv::Point3d(0.0, axis_length, 0.0),    // Y-axis (Green)
            cv::Point3d(0.0, 0.0, axis_length)     // Z-axis (Blue)
        };

        std::vector<cv::Point2d> axis_points_2d;
        cv::projectPoints(axis_points_3d, result.rvec, result.tvec, camera_matrix, dist_coeffs, axis_points_2d);

        if (axis_points_2d.size() >= 4) {
            cv::Point p0((int)std::round(axis_points_2d[0].x * scale_x), (int)std::round(axis_points_2d[0].y * scale_y));
            cv::Point px((int)std::round(axis_points_2d[1].x * scale_x), (int)std::round(axis_points_2d[1].y * scale_y));
            cv::Point py((int)std::round(axis_points_2d[2].x * scale_x), (int)std::round(axis_points_2d[2].y * scale_y));
            cv::Point pz((int)std::round(axis_points_2d[3].x * scale_x), (int)std::round(axis_points_2d[3].y * scale_y));

            int thickness = 3;
            cv::line(image, p0, px, cv::Scalar(0, 0, 255), thickness);   // X-axis: Red
            cv::line(image, p0, py, cv::Scalar(0, 255, 0), thickness);   // Y-axis: Green
            cv::line(image, p0, pz, cv::Scalar(255, 0, 0), thickness);   // Z-axis: Blue

            cv::putText(image, "X", px, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            cv::putText(image, "Y", py, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            cv::putText(image, "Z", pz, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        }
    }
    catch (const std::exception& e) {
        logger::error("[HeadPoseEstimation] Exception during drawPoseAxes: {}", e.what());
    }
}
