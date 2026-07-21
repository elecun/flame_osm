#include "head_pose_estimation_from_2d.hpp"
#include <flame/log.hpp>
#include <dep/json.hpp>
#include <fstream>
#include <filesystem>
#include <cmath>

using json = nlohmann::json;

head_pose_estimation_from_2d::head_pose_estimation_from_2d() {
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

head_pose_estimation_from_2d::~head_pose_estimation_from_2d() {}

bool head_pose_estimation_from_2d::loadCalibration(const std::string& json_path) {
    if (json_path.empty() || !std::filesystem::exists(json_path)) {
        logger::warn("[HeadPoseEstimation2D] Calibration JSON file not found: '{}'", json_path);
        _has_calibration = false;
        return false;
    }

    try {
        std::ifstream f(json_path);
        json calib_json = json::parse(f);

        if (calib_json.contains("camera_matrix") && calib_json["camera_matrix"].is_array()) {
            _camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            const auto& cm = calib_json["camera_matrix"];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    _camera_matrix.at<double>(r, c) = cm[r][c].get<double>();
                }
            }
        } else if (calib_json.contains("intrinsics")) {
            const auto& intr = calib_json["intrinsics"];
            double fx = intr.value("fx", 0.0);
            double fy = intr.value("fy", 0.0);
            double cx = intr.value("cx", 0.0);
            double cy = intr.value("cy", 0.0);
            _camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        }

        if (calib_json.contains("distortion_coefficients") && calib_json["distortion_coefficients"].is_array()) {
            const auto& dist = calib_json["distortion_coefficients"];
            int dist_size = dist.size();
            _dist_coeffs = cv::Mat(1, dist_size, CV_64F);
            for (int i = 0; i < dist_size; ++i) {
                _dist_coeffs.at<double>(0, i) = dist[i].get<double>();
            }
        }

        if (!_camera_matrix.empty()) {
            _has_calibration = true;
            logger::info("[HeadPoseEstimation2D] Loaded calibrated camera matrix from '{}'", json_path);
            return true;
        }
    }
    catch (const std::exception& e) {
        logger::error("[HeadPoseEstimation2D] Failed to parse calibration JSON: {}", e.what());
    }

    _has_calibration = false;
    return false;
}

head_pose::PoseResult head_pose_estimation_from_2d::estimate(const std::vector<cv::Point2f>& landmarks_68, const cv::Size& image_size) {
    head_pose::PoseResult res;
    if (landmarks_68.size() < 68 || image_size.width <= 0 || image_size.height <= 0) {
        return res;
    }

    try {
        res.nose_tip_2d = landmarks_68[30]; // Nose tip

        std::vector<cv::Point2d> image_points = {
            cv::Point2d(landmarks_68[30].x, landmarks_68[30].y),
            cv::Point2d(landmarks_68[8].x, landmarks_68[8].y),
            cv::Point2d(landmarks_68[36].x, landmarks_68[36].y),
            cv::Point2d(landmarks_68[45].x, landmarks_68[45].y),
            cv::Point2d(landmarks_68[48].x, landmarks_68[48].y),
            cv::Point2d(landmarks_68[54].x, landmarks_68[54].y)
        };

        cv::Mat camera_mat;
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

        if (_has_calibration && !_camera_matrix.empty()) {
            camera_mat = _camera_matrix;
        } else {
            double focal_length = (double)image_size.width;
            cv::Point2d center((double)image_size.width / 2.0, (double)image_size.height / 2.0);
            camera_mat = (cv::Mat_<double>(3, 3) <<
                focal_length, 0.0, center.x,
                0.0, focal_length, center.y,
                0.0, 0.0, 1.0);
        }

        bool success = cv::solvePnP(_model_points, image_points, camera_mat, dist_coeffs, res.rvec, res.tvec, false, cv::SOLVEPNP_SQPNP);
        if (!success) {
            success = cv::solvePnP(_model_points, image_points, camera_mat, dist_coeffs, res.rvec, res.tvec, false, cv::SOLVEPNP_ITERATIVE);
        }

        if (success && !res.rvec.empty() && !res.tvec.empty()) {
            res.success = true;

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

            res.euler = cv::Vec3d(
                pitch * 180.0 / M_PI,
                yaw   * 180.0 / M_PI,
                roll  * 180.0 / M_PI
            );
        }
    }
    catch (const std::exception& e) {
        logger::error("[HeadPoseEstimation2D] Exception during pose estimation: {}", e.what());
    }

    return res;
}

void head_pose_estimation_from_2d::drawPoseAxes(cv::Mat& image, const head_pose::PoseResult& result, const cv::Size& orig_image_size, float scale_x, float scale_y) {
    if (!result.success || result.rvec.empty() || result.tvec.empty()) {
        return;
    }

    try {
        cv::Mat camera_mat;
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

        if (_has_calibration && !_camera_matrix.empty()) {
            camera_mat = _camera_matrix;
        } else {
            double focal_length = (double)orig_image_size.width;
            cv::Point2d center((double)orig_image_size.width / 2.0, (double)orig_image_size.height / 2.0);
            camera_mat = (cv::Mat_<double>(3, 3) <<
                focal_length, 0.0, center.x,
                0.0, focal_length, center.y,
                0.0, 0.0, 1.0);
        }

        double axis_length = 300.0;
        std::vector<cv::Point3d> axis_points_3d = {
            cv::Point3d(0.0, 0.0, 0.0),            // Origin (Nose tip)
            cv::Point3d(axis_length, 0.0, 0.0),    // X-axis (Red)
            cv::Point3d(0.0, axis_length, 0.0),    // Y-axis (Green)
            cv::Point3d(0.0, 0.0, axis_length)     // Z-axis (Blue)
        };

        std::vector<cv::Point2d> axis_points_2d;
        cv::projectPoints(axis_points_3d, result.rvec, result.tvec, camera_mat, dist_coeffs, axis_points_2d);

        if (axis_points_2d.size() >= 4) {
            // Anchor origin p0 EXACTLY at 2D nose tip
            cv::Point2f nose_tip = result.nose_tip_2d;
            cv::Point p0((int)std::round(nose_tip.x * scale_x), (int)std::round(nose_tip.y * scale_y));

            cv::Point2d origin_proj = axis_points_2d[0];
            cv::Point px((int)std::round((axis_points_2d[1].x - origin_proj.x + nose_tip.x) * scale_x),
                         (int)std::round((axis_points_2d[1].y - origin_proj.y + nose_tip.y) * scale_y));
            cv::Point py((int)std::round((axis_points_2d[2].x - origin_proj.x + nose_tip.x) * scale_x),
                         (int)std::round((axis_points_2d[2].y - origin_proj.y + nose_tip.y) * scale_y));
            cv::Point pz((int)std::round((axis_points_2d[3].x - origin_proj.x + nose_tip.x) * scale_x),
                         (int)std::round((axis_points_2d[3].y - origin_proj.y + nose_tip.y) * scale_y));

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
        logger::error("[HeadPoseEstimation2D] Exception during drawPoseAxes: {}", e.what());
    }
}
