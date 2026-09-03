#include "face_analysis_e2e.hpp"
#include <flame/log.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

face_analysis_e2e::face_analysis_e2e() {
}

face_analysis_e2e::~face_analysis_e2e() {
}

bool face_analysis_e2e::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    if (torch::cuda::is_available() && _gpu_id >= 0) {
        _device = torch::Device(torch::kCUDA, _gpu_id);
    } else {
        _device = torch::Device(torch::kCPU);
    }

    try {
        _module = torch::jit::load(model_path, _device);
        _module.eval();
        _is_loaded = true;
        logger::info("[face_analysis_e2e] Successfully loaded DAD-3DHeads E2E model from {} on device: {}",
                     model_path, _device.str());
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[face_analysis_e2e] Error loading model from {}: {}", model_path, e.what());
        _is_loaded = false;
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[face_analysis_e2e] Exception loading model from {}: {}", model_path, e.what());
        _is_loaded = false;
        return false;
    }
}

cv::Rect face_analysis_e2e::makeSquareBox(const cv::Rect& bbox, int img_w, int img_h) {
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    int max_side = std::max(bbox.width, bbox.height);

    int x1 = static_cast<int>(std::round(cx - max_side / 2.0f));
    int y1 = static_cast<int>(std::round(cy - max_side / 2.0f));

    return cv::Rect(x1, y1, max_side, max_side);
}

void face_analysis_e2e::computeHeadPose(
    const float* pred_3dmm_ptr,
    const cv::Point2f& nose_tip,
    head_pose::PoseResult& out_pose
) {
    // Rotation 6D is stored at indices 403 to 408 (size 6)
    // x_raw: [0, 1, 2], y_raw: [3, 4, 5]
    const float* r6d = pred_3dmm_ptr + 403;

    double x0 = r6d[0], x1 = r6d[1], x2 = r6d[2];
    double y0 = r6d[3], y1 = r6d[4], y2 = r6d[5];

    double norm_x = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    if (norm_x < 1e-8) norm_x = 1e-8;
    x0 /= norm_x; x1 /= norm_x; x2 /= norm_x;

    // z = cross(x, y_raw)
    double z0 = x1 * y2 - x2 * y1;
    double z1 = x2 * y0 - x0 * y2;
    double z2 = x0 * y1 - x1 * y0;

    double norm_z = std::sqrt(z0 * z0 + z1 * z1 + z2 * z2);
    if (norm_z < 1e-8) norm_z = 1e-8;
    z0 /= norm_z; z1 /= norm_z; z2 /= norm_z;

    // y = cross(z, x)
    double vy0 = z1 * x2 - z2 * x1;
    double vy1 = z2 * x0 - z0 * x2;
    double vy2 = z0 * x1 - z1 * x0;

    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        x0, vy0, z0,
        x1, vy1, z1,
        x2, vy2, z2
    );

    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    cv::Vec3d euler_angles;
    cv::Mat mtxR, mtxQ, qx, qy, qz;
    euler_angles = cv::RQDecomp3x3(R, mtxR, mtxQ, qx, qy, qz);

    out_pose.rvec = rvec;
    out_pose.euler = euler_angles; // pitch, yaw, roll in degrees
    out_pose.nose_tip_2d = nose_tip;

    // Translation at indices 409 to 411
    const float* trans = pred_3dmm_ptr + 409;
    out_pose.tvec = (cv::Mat_<double>(3, 1) << trans[0], trans[1], trans[2]);
    out_pose.success = true;
}

face_analysis::FaceAnalysisResult face_analysis_e2e::process(
    const cv::Mat& orig_image,
    const cv::Rect& face_bbox
) {
    face_analysis::FaceAnalysisResult result;
    if (!_is_loaded || orig_image.empty() || face_bbox.width <= 0 || face_bbox.height <= 0) {
        return result;
    }

    // 1. Compute 1:1 square bounding box centered around longest side
    cv::Rect sq_box = makeSquareBox(face_bbox, orig_image.cols, orig_image.rows);
    result.square_bbox = sq_box;

    // 2. Safely crop square face patch with padding if outside image boundaries
    int src_x1 = std::max(0, sq_box.x);
    int src_y1 = std::max(0, sq_box.y);
    int src_x2 = std::min(orig_image.cols, sq_box.x + sq_box.width);
    int src_y2 = std::min(orig_image.rows, sq_box.y + sq_box.height);

    if (src_x2 <= src_x1 || src_y2 <= src_y1) {
        return result;
    }

    cv::Mat cropped_valid = orig_image(cv::Rect(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1));
    cv::Mat square_patch = cv::Mat::zeros(sq_box.height, sq_box.width, orig_image.type());

    int dst_x1 = src_x1 - sq_box.x;
    int dst_y1 = src_y1 - sq_box.y;
    cropped_valid.copyTo(square_patch(cv::Rect(dst_x1, dst_y1, src_x2 - src_x1, src_y2 - src_y1)));

    // 3. Resize to model input size (256x256)
    cv::Mat resized_patch;
    cv::resize(square_patch, resized_patch, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), 0, 0, cv::INTER_LINEAR);

    // 4. Convert BGR to RGB and normalize (ImageNet Mean/Std)
    cv::Mat rgb_patch;
    cv::cvtColor(resized_patch, rgb_patch, cv::COLOR_BGR2RGB);

    cv::Mat float_patch;
    rgb_patch.convertTo(float_patch, CV_32FC3, 1.0 / 255.0);
    cv::subtract(float_patch, cv::Scalar(0.485f, 0.456f, 0.406f), float_patch);
    cv::divide(float_patch, cv::Scalar(0.229f, 0.224f, 0.225f), float_patch);

    // 5. Construct Tensor: [1, 3, 256, 256]
    torch::Tensor input_tensor = torch::from_blob(float_patch.data, {1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3}, torch::kFloat32);
    input_tensor = input_tensor.permute({0, 3, 1, 2}).clone().to(_device);

    // 6. Run Inference
    try {
        auto output_tuple = _module.forward({input_tensor}).toTuple();
        auto elements = output_tuple->elements();

        // elements[0]: 3dmm_params [1, 413]
        // elements[1]: landmarks_191 [1, 191, 2]
        // elements[2]: landmarks_68 [1, 68, 2]
        // elements[3]: vertices_3d [1, 5023, 3]
        // elements[4]: projected_vertices_2d [1, 5023, 2]
        torch::Tensor t_3dmm = elements[0].toTensor().to(torch::kCPU);
        torch::Tensor t_lm191 = elements[1].toTensor().to(torch::kCPU);
        torch::Tensor t_lm68 = elements[2].toTensor().to(torch::kCPU);
        torch::Tensor t_v3d = elements[3].toTensor().to(torch::kCPU);
        torch::Tensor t_v2d = elements[4].toTensor().to(torch::kCPU);

        float scale_ratio = static_cast<float>(sq_box.width) / static_cast<float>(MODEL_INPUT_SIZE);

        // 7. Parse 68 Landmarks
        int num_lm68 = t_lm68.size(1);
        auto lm68_acc = t_lm68.accessor<float, 3>();
        result.landmarks_68.reserve(num_lm68);
        for (int i = 0; i < num_lm68; ++i) {
            float px = lm68_acc[0][i][0] * scale_ratio + sq_box.x;
            float py = lm68_acc[0][i][1] * scale_ratio + sq_box.y;
            result.landmarks_68.emplace_back(px, py);
        }

        // 8. Parse 191 Head Landmarks
        int num_lm191 = t_lm191.size(1);
        auto lm191_acc = t_lm191.accessor<float, 3>();
        result.landmarks_191.reserve(num_lm191);
        for (int i = 0; i < num_lm191; ++i) {
            float px = lm191_acc[0][i][0] * scale_ratio + sq_box.x;
            float py = lm191_acc[0][i][1] * scale_ratio + sq_box.y;
            result.landmarks_191.emplace_back(px, py);
        }

        // 9. Parse 3DMM Parameters
        int num_3dmm = t_3dmm.size(1);
        const float* p_3dmm = t_3dmm.data_ptr<float>();
        result.params_3dmm.assign(p_3dmm, p_3dmm + num_3dmm);

        // 10. Compute 3D Head Pose
        cv::Point2f nose_anchor = (result.landmarks_68.size() > 30) ? result.landmarks_68[30] : cv::Point2f(sq_box.x + sq_box.width / 2.0f, sq_box.y + sq_box.height / 2.0f);
        computeHeadPose(p_3dmm, nose_anchor, result.pose);

        // 11. Parse 3D Mesh Vertices & 2D Projected Vertices (5023 vertices)
        int num_v = t_v3d.size(1);
        auto v3d_acc = t_v3d.accessor<float, 3>();
        auto v2d_acc = t_v2d.accessor<float, 3>();
        result.vertices_3d.reserve(num_v);
        result.projected_vertices.reserve(num_v);

        for (int i = 0; i < num_v; ++i) {
            result.vertices_3d.emplace_back(v3d_acc[0][i][0], v3d_acc[0][i][1], v3d_acc[0][i][2]);
            float px = v2d_acc[0][i][0] * scale_ratio + sq_box.x;
            float py = v2d_acc[0][i][1] * scale_ratio + sq_box.y;
            result.projected_vertices.emplace_back(px, py);
        }

        result.valid = true;
    }
    catch (const std::exception& e) {
        logger::error("[face_analysis_e2e] Inference error: {}", e.what());
        result.valid = false;
    }

    return result;
}

void face_analysis_e2e::drawResult(
    cv::Mat& image,
    const face_analysis::FaceAnalysisResult& result,
    bool draw_68,
    bool draw_191,
    bool draw_pose,
    bool draw_box,
    bool draw_mesh
) {
    if (!result.valid || image.empty()) {
        return;
    }

    // 1. Draw 1:1 Square Bounding Box
    if (draw_box) {
        cv::rectangle(image, result.square_bbox, cv::Scalar(0, 255, 255), 2);
    }

    // 2. Draw 68 Facial Landmarks
    if (draw_68) {
        for (size_t i = 0; i < result.landmarks_68.size(); ++i) {
            cv::circle(image, result.landmarks_68[i], 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }
    }

    // 3. Draw 191 Head Landmarks
    if (draw_191) {
        for (size_t i = 0; i < result.landmarks_191.size(); ++i) {
            cv::circle(image, result.landmarks_191[i], 1, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
        }
    }

    // 4. Draw 3D Head Mesh (Projected Vertices)
    if (draw_mesh) {
        for (size_t i = 0; i < result.projected_vertices.size(); i += 5) {
            cv::circle(image, result.projected_vertices[i], 1, cv::Scalar(200, 200, 200), -1, cv::LINE_AA);
        }
    }

    // 5. Draw 3D Pose Axes (Pitch, Yaw, Roll)
    if (draw_pose && result.pose.success) {
        double pitch = result.pose.euler[0];
        double yaw = result.pose.euler[1];
        double roll = result.pose.euler[2];

        // Draw pose angle text
        std::string pose_text = cv::format("P:%.1f Y:%.1f R:%.1f", pitch, yaw, roll);
        cv::putText(image, pose_text, cv::Point(result.square_bbox.x, std::max(20, result.square_bbox.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        // Render 3D coordinate axes from nose tip
        cv::Point2f origin = result.pose.nose_tip_2d;
        float axis_len = static_cast<float>(result.square_bbox.width) * 0.4f;

        // Approximate 2D projection of axes based on pitch, yaw, roll
        double pitch_rad = pitch * CV_PI / 180.0;
        double yaw_rad = yaw * CV_PI / 180.0;
        double roll_rad = roll * CV_PI / 180.0;

        // X-axis (Red: Right)
        cv::Point2f x_end(
            origin.x + axis_len * static_cast<float>(std::cos(yaw_rad) * std::cos(roll_rad)),
            origin.y + axis_len * static_cast<float>(std::cos(yaw_rad) * std::sin(roll_rad))
        );
        // Y-axis (Green: Down)
        cv::Point2f y_end(
            origin.x - axis_len * static_cast<float>(std::sin(roll_rad) * std::cos(pitch_rad)),
            origin.y + axis_len * static_cast<float>(std::cos(roll_rad) * std::cos(pitch_rad))
        );
        // Z-axis (Blue: Out / Forward)
        cv::Point2f z_end(
            origin.x + axis_len * static_cast<float>(std::sin(yaw_rad)),
            origin.y - axis_len * static_cast<float>(std::sin(pitch_rad))
        );

        cv::line(image, origin, x_end, cv::Scalar(0, 0, 255), 2, cv::LINE_AA); // X: Red
        cv::line(image, origin, y_end, cv::Scalar(0, 255, 0), 2, cv::LINE_AA); // Y: Green
        cv::line(image, origin, z_end, cv::Scalar(255, 0, 0), 2, cv::LINE_AA); // Z: Blue
    }
}
