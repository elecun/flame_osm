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

void face_analysis_e2e::calculateRPYFrom6D(
    const float* r6d,
    double& out_roll,
    double& out_pitch,
    double& out_yaw,
    cv::Mat& out_rvec
) {
    // Continuous 6D rotation vector: vx = r6d[0..2], vy = r6d[3..5]
    double vx0 = r6d[0], vx1 = r6d[1], vx2 = r6d[2];
    double vy0 = r6d[3], vy1 = r6d[4], vy2 = r6d[5];

    // b1 = normalize(vx)
    double norm_vx = std::sqrt(vx0 * vx0 + vx1 * vx1 + vx2 * vx2);
    if (norm_vx < 1e-8) norm_vx = 1e-8;
    double b1_0 = vx0 / norm_vx;
    double b1_1 = vx1 / norm_vx;
    double b1_2 = vx2 / norm_vx;

    // c = cross(b1, vy)
    double c0 = b1_1 * vy2 - b1_2 * vy1;
    double c1 = b1_2 * vy0 - b1_0 * vy2;
    double c2 = b1_0 * vy1 - b1_1 * vy0;

    // b3 = normalize(c)
    double norm_c = std::sqrt(c0 * c0 + c1 * c1 + c2 * c2);
    if (norm_c < 1e-8) norm_c = 1e-8;
    double b3_0 = c0 / norm_c;
    double b3_1 = c1 / norm_c;
    double b3_2 = c2 / norm_c;

    // b2 = -cross(b1, b3) = cross(b3, b1)
    double b2_0 = b3_1 * b1_2 - b3_2 * b1_1;
    double b2_1 = b3_2 * b1_0 - b3_0 * b1_2;
    double b2_2 = b3_0 * b1_1 - b3_1 * b1_0;

    // In demo_e2e_test.py:
    // rot_mat has columns [b1, b2, b3]
    // rot_mat_2 = np.transpose(rot_mat)
    // Rows of rot_mat_2 are b1, b2, b3.
    // Scipy as_euler('xyz', degrees=True) on rot_mat_2:
    double sin_y = std::clamp(-b3_0, -1.0, 1.0);
    double y = std::asin(sin_y);
    double x = 0.0;
    double z = 0.0;

    if (std::abs(b3_0) < 0.9999999) {
        x = std::atan2(b3_1, b3_2);
        z = std::atan2(b2_0, b1_0);
    } else {
        x = std::atan2(-b1_1, b2_1);
        z = 0.0;
    }

    constexpr double RAD2DEG = 180.0 / M_PI;
    double angle_x = x * RAD2DEG;
    double angle_y = y * RAD2DEG;
    double angle_z = z * RAD2DEG;

    auto limit_angle = [](double ang) {
        while (ang < -180.0) ang += 360.0;
        while (ang > 180.0) ang -= 360.0;
        return ang;
    };

    out_roll = limit_angle(angle_z);
    out_pitch = limit_angle(angle_x - 180.0);
    out_yaw = limit_angle(angle_y);

    // Rotation matrix R = [b1, b2, b3]
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        b1_0, b2_0, b3_0,
        b1_1, b2_1, b3_1,
        b1_2, b2_2, b3_2
    );
    cv::Rodrigues(R, out_rvec);
}

void face_analysis_e2e::computeHeadPose(
    const float* pred_3dmm_ptr,
    const cv::Point2f& nose_tip,
    head_pose::PoseResult& out_pose
) {
    const float* r6d = pred_3dmm_ptr + 403;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    calculateRPYFrom6D(r6d, roll, pitch, yaw, out_pose.rvec);

    out_pose.euler = cv::Vec3d(pitch, yaw, roll); // (pitch, yaw, roll) in degrees
    out_pose.nose_tip_2d = nose_tip;

    const float* trans = pred_3dmm_ptr + 409;
    out_pose.tvec = (cv::Mat_<double>(3, 1) << trans[0], trans[1], trans[2]);
    out_pose.success = true;
}

face_analysis::FaceAnalysisResult face_analysis_e2e::process(
    const cv::Mat& orig_image,
    const cv::Rect& face_bbox,
    float score
) {
    face_analysis::FaceAnalysisResult result;
    if (!_is_loaded || orig_image.empty() || face_bbox.width <= 0 || face_bbox.height <= 0) {
        return result;
    }

    // 1. Clamped square crop matching demo_e2e_test.py
    int bx = std::max(0, std::min(face_bbox.x, orig_image.cols - 1));
    int by = std::max(0, std::min(face_bbox.y, orig_image.rows - 1));
    int bw = std::max(1, std::min(face_bbox.width, orig_image.cols - bx));
    int bh = std::max(1, std::min(face_bbox.height, orig_image.rows - by));

    cv::Rect crop_rect(bx, by, bw, bh);
    cv::Mat crop = orig_image(crop_rect);
    if (crop.empty()) {
        return result;
    }

    result.square_bbox = crop_rect;
    result.score = score;
    result.center = cv::Point2f(bx + bw / 2.0f, by + bh / 2.0f);
    result.scale_size = static_cast<float>(std::min(bw, bh));

    // 2. Resize to model input size (256x256)
    cv::Mat resized_crop;
    cv::resize(crop, resized_crop, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), 0, 0, cv::INTER_LINEAR);

    // 3. Convert BGR to RGB and normalize with ImageNet Mean/Std
    cv::Mat rgb_crop;
    cv::cvtColor(resized_crop, rgb_crop, cv::COLOR_BGR2RGB);

    cv::Mat float_patch;
    rgb_crop.convertTo(float_patch, CV_32FC3, 1.0f / 255.0f);
    cv::subtract(float_patch, cv::Scalar(0.485f, 0.456f, 0.406f), float_patch);
    cv::divide(float_patch, cv::Scalar(0.229f, 0.224f, 0.225f), float_patch);

    // 4. Construct Tensor: [1, 3, 256, 256]
    torch::Tensor input_tensor = torch::from_blob(float_patch.data, {1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3}, torch::kFloat32);
    input_tensor = input_tensor.permute({0, 3, 1, 2}).to(_device);

    // 5. Run Inference
    try {
        torch::NoGradGuard no_grad;
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

        float scale_w = static_cast<float>(bw) / static_cast<float>(MODEL_INPUT_SIZE);
        float scale_h = static_cast<float>(bh) / static_cast<float>(MODEL_INPUT_SIZE);

        // 6. Parse 68 Landmarks
        int num_lm68 = t_lm68.size(1);
        auto lm68_acc = t_lm68.accessor<float, 3>();
        result.landmarks_68.reserve(num_lm68);
        for (int i = 0; i < num_lm68; ++i) {
            float px = lm68_acc[0][i][0] * scale_w + bx;
            float py = lm68_acc[0][i][1] * scale_h + by;
            result.landmarks_68.emplace_back(px, py);
        }

        // 7. Parse 191 Head Landmarks
        int num_lm191 = t_lm191.size(1);
        auto lm191_acc = t_lm191.accessor<float, 3>();
        result.landmarks_191.reserve(num_lm191);
        for (int i = 0; i < num_lm191; ++i) {
            float px = lm191_acc[0][i][0] * scale_w + bx;
            float py = lm191_acc[0][i][1] * scale_h + by;
            result.landmarks_191.emplace_back(px, py);
        }

        // 8. Parse 3DMM Parameters
        int num_3dmm = t_3dmm.size(1);
        const float* p_3dmm = t_3dmm.data_ptr<float>();
        result.params_3dmm.assign(p_3dmm, p_3dmm + num_3dmm);

        // 9. Compute 3D Head Pose
        cv::Point2f nose_anchor = (result.landmarks_68.size() > 30) ? result.landmarks_68[30] : result.center;
        computeHeadPose(p_3dmm, nose_anchor, result.pose);

        // 10. Parse 3D Mesh Vertices & 2D Projected Vertices (5023 vertices)
        int num_v = t_v3d.size(1);
        auto v3d_acc = t_v3d.accessor<float, 3>();
        auto v2d_acc = t_v2d.accessor<float, 3>();
        result.vertices_3d.reserve(num_v);
        result.projected_vertices.reserve(num_v);

        for (int i = 0; i < num_v; ++i) {
            result.vertices_3d.emplace_back(v3d_acc[0][i][0], v3d_acc[0][i][1], v3d_acc[0][i][2]);
            float px = v2d_acc[0][i][0] * scale_w + bx;
            float py = v2d_acc[0][i][1] * scale_h + by;
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

    // 1. Draw Bounding Box (1:1 Head Region) matching demo_e2e_test.py
    if (draw_box) {
        cv::rectangle(image, result.square_bbox, cv::Scalar(0, 255, 128), 1);
    }

    // 2. Draw 191 Landmarks (Yellow points) matching demo_e2e_test.py
    if (draw_191 && !result.landmarks_191.empty()) {
        int pt_radius = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.003f));
        for (const auto& pt : result.landmarks_191) {
            cv::circle(image, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                       pt_radius, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
        }
    }

    // Optional: Draw 68 Facial Landmarks
    if (draw_68 && !result.landmarks_68.empty()) {
        for (const auto& pt : result.landmarks_68) {
            cv::circle(image, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                       2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }
    }

    // Optional: Draw 3D Head Mesh (Projected Vertices)
    if (draw_mesh && !result.projected_vertices.empty()) {
        for (size_t i = 0; i < result.projected_vertices.size(); i += 5) {
            cv::circle(image, cv::Point(static_cast<int>(result.projected_vertices[i].x),
                                        static_cast<int>(result.projected_vertices[i].y)),
                       1, cv::Scalar(200, 200, 200), -1, cv::LINE_AA);
        }
    }

    // 3. Draw 3D Head Pose Axis matching _draw_pose_axis in demo_e2e_test.py
    if (draw_pose && result.pose.success) {
        float tdx = result.center.x;
        float tdy = result.center.y;
        int size = std::max(10, static_cast<int>(result.scale_size * 0.35f));

        double roll = result.pose.euler[2] * CV_PI / 180.0;
        double pitch = result.pose.euler[0] * CV_PI / 180.0;
        double yaw = -(result.pose.euler[1] * CV_PI / 180.0);

        // X-Axis (Pitch / Red)
        double x1 = size * (std::cos(yaw) * std::cos(roll)) + tdx;
        double y1 = size * (std::cos(pitch) * std::sin(roll) + std::cos(roll) * std::sin(pitch) * std::sin(yaw)) + tdy;

        // Y-Axis (Yaw / Green)
        double x2 = size * (-std::cos(yaw) * std::sin(roll)) + tdx;
        double y2 = size * (std::cos(pitch) * std::cos(roll) - std::sin(pitch) * std::sin(yaw) * std::sin(roll)) + tdy;

        // Z-Axis (Roll / Blue)
        double x3 = size * (std::sin(yaw)) + tdx;
        double y3 = size * (-std::cos(yaw) * std::sin(pitch)) + tdy;

        int thickness = std::max(2, static_cast<int>(size * 0.05f));
        cv::arrowedLine(image, cv::Point(static_cast<int>(tdx), static_cast<int>(tdy)),
                        cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                        cv::Scalar(0, 0, 255), thickness, cv::LINE_AA, 0.2);
        cv::arrowedLine(image, cv::Point(static_cast<int>(tdx), static_cast<int>(tdy)),
                        cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                        cv::Scalar(0, 255, 0), thickness, cv::LINE_AA, 0.2);
        cv::arrowedLine(image, cv::Point(static_cast<int>(tdx), static_cast<int>(tdy)),
                        cv::Point(static_cast<int>(x3), static_cast<int>(y3)),
                        cv::Scalar(255, 0, 0), thickness, cv::LINE_AA, 0.2);
    }
}

void face_analysis_e2e::drawInfoPanel(
    cv::Mat& image,
    const head_pose::PoseResult& pose,
    int num_faces
) {
    int panel_w = 260, panel_h = 115;
    if (image.cols < panel_w + 20 || image.rows < panel_h + 20) return;

    cv::Rect panel_rect(10, 10, panel_w, panel_h);
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, panel_rect, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.65, image, 0.35, 0, image);
    cv::rectangle(image, panel_rect, cv::Scalar(255, 255, 255), 1);

    int font = cv::FONT_HERSHEY_SIMPLEX;
    char title[64], txt_pitch[64], txt_yaw[64], txt_roll[64];
    snprintf(title, sizeof(title), "E2E TorchScript (%d face%s)", num_faces, num_faces > 1 ? "s" : "");
    snprintf(txt_pitch, sizeof(txt_pitch), "Pitch: %+6.1f deg", pose.euler[0]);
    snprintf(txt_yaw,   sizeof(txt_yaw),   "Yaw:   %+6.1f deg", pose.euler[1]);
    snprintf(txt_roll,  sizeof(txt_roll),  "Roll:  %+6.1f deg", pose.euler[2]);

    cv::putText(image, title,     cv::Point(18, 30),  font, 0.45, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    cv::putText(image, txt_pitch, cv::Point(18, 55),  font, 0.55, cv::Scalar(0, 0, 255),     1, cv::LINE_AA);
    cv::putText(image, txt_yaw,   cv::Point(18, 80),  font, 0.55, cv::Scalar(0, 255, 0),     1, cv::LINE_AA);
    cv::putText(image, txt_roll,  cv::Point(18, 105), font, 0.55, cv::Scalar(255, 150, 50),   1, cv::LINE_AA);
}
