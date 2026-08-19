#include "face_landmark_3d.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

face_landmark_3d::face_landmark_3d() {}
face_landmark_3d::~face_landmark_3d() {}

bool face_landmark_3d::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    try {
        std::string path = model_path;

        if (!fs::exists(path)) {
            logger::error("[FaceLandmark3D] Model file not found: {}", path);
            return false;
        }

        // Set device (GPU or CPU)
        if (torch::cuda::is_available() && gpu_id >= 0) {
            _device = torch::Device(torch::kCUDA, gpu_id);
            logger::info("[FaceLandmark3D] CUDA is available. Using GPU: {}", gpu_id);
        } else {
            _device = torch::Device(torch::kCPU);
            logger::warn("[FaceLandmark3D] CUDA is not available. Using CPU");
        }

        // Load the TorchScript module
        _module = torch::jit::load(path);
        _module.to(_device);
        _module.eval();

        logger::info("[FaceLandmark3D] Loaded TorchScript model successfully from {}", path);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[FaceLandmark3D] Failed to load TorchScript model: {}", e.what());
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[FaceLandmark3D] Exception during model load: {}", e.what());
        return false;
    }
}

std::vector<face_landmark_3d_ns::Landmark3DResult> face_landmark_3d::process(const cv::Mat& image, const std::vector<cv::Rect>& bboxes) {
    std::vector<face_landmark_3d_ns::Landmark3DResult> all_landmarks;
    if (image.empty() || bboxes.empty()) {
        return all_landmarks;
    }

    try {
        torch::NoGradGuard no_grad;

        for (const auto& box : bboxes) {
            float w_box = (float)box.width;
            float h_box = (float)box.height;
            if (w_box <= 0 || h_box <= 0) continue;

            // Unpad face_detection bbox
            float raw_w = w_box / 1.3f;
            float raw_h = h_box / 1.2f;

            // FAN Affine Center & Scale calculation
            float center_x = box.x + w_box / 2.0f;
            float center_y = box.y + h_box / 2.0f - raw_h * 0.12f;

            float scale = (raw_w + raw_h) / 195.0f;
            float crop_size = 200.0f * scale;

            int ul_x = (int)std::round(center_x - crop_size / 2.0f);
            int ul_y = (int)std::round(center_y - crop_size / 2.0f);
            int br_x = (int)std::round(center_x + crop_size / 2.0f);
            int br_y = (int)std::round(center_y + crop_size / 2.0f);

            int patch_w = br_x - ul_x;
            int patch_h = br_y - ul_y;
            if (patch_w <= 0 || patch_h <= 0) continue;

            // Extract crop patch
            cv::Mat cropped_patch = cv::Mat::zeros(patch_h, patch_w, image.type());

            int src_x1 = std::max(0, ul_x);
            int src_y1 = std::max(0, ul_y);
            int src_x2 = std::min(image.cols, br_x);
            int src_y2 = std::min(image.rows, br_y);

            int dst_x1 = src_x1 - ul_x;
            int dst_y1 = src_y1 - ul_y;
            int dst_x2 = dst_x1 + (src_x2 - src_x1);
            int dst_y2 = dst_y1 + (src_y2 - src_y1);

            if (src_x2 > src_x1 && src_y2 > src_y1) {
                image(cv::Rect(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1))
                    .copyTo(cropped_patch(cv::Rect(dst_x1, dst_y1, dst_x2 - dst_x1, dst_y2 - dst_y1)));
            }

            // Preprocess: Resize patch to 256x256 and BGR to RGB conversion
            cv::Mat resized;
            cv::resize(cropped_patch, resized, cv::Size(_input_width, _input_height));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

            cv::Mat float_image;
            resized.convertTo(float_image, CV_32FC3, 1.0f / 255.0f);

            auto input_tensor = torch::from_blob(float_image.data, {1, _input_height, _input_width, 3}, torch::kFloat32);
            input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous().to(_device);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // Run model inference
            auto outputs = _module.forward(inputs);

            torch::Tensor heatmap_tensor;
            torch::Tensor depth_tensor;

            if (outputs.isTuple()) {
                auto elements = outputs.toTuple()->elements();
                if (elements.size() >= 2) {
                    heatmap_tensor = elements[0].toTensor();
                    depth_tensor = elements[1].toTensor();
                } else if (elements.size() == 1) {
                    heatmap_tensor = elements[0].toTensor();
                }
            } else if (outputs.isList()) {
                auto elements = outputs.toList();
                if (elements.size() >= 2) {
                    heatmap_tensor = elements.get(0).toTensor();
                    depth_tensor = elements.get(1).toTensor();
                } else if (elements.size() == 1) {
                    heatmap_tensor = elements.get(0).toTensor();
                }
            } else if (outputs.isTensor()) {
                heatmap_tensor = outputs.toTensor();
            } else {
                logger::error("[FaceLandmark3D] Unexpected model output format");
                continue;
            }

            if (!heatmap_tensor.defined()) {
                logger::error("[FaceLandmark3D] Heatmap tensor is undefined");
                continue;
            }

            // Move heatmap tensor to CPU: shape [68, 64, 64]
            heatmap_tensor = heatmap_tensor.to(torch::kCPU).contiguous().squeeze(0);
            if (depth_tensor.defined()) {
                depth_tensor = depth_tensor.to(torch::kCPU).contiguous().squeeze(0); // [68]
            }

            int num_landmarks = heatmap_tensor.size(0); // 68
            int hm_height = heatmap_tensor.size(1);    // 64
            int hm_width = heatmap_tensor.size(2);     // 64

            float* hm_data = heatmap_tensor.data_ptr<float>();
            float* depth_data = (depth_tensor.defined() && depth_tensor.numel() >= num_landmarks) ? depth_tensor.data_ptr<float>() : nullptr;

            std::vector<cv::Point3f> landmarks_3d;
            std::vector<cv::Point2f> landmarks_2d;
            landmarks_3d.reserve(num_landmarks);
            landmarks_2d.reserve(num_landmarks);

            for (int i = 0; i < num_landmarks; ++i) {
                float max_val = -1e9f;
                int pX = 0, pY = 0;
                int channel_offset = i * hm_height * hm_width;

                for (int r = 0; r < hm_height; ++r) {
                    for (int c = 0; c < hm_width; ++c) {
                        float val = hm_data[channel_offset + r * hm_width + c];
                        if (val > max_val) {
                            max_val = val;
                            pY = r;
                            pX = c;
                        }
                    }
                }

                float diff_x = 0.0f;
                float diff_y = 0.0f;
                if (pX > 0 && pX < hm_width - 1 && pY > 0 && pY < hm_height - 1) {
                    diff_x = hm_data[channel_offset + pY * hm_width + (pX + 1)] - hm_data[channel_offset + pY * hm_width + (pX - 1)];
                    diff_y = hm_data[channel_offset + (pY + 1) * hm_width + pX] - hm_data[channel_offset + (pY - 1) * hm_width + pX];
                }

                float px_pred = (pX + 1.0f) + (diff_x > 0.0f ? 0.25f : (diff_x < 0.0f ? -0.25f : 0.0f)) - 0.5f;
                float py_pred = (pY + 1.0f) + (diff_y > 0.0f ? 0.25f : (diff_y < 0.0f ? -0.25f : 0.0f)) - 0.5f;

                float x_orig = (px_pred / (float)hm_width - 0.5f) * (float)patch_w + center_x;
                float y_orig = (py_pred / (float)hm_height - 0.5f) * (float)patch_h + center_y;
                float z_orig = depth_data ? depth_data[i] : max_val; // Use predicted 3D depth Z if available

                landmarks_2d.push_back(cv::Point2f(x_orig, y_orig));
                landmarks_3d.push_back(cv::Point3f(x_orig, y_orig, z_orig));
            }

            face_landmark_3d_ns::Landmark3DResult res;
            res.points_2d = landmarks_2d;
            res.points_3d = landmarks_3d;
            res.crop_bbox = cv::Rect(ul_x, ul_y, patch_w, patch_h);
            all_landmarks.push_back(res);
        }
    }
    catch (const c10::Error& e) {
        logger::error("[FaceLandmark3D] LibTorch error during process: {}", e.what());
    }
    catch (const std::exception& e) {
        logger::error("[FaceLandmark3D] Exception during process: {}", e.what());
    }

    return all_landmarks;
}
