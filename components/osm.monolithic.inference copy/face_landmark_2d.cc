#include "face_landmark_2d.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

face_landmark_2d::face_landmark_2d() {}
face_landmark_2d::~face_landmark_2d() {}

bool face_landmark_2d::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    try {
        std::string path = model_path;

        if (!fs::exists(path)) {
            logger::error("[FaceLandmark2D] Model file not found: {}", path);
            return false;
        }

        // Set device (GPU or CPU)
        if (torch::cuda::is_available() && gpu_id >= 0) {
            _device = torch::Device(torch::kCUDA, gpu_id);
            logger::info("[FaceLandmark2D] CUDA is available. Using GPU: {}", gpu_id);
        } else {
            _device = torch::Device(torch::kCPU);
            logger::warn("[FaceLandmark2D] CUDA is not available. Using CPU");
        }

        // Load the TorchScript module
        _module = torch::jit::load(path);
        _module.to(_device);
        _module.eval();

        logger::info("[FaceLandmark2D] Loaded TorchScript model successfully from {}", path);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[FaceLandmark2D] Failed to load TorchScript model: {}", e.what());
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[FaceLandmark2D] Exception during model load: {}", e.what());
        return false;
    }
}

std::vector<face_landmark::LandmarkResult> face_landmark_2d::process(const cv::Mat& image, const std::vector<cv::Rect>& bboxes) {
    std::vector<face_landmark::LandmarkResult> all_landmarks;
    if (image.empty() || bboxes.empty()) {
        return all_landmarks;
    }

    try {
        torch::NoGradGuard no_grad;

        for (const auto& box : bboxes) {
            float w_box = (float)box.width;
            float h_box = (float)box.height;
            if (w_box <= 0 || h_box <= 0) continue;

            // Unpad face_detection bbox (face_detection adds padding_w=0.3, padding_h=0.2)
            float raw_w = w_box / 1.3f;
            float raw_h = h_box / 1.2f;

            // FAN Affine Center & Scale calculation
            // Center is shifted up by 12% of raw height (CENTER_Y_OFFSET = 0.12)
            float center_x = box.x + w_box / 2.0f;
            float center_y = box.y + h_box / 2.0f - raw_h * 0.12f;

            // reference_scale = 195.0f, SCALE_FACTOR = 200.0f
            float scale = (raw_w + raw_h) / 195.0f;
            float crop_size = 200.0f * scale;

            // Exact FAN crop bounding box in original image coordinates
            int ul_x = (int)std::round(center_x - crop_size / 2.0f);
            int ul_y = (int)std::round(center_y - crop_size / 2.0f);
            int br_x = (int)std::round(center_x + crop_size / 2.0f);
            int br_y = (int)std::round(center_y + crop_size / 2.0f);

            int patch_w = br_x - ul_x;
            int patch_h = br_y - ul_y;
            if (patch_w <= 0 || patch_h <= 0) continue;

            // Extract crop patch with zero padding for out-of-boundary regions (matches FAN utils.crop)
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

            // Convert to float32 and scale to [0, 1]
            cv::Mat float_image;
            resized.convertTo(float_image, CV_32FC3, 1.0f / 255.0f);

            // Create Torch Tensor (make contiguous to avoid memory stride issues)
            auto input_tensor = torch::from_blob(float_image.data, {1, _input_height, _input_width, 3}, torch::kFloat32);
            input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous().to(_device);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // Run model inference
            auto outputs = _module.forward(inputs);

            torch::Tensor output_tensor;
            if (outputs.isTensor()) {
                output_tensor = outputs.toTensor();
            } else if (outputs.isTuple()) {
                auto elements = outputs.toTuple()->elements();
                output_tensor = elements.back().toTensor();
            } else if (outputs.isList()) {
                auto elements = outputs.toList();
                output_tensor = elements.get(elements.size() - 1).toTensor();
            } else {
                logger::error("[FaceLandmark2D] Unexpected model output format");
                continue;
            }

            // Move heatmap tensor to CPU and ensure contiguous memory layout: shape [68, 64, 64]
            output_tensor = output_tensor.to(torch::kCPU).contiguous().squeeze(0);
            int num_landmarks = output_tensor.size(0); // 68
            int hm_height = output_tensor.size(1);    // 64
            int hm_width = output_tensor.size(2);     // 64

            float* data = output_tensor.data_ptr<float>();

            std::vector<cv::Point2f> landmarks;
            landmarks.reserve(num_landmarks);

            for (int i = 0; i < num_landmarks; ++i) {
                float max_val = -1e9f;
                int pX = 0, pY = 0;
                int channel_offset = i * hm_height * hm_width;

                for (int r = 0; r < hm_height; ++r) {
                    for (int c = 0; c < hm_width; ++c) {
                        float val = data[channel_offset + r * hm_width + c];
                        if (val > max_val) {
                            max_val = val;
                            pY = r;
                            pX = c;
                        }
                    }
                }

                // Subpixel refinement according to FAN (_get_preds_fromhm)
                float diff_x = 0.0f;
                float diff_y = 0.0f;
                if (pX > 0 && pX < hm_width - 1 && pY > 0 && pY < hm_height - 1) {
                    diff_x = data[channel_offset + pY * hm_width + (pX + 1)] - data[channel_offset + pY * hm_width + (pX - 1)];
                    diff_y = data[channel_offset + (pY + 1) * hm_width + pX] - data[channel_offset + (pY - 1) * hm_width + pX];
                }

                float px_pred = (pX + 1.0f) + (diff_x > 0.0f ? 0.25f : (diff_x < 0.0f ? -0.25f : 0.0f)) - 0.5f;
                float py_pred = (pY + 1.0f) + (diff_y > 0.0f ? 0.25f : (diff_y < 0.0f ? -0.25f : 0.0f)) - 0.5f;

                // Transform (px_pred, py_pred) from 64x64 heatmap back to original image space
                float x_orig = (px_pred / (float)hm_width - 0.5f) * (float)patch_w + center_x;
                float y_orig = (py_pred / (float)hm_height - 0.5f) * (float)patch_h + center_y;

                landmarks.push_back(cv::Point2f(x_orig, y_orig));
            }

            face_landmark::LandmarkResult res;
            res.points = landmarks;
            res.crop_bbox = cv::Rect(ul_x, ul_y, patch_w, patch_h);
            all_landmarks.push_back(res);
        }
    }
    catch (const c10::Error& e) {
        logger::error("[FaceLandmark2D] LibTorch error during process: {}", e.what());
    }
    catch (const std::exception& e) {
        logger::error("[FaceLandmark2D] Exception during process: {}", e.what());
    }

    return all_landmarks;
}
