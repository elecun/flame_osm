#include "face_detection.hpp"
#include <flame/log.hpp>
#include <filesystem>
#include <algorithm>
namespace fs = std::filesystem;

face_detection::face_detection() {}
face_detection::~face_detection() {}

bool face_detection::loadModel(const std::string& model_path, int gpu_id) {
    _gpu_id = gpu_id;
    try {
        std::string path = model_path;

        if (!fs::exists(path)) {
            logger::error("[FaceDetection] Model file not found: {}", path);
            return false;
        }

        // Set device (GPU or CPU)
        if (torch::cuda::is_available() && gpu_id >= 0) {
            _device = torch::Device(torch::kCUDA, gpu_id);
            logger::info("[FaceDetection] CUDA is available. Using GPU: {}", gpu_id);
        } else {
            _device = torch::Device(torch::kCPU);
            logger::warn("[FaceDetection] CUDA is not available. Using CPU");
        }

        // Load the TorchScript module
        _module = torch::jit::load(path);
        _module.to(_device);
        _module.eval(); // set to evaluation mode

        logger::info("[FaceDetection] Loaded TorchScript model successfully from {}", path);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[FaceDetection] Failed to load TorchScript model: {}", e.what());
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during model load: {}", e.what());
        return false;
    }
}

std::vector<cv::Rect> face_detection::process(const cv::Mat& image, float nms_threshold, float padding_w, float padding_h) {
    std::vector<cv::Rect> bboxes;
    if (image.empty()) {
        logger::warn("[FaceDetection] Input image is empty");
        return bboxes;
    }

    try {
        // 1. Preprocess using OpenCV: Resize and BGR to RGB conversion
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(_input_width, _input_height));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // Convert data type to float and scale to [0, 1]
        cv::Mat float_image;
        resized.convertTo(float_image, CV_32FC3, 1.0f / 255.0f);

        // 2. Create Torch Tensor from OpenCV Mat
        // Shape of float_image: [Height, Width, Channels]
        auto input_tensor = torch::from_blob(float_image.data, {1, _input_height, _input_width, 3}, torch::kFloat32);

        // Convert BHWC layout to BCHW (necessary for YOLO PyTorch model)
        input_tensor = input_tensor.permute({0, 3, 1, 2});

        // Move to target device (GPU or CPU)
        input_tensor = input_tensor.to(_device);

        // 3. Run model inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Disable gradient calculations during inference for speed and memory savings
        torch::NoGradGuard no_grad;
        auto outputs = _module.forward(inputs);

        // 4. Handle output formats
        torch::Tensor output_tensor;
        if (outputs.isTensor()) {
            output_tensor = outputs.toTensor();
        } else if (outputs.isTuple()) {
            output_tensor = outputs.toTuple()->elements()[0].toTensor();
        } else {
            logger::error("[FaceDetection] Unexpected model output format");
            return bboxes;
        }

        // Output shape expected: [1, channels, 8400]
        output_tensor = output_tensor.to(torch::kCPU);
        output_tensor = output_tensor.squeeze(0);       // shape: [channels, 8400]

        int channels = output_tensor.size(0); // dynamic channels (e.g. 5 or 6)
        int cols = output_tensor.size(1);     // 8400

        output_tensor = output_tensor.transpose(0, 1).contiguous();   // shape: [8400, channels]
        float* data = output_tensor.data_ptr<float>();

        const float conf_threshold = 0.5f;
        std::vector<cv::Rect> candidate_boxes;
        std::vector<float> confidences;

        for (int i = 0; i < cols; i++) {
            float obj_conf = data[i * channels + 4];
            if (obj_conf > conf_threshold) {
                float xc = data[i * channels + 0];
                float yc = data[i * channels + 1];
                float w = data[i * channels + 2];
                float h = data[i * channels + 3];

                // Scale bounding box back to original image size
                float scale_x = (float)image.cols / _input_width;
                float scale_y = (float)image.rows / _input_height;

                float x1 = (xc - w / 2.0f) * scale_x;
                float y1 = (yc - h / 2.0f) * scale_y;
                float width = w * scale_x;
                float height = h * scale_y;

                // Apply padding based on center of current bounding box
                float pad_x = width * padding_w;
                float pad_y = height * padding_h;
                float new_x1 = x1 - pad_x / 2.0f;
                float new_y1 = y1 - pad_y / 2.0f;
                float new_width = width + pad_x;
                float new_height = height + pad_y;

                cv::Rect bbox(
                    std::max(0.0f, new_x1),
                    std::max(0.0f, new_y1),
                    std::max(0.0f, new_width),
                    std::max(0.0f, new_height)
                );

                bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
                if (bbox.width > 0 && bbox.height > 0) {
                    candidate_boxes.push_back(bbox);
                    confidences.push_back(obj_conf);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(candidate_boxes, confidences, conf_threshold, nms_threshold, indices);

        for (int idx : indices) {
            bboxes.push_back(candidate_boxes[idx]);
        }
    }
    catch (const c10::Error& e) {
        logger::error("[FaceDetection] LibTorch error during process: {}", e.what());
    }
    catch (const std::exception& e) {
        logger::error("[FaceDetection] Exception during process: {}", e.what());
    }

    return bboxes;
}
