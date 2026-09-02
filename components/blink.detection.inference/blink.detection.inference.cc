/**
 * @file blink.detection.inference.cc
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Blink Detection Inference Component using BlinkLinMulT with LibTorch
 * @version 0.1
 * @date 2026-09-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "blink.detection.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace std;
namespace fs = std::filesystem;
using json = nlohmann::json;

/* create component instance */
static blink_detection_inference* _instance = nullptr;
flame::component::Object* Create() {
    if (!_instance) _instance = new blink_detection_inference();
    return _instance;
}
void Release() {
    if (_instance) {
        delete _instance;
        _instance = nullptr;
    }
}

blink_detection_inference::blink_detection_inference() {
}

bool blink_detection_inference::onInit() {
    try {
        logger::info("[{}] Initializing blink detection inference component", getName());

        /* Load component parameters from profile JSON */
        json parameters = getProfile()->parameters();

        _model_path = parameters.value("model_path", "bin/x86_64/models/blinklinmult-union.torchscript");
        _gpu_id = parameters.value("gpu_id", 0);
        _seq_len = parameters.value("seq_len", 15);
        _crop_width = parameters.value("crop_width", 64);
        _crop_height = parameters.value("crop_height", 64);
        _threshold = parameters.value("threshold", 0.5f);
        _eye_selection = parameters.value("eye_selection", "left");
        _show_info = parameters.value("show_info", true);
        _visualize = parameters.value("visualize", true);
        _input_port = parameters.value("input_port", "image_stream_1");
        _output_data_port = parameters.value("output_data_port", "blink_result");
        _output_monitor_port = parameters.value("output_monitor_port", "image_stream_1_processed_monitor");

        /* Monitor port target resolution */
        const json& dataport_cfg = getProfile()->dataPort();
        if (dataport_cfg.contains(_output_monitor_port)) {
            const auto& port_cfg = dataport_cfg[_output_monitor_port];
            if (port_cfg.contains("resolution")) {
                const auto& r = port_cfg["resolution"];
                if (r.contains("width") && r.contains("height")) {
                    _target_width = r["width"].get<int>();
                    _target_height = r["height"].get<int>();
                }
            }
        }

        /* Check candidate paths for model file */
        if (!fs::exists(_model_path)) {
            std::vector<std::string> candidates = {
                "bin/x86_64/models/blinklinmult-union.torchscript",
                "models/blinklinmult-union.torchscript",
                "bin/x86_64/models/blinklinmul-union.torchscript",
                "models/blinklinmul-union.torchscript",
                "test/blink_detection/blinklinmult-union.torchscript",
                "test/blink_detection/blinklinmult-union.ts"
            };
            for (const auto& cand : candidates) {
                if (fs::exists(cand)) {
                    logger::info("[{}] Model path '{}' not found, using candidate: '{}'", getName(), _model_path, cand);
                    _model_path = cand;
                    break;
                }
            }
        }

        /* Load TorchScript model */
        if (!_load_model(_model_path, _gpu_id)) {
            logger::error("[{}] Failed to load TorchScript model: {}", getName(), _model_path);
            return false;
        }

        /* Try loading OpenCV Haar Cascades for automatic eye localization */
        std::vector<std::string> face_cascade_paths = {
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        };
        std::vector<std::string> eye_cascade_paths = {
            "/usr/share/opencv4/haarcascades/haarcascade_eye.xml",
            "/usr/share/opencv/haarcascades/haarcascade_eye.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml"
        };

        for (const auto& p : face_cascade_paths) {
            if (fs::exists(p) && _face_cascade.load(p)) {
                logger::info("[{}] Loaded face cascade from {}", getName(), p);
                break;
            }
        }
        for (const auto& p : eye_cascade_paths) {
            if (fs::exists(p) && _eye_cascade.load(p)) {
                logger::info("[{}] Loaded eye cascade from {}", getName(), p);
                break;
            }
        }
        _has_cascade = !_face_cascade.empty() && !_eye_cascade.empty();

        /* Start inference worker thread */
        _worker_stop.store(false);
        _inference_worker = std::thread(&blink_detection_inference::_inference_process, this);

        logger::info("[{}] Blink detection inference component successfully initialized", getName());
        return true;
    }
    catch (const json::exception& e) {
        logger::error("[{}] Profile JSON error: {}", getName(), e.what());
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[{}] Initialization error: {}", getName(), e.what());
        return false;
    }
}

void blink_detection_inference::onLoop() {
}

void blink_detection_inference::onClose() {
    logger::info("[{}] Closing blink detection inference component", getName());

    _worker_stop.store(true);
    _queue_cv.notify_all();

    if (_inference_worker.joinable()) {
        _inference_worker.join();
    }

    logger::info("[{}] Blink detection inference component stopped", getName());
}

void blink_detection_inference::onData(flame::component::ZData& data) {
    if (_worker_stop.load() || data.size() < 2) return;

    try {
        zmq::message_t tag_msg = data.pop();
        zmq::message_t img_msg = data.pop();

        std::string tag_str(static_cast<char*>(tag_msg.data()), tag_msg.size());
        json tag;
        try {
            tag = json::parse(tag_str);
        } catch (...) {
            tag = json::object();
        }

        int height = tag.value("height", 0);
        int width = tag.value("width", 0);
        int type = tag.value("type", CV_8UC3);
        int64_t timestamp = tag.value("timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

        cv::Mat decoded;
        if (height > 0 && width > 0 && img_msg.size() == static_cast<size_t>(height * width * (type == CV_8UC1 ? 1 : 3))) {
            cv::Mat raw(height, width, type, img_msg.data());
            decoded = raw.clone();
        } else {
            cv::Mat buf(1, img_msg.size(), CV_8UC1, img_msg.data());
            decoded = cv::imdecode(buf, cv::IMREAD_COLOR);
        }

        if (!decoded.empty()) {
            std::unique_lock<std::mutex> lock(_queue_mtx);
            if (_data_queue.size() >= _max_queue_size) {
                _data_queue.pop(); // Drop oldest frame to avoid latency buildup
            }
            _data_queue.push({decoded, tag, timestamp});
            lock.unlock();
            _queue_cv.notify_one();
        }
    }
    catch (const std::exception& e) {
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}

bool blink_detection_inference::_load_model(const std::string& model_path, int gpu_id) {
    try {
        if (!fs::exists(model_path)) {
            logger::error("[{}] Model file does not exist: {}", getName(), model_path);
            return false;
        }

        if (torch::cuda::is_available() && gpu_id >= 0) {
            _device = torch::Device(torch::kCUDA, gpu_id);
            logger::info("[{}] Using CUDA GPU device: {}", getName(), gpu_id);
        } else {
            _device = torch::Device(torch::kCPU);
            logger::warn("[{}] CUDA not available or CPU specified. Using CPU", getName());
        }

        _module = torch::jit::load(model_path, _device);
        _module.eval();

        /* Warmup forward pass */
        torch::NoGradGuard no_grad;
        auto dummy_low = torch::zeros({1, _seq_len, 3, _crop_height, _crop_width}, _device);
        auto dummy_high = torch::zeros({1, _seq_len, 160}, _device);
        std::vector<torch::jit::IValue> inputs = {dummy_low, dummy_high};
        _module.forward(inputs);

        _is_model_loaded = true;
        logger::info("[{}] Successfully loaded and warmed up TorchScript model from {}", getName(), model_path);
        return true;
    }
    catch (const c10::Error& e) {
        logger::error("[{}] LibTorch error loading model: {}", getName(), e.what());
        _is_model_loaded = false;
        return false;
    }
    catch (const std::exception& e) {
        logger::error("[{}] Exception loading model: {}", getName(), e.what());
        _is_model_loaded = false;
        return false;
    }
}

cv::Rect blink_detection_inference::_detect_face_or_estimate_roi(const cv::Mat& image) {
    if (image.empty()) return cv::Rect(0, 0, 0, 0);

    if (_has_cascade) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        _face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(image.cols / 6, image.rows / 6));
        if (!faces.empty()) {
            auto largest = std::max_element(faces.begin(), faces.end(), [](const cv::Rect& a, const cv::Rect& b) {
                return a.area() < b.area();
            });
            return *largest;
        }
    }

    // Default geometric driver face ROI assumption: center-upper 50%
    int fw = static_cast<int>(image.cols * 0.50f);
    int fh = static_cast<int>(image.rows * 0.60f);
    int fx = (image.cols - fw) / 2;
    int fy = static_cast<int>(image.rows * 0.15f);
    return cv::Rect(fx, fy, fw, fh) & cv::Rect(0, 0, image.cols, image.rows);
}

std::pair<cv::Rect, cv::Rect> blink_detection_inference::_extract_eye_rois(const cv::Rect& face_box, const cv::Size& img_size) {
    cv::Rect left_eye, right_eye;
    if (face_box.width <= 0 || face_box.height <= 0) {
        return {left_eye, right_eye};
    }

    int eye_w = static_cast<int>(face_box.width * 0.33f);
    int eye_h = static_cast<int>(face_box.height * 0.26f);
    int eye_y = face_box.y + static_cast<int>(face_box.height * 0.23f);

    int left_x = face_box.x + static_cast<int>(face_box.width * 0.14f);
    int right_x = face_box.x + static_cast<int>(face_box.width * 0.53f);

    left_eye = cv::Rect(left_x, eye_y, eye_w, eye_h) & cv::Rect(0, 0, img_size.width, img_size.height);
    right_eye = cv::Rect(right_x, eye_y, eye_w, eye_h) & cv::Rect(0, 0, img_size.width, img_size.height);

    return {left_eye, right_eye};
}

torch::Tensor blink_detection_inference::_preprocess_eye_patch(const cv::Mat& eye_img) {
    cv::Mat resized;
    cv::resize(eye_img, resized, cv::Size(_crop_width, _crop_height), 0, 0, cv::INTER_CUBIC);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // ImageNet standardization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    cv::merge(channels, float_img);

    // Shape: (3, 64, 64) on CPU
    auto tensor = torch::from_blob(float_img.data, {_crop_height, _crop_width, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).clone(); // (3, H, W)
    return tensor;
}

void blink_detection_inference::_inference_process() {
    logger::info("[{}] Blink detection inference worker thread started", getName());

    auto last_fps_time = std::chrono::high_resolution_clock::now();
    int frame_counter = 0;
    double fps = 0.0;
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 90};

    while (!_worker_stop.load()) {
        blink_detection::QueuedFrame frame_item;
        {
            std::unique_lock<std::mutex> lock(_queue_mtx);
            _queue_cv.wait_for(lock, std::chrono::milliseconds(50), [this]() {
                return !_data_queue.empty() || _worker_stop.load();
            });

            if (_worker_stop.load()) break;
            if (_data_queue.empty()) continue;

            frame_item = std::move(_data_queue.front());
            _data_queue.pop();
        }

        try {
            cv::Mat decoded = frame_item.image;
            json tag = frame_item.tag;
            int64_t timestamp = frame_item.timestamp;

            if (decoded.empty()) {
                continue;
            }

            // Calculate FPS
            frame_counter++;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_fps = std::chrono::duration<double>(now - last_fps_time).count();
            if (elapsed_fps >= 1.0) {
                fps = frame_counter / elapsed_fps;
                frame_counter = 0;
                last_fps_time = now;
            }

            /* Extract Face & Eye ROIs */
            cv::Rect face_box = _detect_face_or_estimate_roi(decoded);
            auto [left_eye_roi, right_eye_roi] = _extract_eye_rois(face_box, decoded.size());

            // Choose eye patch based on configuration
            cv::Rect active_eye_roi = left_eye_roi;
            if (_eye_selection == "right" && right_eye_roi.area() > 0) {
                active_eye_roi = right_eye_roi;
            }

            cv::Mat eye_patch;
            if (active_eye_roi.area() > 0) {
                eye_patch = decoded(active_eye_roi).clone();
            } else {
                eye_patch = cv::Mat::zeros(_crop_height, _crop_width, CV_8UC3);
            }

            /* Preprocess eye patch into (3, 64, 64) tensor */
            torch::Tensor low_feat = _preprocess_eye_patch(eye_patch);

            /* High level features (160 dims): initialize with zeros or tag features */
            torch::Tensor high_feat = torch::zeros({160}, torch::kFloat32);

            // Populate high level features if available in tag
            if (tag.contains("headpose") && tag["headpose"].is_array() && tag["headpose"].size() >= 3) {
                high_feat[0] = tag["headpose"][0].get<float>();
                high_feat[1] = tag["headpose"][1].get<float>();
                high_feat[2] = tag["headpose"][2].get<float>();
            }
            if (tag.contains("ear") && tag["ear"].is_number()) {
                high_feat[159] = tag["ear"].get<float>();
            }

            /* Push to temporal sliding window buffer */
            blink_detection::EyeFrameFeature frame_feat;
            frame_feat.low_feature = low_feat;
            frame_feat.high_feature = high_feat;
            frame_feat.timestamp = timestamp;
            frame_feat.face_rect = face_box;
            frame_feat.left_eye_rect = left_eye_roi;
            frame_feat.right_eye_rect = right_eye_roi;
            frame_feat.has_eye = (active_eye_roi.area() > 0);

            _sequence_buffer.push_back(frame_feat);
            if (_sequence_buffer.size() > static_cast<size_t>(_seq_len)) {
                _sequence_buffer.pop_front();
            }

            /* Run Inference when sequence buffer is full (15 frames) */
            if (_sequence_buffer.size() == static_cast<size_t>(_seq_len) && _is_model_loaded) {
                std::vector<torch::Tensor> low_list;
                std::vector<torch::Tensor> high_list;
                low_list.reserve(_seq_len);
                high_list.reserve(_seq_len);

                for (const auto& f : _sequence_buffer) {
                    low_list.push_back(f.low_feature);
                    high_list.push_back(f.high_feature);
                }

                // input_low: (1, 15, 3, 64, 64)
                // input_high: (1, 15, 160)
                auto input_low = torch::stack(low_list, 0).unsqueeze(0).to(_device);
                auto input_high = torch::stack(high_list, 0).unsqueeze(0).to(_device);

                torch::NoGradGuard no_grad;
                std::vector<torch::jit::IValue> model_inputs = {input_low, input_high};
                auto output = _module.forward(model_inputs);

                float cls_prob = 0.0f;
                float seq_prob = 0.0f;

                if (output.isTuple()) {
                    auto elements = output.toTuple()->elements();
                    if (elements.size() >= 1 && elements[0].isTensor()) {
                        auto y_cls = elements[0].toTensor();
                        cls_prob = torch::sigmoid(y_cls).item<float>();
                    }
                    if (elements.size() >= 2 && elements[1].isTensor()) {
                        auto y_seq = elements[1].toTensor(); // (1, 15, 1)
                        seq_prob = torch::sigmoid(y_seq[0][-1][0]).item<float>();
                    }
                } else if (output.isTensor()) {
                    auto y = output.toTensor();
                    cls_prob = torch::sigmoid(y).item<float>();
                    seq_prob = cls_prob;
                }

                // Weighted combination of sequence score and classification score
                _current_blink_prob = 0.7f * cls_prob + 0.3f * seq_prob;
                bool is_blinking = (_current_blink_prob >= _threshold);

                // State transition & blink count
                if (is_blinking && !_prev_is_blinking) {
                    _blink_count++;
                    _last_blink_time = std::chrono::steady_clock::now();
                }
                _prev_is_blinking = is_blinking;

                // PERCLOS calculation (percentage of eye closure over last N frames)
                _blink_history.push_back(is_blinking);
                if (_blink_history.size() > _perclos_history_size) {
                    _blink_history.pop_front();
                }

                int closed_count = std::count(_blink_history.begin(), _blink_history.end(), true);
                _current_perclos = (_blink_history.empty()) ? 0.0f : (static_cast<float>(closed_count) / _blink_history.size());
            }

            /* Construct Output Result JSON */
            json result_tag;
            result_tag["timestamp"] = timestamp;
            result_tag["blink_prob"] = _current_blink_prob;
            result_tag["is_blinking"] = _prev_is_blinking;
            result_tag["blink_count"] = _blink_count;
            result_tag["perclos"] = _current_perclos;
            result_tag["fps"] = fps;

            /* 1. Dispatch data result */
            flame::component::ZData out_data_msg;
            out_data_msg.from = _output_data_port;
            out_data_msg.meta = result_tag.dump();
            dispatch(_output_data_port, out_data_msg);

            /* 2. Dispatch Monitor Visualized Image if enabled */
            if (_visualize) {
                cv::Mat out_vis;
                if (_target_width > 0 && _target_height > 0) {
                    cv::resize(decoded, out_vis, cv::Size(_target_width, _target_height));
                } else {
                    out_vis = decoded.clone();
                }

                float scale_x = static_cast<float>(out_vis.cols) / decoded.cols;
                float scale_y = static_cast<float>(out_vis.rows) / decoded.rows;

                // Draw face & eye rectangles
                if (face_box.area() > 0) {
                    cv::Rect scaled_face(
                        face_box.x * scale_x,
                        face_box.y * scale_y,
                        face_box.width * scale_x,
                        face_box.height * scale_y
                    );
                    cv::rectangle(out_vis, scaled_face, cv::Scalar(255, 200, 0), 1);
                }

                if (left_eye_roi.area() > 0) {
                    cv::Rect scaled_left(
                        left_eye_roi.x * scale_x,
                        left_eye_roi.y * scale_y,
                        left_eye_roi.width * scale_x,
                        left_eye_roi.height * scale_y
                    );
                    cv::rectangle(out_vis, scaled_left, _prev_is_blinking ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
                }

                if (right_eye_roi.area() > 0) {
                    cv::Rect scaled_right(
                        right_eye_roi.x * scale_x,
                        right_eye_roi.y * scale_y,
                        right_eye_roi.width * scale_x,
                        right_eye_roi.height * scale_y
                    );
                    cv::rectangle(out_vis, scaled_right, _prev_is_blinking ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
                }

                // Overlay Status Text
                if (_show_info) {
                    // Status Badge
                    std::string status_str = _prev_is_blinking ? "[BLINKING]" : "[EYES OPEN]";
                    cv::Scalar status_col = _prev_is_blinking ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                    cv::putText(out_vis, status_str, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2, cv::LINE_AA);

                    // Metrics
                    char prob_buf[64];
                    snprintf(prob_buf, sizeof(prob_buf), "Blink Prob: %.2f", _current_blink_prob);
                    cv::putText(out_vis, prob_buf, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                    char count_buf[64];
                    snprintf(count_buf, sizeof(count_buf), "Blink Count: %d", _blink_count);
                    cv::putText(out_vis, count_buf, cv::Point(20, 105), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                    char perclos_buf[64];
                    snprintf(perclos_buf, sizeof(perclos_buf), "PERCLOS: %.1f%%", _current_perclos * 100.0f);
                    cv::putText(out_vis, perclos_buf, cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

                    char fps_buf[32];
                    snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
                    cv::putText(out_vis, fps_buf, cv::Point(out_vis.cols - 120, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }

                // Encode JPEG and dispatch
                std::vector<uchar> jpeg_buf;
                if (cv::imencode(".jpg", out_vis, jpeg_buf, encode_params)) {
                    json monitor_tag = result_tag;
                    monitor_tag["width"] = out_vis.cols;
                    monitor_tag["height"] = out_vis.rows;
                    monitor_tag["type"] = out_vis.type();

                    flame::component::ZData monitor_msg;
                    monitor_msg.from = _output_monitor_port;
                    monitor_msg.meta = monitor_tag.dump();
                    monitor_msg.addmem(jpeg_buf.data(), jpeg_buf.size());
                    dispatch(_output_monitor_port, monitor_msg);
                }
            }
        }
        catch (const std::exception& e) {
            logger::error("[{}] Error in inference processing loop: {}", getName(), e.what());
        }
    }

    logger::info("[{}] Blink detection inference worker thread stopped", getName());
}
