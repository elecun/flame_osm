#include "osm.monolithic.inference.hpp"
#include <flame/log.hpp>
#include <dep/json.hpp>
#include <chrono>

using json = nlohmann::json;

/* create component instance */
static osm_monolithic_inference* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new osm_monolithic_inference(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

osm_monolithic_inference::osm_monolithic_inference() {
}

bool osm_monolithic_inference::onInit(){
    try{
        const json& parameters = getProfile()->parameters();
        
        std::string model_path = "bin/x86_64/models/yolo11n-face.pt";
        int gpu_id = 0;
        if (parameters.contains("face_detection")) {
            const auto& fd_params = parameters["face_detection"];
            model_path = fd_params.value("model_path", model_path);
            gpu_id = fd_params.value("gpu_id", gpu_id);
        }

        /* Set stream enable flags from parameters */
        _enable_stream_1 = false;
        _enable_stream_2 = false;
        if (parameters.contains("use_image_stream") && parameters["use_image_stream"].is_array()) {
            for (const auto& val : parameters["use_image_stream"]) {
                if (val.is_number()) {
                    int id = val.get<int>();
                    if (id == 1) _enable_stream_1 = true;
                    else if (id == 2) _enable_stream_2 = true;
                }
            }
        } else {
            _enable_stream_1 = true; // default
        }
        logger::info("[{}] Enabled image streams: stream_1={}, stream_2={}", getName(), _enable_stream_1, _enable_stream_2);

        /* Load monitor port configuration for image_stream_1_processed_monitor */
        const json& dataport_cfg = getProfile()->dataPort();
        if (dataport_cfg.contains("image_stream_1_processed_monitor")) {
            const auto& port_cfg = dataport_cfg["image_stream_1_processed_monitor"];
            if (port_cfg.contains("resolution")) {
                const auto& r = port_cfg["resolution"];
                if (r.contains("width") && r.contains("height")) {
                    _has_target_resolution = true;
                    _target_width = r["width"].get<int>();
                    _target_height = r["height"].get<int>();
                    logger::info("[{}] Monitor port target resolution: {}x{}", getName(), _target_width, _target_height);
                }
            }
        }

        _face_detector = std::make_unique<face_detection>();
        if (!_face_detector->loadModel(model_path, gpu_id)) {
            logger::error("[{}] Failed to load face detection model: {}", getName(), model_path);
            return false;
        }

        /* Start Inference thread */
        _worker_stop.store(false);
        _inference_worker = std::thread(&osm_monolithic_inference::_inference_process, this);

        logger::info("[{}] Initialized osm.monolithic.inference component", getName());
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }
    catch(const std::exception& e){
        logger::error("[{}] Initialization Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void osm_monolithic_inference::onLoop(){
}

void osm_monolithic_inference::onClose(){
    logger::info("[{}] Closing osm.monolithic.inference component", getName());

    /* Stop Worker Thread */
    _worker_stop.store(true);
    if (_inference_worker.joinable()) {
        _inference_worker.join();
    }

    if (_face_detector) {
        _face_detector.reset();
        logger::info("[{}] Face detector instance successfully released", getName());
    }
}

void osm_monolithic_inference::onData(flame::component::ZData& data){
    try {
        std::string portname = data.from;

        if ((portname == "image_stream_1" && _enable_stream_1) || (portname == "image_stream_2" && _enable_stream_2)) {
            if (data.size() >= 2) {
                zmq::message_t tag_msg = data.pop();
                zmq::message_t img_msg = data.pop();

                std::string tag_str(static_cast<char*>(tag_msg.data()), tag_msg.size());
                json tag = json::parse(tag_str);
                int height = tag["height"].get<int>();
                int width = tag["width"].get<int>();
                int type = tag["type"].get<int>();

                // Restore image Mat from payload
                cv::Mat raw_img(height, width, type, img_msg.data());
                cv::Mat cloned_img = raw_img.clone();

                if (portname == "image_stream_1") {
                    std::lock_guard<std::mutex> lock(_img_mutex_1);
                    _latest_image_1 = cloned_img;
                } else if (portname == "image_stream_2") {
                    std::lock_guard<std::mutex> lock(_img_mutex_2);
                    _latest_image_2 = cloned_img;
                }
            }
        }
    }
    catch (const std::exception& e) {
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}

void osm_monolithic_inference::_inference_process() {
    logger::info("[{}] Inference worker thread started", getName());

    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};
    auto last_time_1 = std::chrono::high_resolution_clock::now();
    auto last_time_2 = std::chrono::high_resolution_clock::now();

    while (!_worker_stop.load()) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        bool processed = false;

        // Process Stream 1 if enabled
        if (_enable_stream_1) {
            cv::Mat image = getLatestImage1();
            if (!image.empty()) {
                processed = true;
                
                // Clear cache
                {
                    std::lock_guard<std::mutex> lock(_img_mutex_1);
                    _latest_image_1.release();
                }

                logger::debug("[{}] Received Stream 1 Image - Size: {}x{}, Channels: {}, Type: {}", 
                             getName(), image.cols, image.rows, image.channels(), image.type());

                try {
                    /* 1. Run YOLO11-Face detection */
                    std::vector<cv::Rect> bboxes = _face_detector->process(image);

                    /* 2. Draw bounding boxes on the image */
                    for (const auto& box : bboxes) {
                        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 3);
                    }

                    /* 3. Resize if target monitor resolution is defined */
                    cv::Mat out_image;
                    if (_has_target_resolution && (_target_width != image.cols || _target_height != image.rows)) {
                        cv::resize(image, out_image, cv::Size(_target_width, _target_height), 0, 0, cv::INTER_LINEAR);
                    } else {
                        out_image = image;
                    }

                    /* 4. Encode as JPEG */
                    std::vector<uchar> jpeg_buf;
                    if (cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        
                        /* 5. Construct metadata tags */
                        json tag;
                        tag["width"] = out_image.cols;
                        tag["height"] = out_image.rows;
                        tag["type"] = out_image.type();
                        tag["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        tag["cam_channel"] = 1;

                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed = std::chrono::duration<double>(now - last_time_1).count();
                        last_time_1 = now;
                        double fps = (elapsed > 0) ? (1.0 / elapsed) : 0.0;
                        tag["fps"] = fps;

                        /* 6. Send multipart message */
                        flame::component::ZData out_msg;
                        out_msg.from = "image_stream_1_processed_monitor";
                        out_msg.meta = tag.dump();
                        out_msg.addmem(jpeg_buf.data(), jpeg_buf.size());

                        if (!dispatch("image_stream_1_processed_monitor", out_msg)) {
                            logger::warn("[{}] Failed to dispatch processed image 1", getName());
                        } else {
                            logger::debug("[{}] Successfully dispatched processed image 1", getName());
                        }
                    }
                } catch (const std::exception& e) {
                    logger::error("[{}] Error in stream 1 inference: {}", getName(), e.what());
                }
            }
        }

        // Process Stream 2 if enabled
        if (_enable_stream_2) {
            cv::Mat image = getLatestImage2();
            if (!image.empty()) {
                processed = true;
                
                // Clear cache
                {
                    std::lock_guard<std::mutex> lock(_img_mutex_2);
                    _latest_image_2.release();
                }

                logger::info("[{}] Received Stream 2 Image - Size: {}x{}, Channels: {}, Type: {}", 
                             getName(), image.cols, image.rows, image.channels(), image.type());

                try {
                    /* 1. Run YOLO11-Face detection */
                    std::vector<cv::Rect> bboxes = _face_detector->process(image);

                    /* 2. Draw bounding boxes on the image */
                    for (const auto& box : bboxes) {
                        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 3);
                    }

                    /* 3. Resize if target monitor resolution is defined */
                    cv::Mat out_image;
                    if (_has_target_resolution && (_target_width != image.cols || _target_height != image.rows)) {
                        cv::resize(image, out_image, cv::Size(_target_width, _target_height), 0, 0, cv::INTER_LINEAR);
                    } else {
                        out_image = image;
                    }

                    /* 4. Encode as JPEG */
                    std::vector<uchar> jpeg_buf;
                    if (cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        
                        /* 5. Construct metadata tags */
                        json tag;
                        tag["width"] = out_image.cols;
                        tag["height"] = out_image.rows;
                        tag["type"] = out_image.type();
                        tag["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        tag["cam_channel"] = 2;

                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed = std::chrono::duration<double>(now - last_time_2).count();
                        last_time_2 = now;
                        double fps = (elapsed > 0) ? (1.0 / elapsed) : 0.0;
                        tag["fps"] = fps;

                        /* 6. Send multipart message */
                        flame::component::ZData out_msg;
                        out_msg.from = "image_stream_2_processed_monitor";
                        out_msg.meta = tag.dump();
                        out_msg.addmem(jpeg_buf.data(), jpeg_buf.size());

                        if (!dispatch("image_stream_2_processed_monitor", out_msg)) {
                            logger::warn("[{}] Failed to dispatch processed image 2", getName());
                        } else {
                            logger::info("[{}] Successfully dispatched processed image 2", getName());
                        }
                    }
                } catch (const std::exception& e) {
                    logger::error("[{}] Error in stream 2 inference: {}", getName(), e.what());
                }
            }
        }

        if (!processed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        /* Sleep to maintain maximum 30 FPS rate */
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        long long sleep_time = std::max(1LL, 33LL - duration);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }

    logger::info("[{}] Inference worker thread stopped", getName());
}

cv::Mat osm_monolithic_inference::getLatestImage1() {
    std::lock_guard<std::mutex> lock(_img_mutex_1);
    return _latest_image_1.clone();
}

cv::Mat osm_monolithic_inference::getLatestImage2() {
    std::lock_guard<std::mutex> lock(_img_mutex_2);
    return _latest_image_2.clone();
}
