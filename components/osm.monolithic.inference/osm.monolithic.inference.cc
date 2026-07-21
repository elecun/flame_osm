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
        
        _show_info = parameters.value("show_info", true);
        logger::info("[{}] Show info parameter: {}", getName(), _show_info);

        std::string model_path = "bin/x86_64/models/yolo11n-face.pt";
        int gpu_id = 0;
        if (parameters.contains("face_detection")) {
            const auto& fd_params = parameters["face_detection"];
            model_path = fd_params.value("model_path", model_path);
            gpu_id = fd_params.value("gpu_id", gpu_id);
            _nms_threshold = fd_params.value("nms", _nms_threshold);
            if (fd_params.contains("padding") && fd_params["padding"].is_array() && fd_params["padding"].size() == 2) {
                _padding_w = fd_params["padding"][0].get<float>();
                _padding_h = fd_params["padding"][1].get<float>();
                logger::info("[{}] Loaded face detection padding: w={}, h={}", getName(), _padding_w, _padding_h);
            }
        }

        std::string body_model_path = "bin/x86_64/models/yolo26m-pose.torchscript";
        int body_gpu_id = 0;
        if (parameters.contains("body_pose_estimation")) {
            const auto& bp_params = parameters["body_pose_estimation"];
            body_model_path = bp_params.value("model_path", body_model_path);
            body_gpu_id = bp_params.value("gpu_id", body_gpu_id);
        }

        std::string landmark_model_path = "/home/osm/dev/flame_osm/bin/x86_64/models/face_alignment_2d_fan_cuda.torchscript";
        int landmark_gpu_id = 0;
        if (parameters.contains("2d_face_landmark")) {
            const auto& fl_params = parameters["2d_face_landmark"];
            landmark_model_path = fl_params.value("model_path", landmark_model_path);
            landmark_gpu_id = fl_params.value("gpu_id", landmark_gpu_id);
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

        _body_pose_estimator = std::make_unique<body_pose_estimation>();
        if (!_body_pose_estimator->loadModel(body_model_path, body_gpu_id)) {
            logger::error("[{}] Failed to load body pose estimation model: {}", getName(), body_model_path);
            return false;
        }

        _face_landmark_2d = std::make_unique<face_landmark_2d>();
        if (!_face_landmark_2d->loadModel(landmark_model_path, landmark_gpu_id)) {
            logger::error("[{}] Failed to load 2D face landmark model: {}", getName(), landmark_model_path);
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

    if (_face_landmark_2d) {
        _face_landmark_2d.reset();
        logger::info("[{}] Face landmark 2d instance successfully released", getName());
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

    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 100};
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

                logger::debug("[{}] Received Stream 1 Image - Size: {}x{}, Channels: {}, Type: {}", getName(), image.cols, image.rows, image.channels(), image.type());

                try {
                    auto proc_start = std::chrono::high_resolution_clock::now();

                    /* 1. Run YOLO11-Face detection */
                    std::vector<cv::Rect> bboxes = _face_detector->process(image, _nms_threshold, _padding_w, _padding_h);

                    /* 2. Run YOLO-Pose estimation */
                    std::vector<body_pose::PoseResult> poses = _body_pose_estimator->process(image, 0.5f, 0.45f);

                    /* 3. Run FAN 2D Face Landmark estimation */
                    std::vector<face_landmark::LandmarkResult> landmarks_2d = _face_landmark_2d->process(image, bboxes);

                    /* 4. Resize if target monitor resolution is defined */
                    cv::Mat out_image;
                    if (_has_target_resolution && (_target_width != image.cols || _target_height != image.rows)) {
                        cv::resize(image, out_image, cv::Size(_target_width, _target_height), 0, 0, cv::INTER_LINEAR);
                    } else {
                        out_image = image;
                    }

                    float scale_x = (float)out_image.cols / image.cols;
                    float scale_y = (float)out_image.rows / image.rows;

                    /* 5. Draw face bounding boxes on the resized out_image */
                    for (const auto& box : bboxes) {
                        cv::Rect scaled_box(
                            box.x * scale_x,
                            box.y * scale_y,
                            box.width * scale_x,
                            box.height * scale_y
                        );
                        cv::rectangle(out_image, scaled_box, cv::Scalar(0, 255, 0), 3);
                    }

                    /* 6. Draw 2D face landmarks and FAN crop region (RED Bounding Box) on out_image */
                    for (const auto& res : landmarks_2d) {
                        // Draw RED bounding box for the actual landmark detection area (FAN Crop Patch)
                        cv::Rect scaled_crop_box(
                            res.crop_bbox.x * scale_x,
                            res.crop_bbox.y * scale_y,
                            res.crop_bbox.width * scale_x,
                            res.crop_bbox.height * scale_y
                        );
                        cv::rectangle(out_image, scaled_crop_box, cv::Scalar(0, 0, 255), 2);

                        // Draw 68 landmark points
                        for (const auto& pt : res.points) {
                            cv::circle(out_image, 
                                       cv::Point2f(pt.x * scale_x, pt.y * scale_y), 
                                       2, cv::Scalar(0, 255, 255), -1);
                        }
                    }

                    /* 7. Draw body poses on the resized out_image (limbs & body only) */
                    const std::vector<std::pair<int, int>> SKELETON_CONNECTIONS = {
                        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
                        {5, 11}, {6, 12}, {11, 12},
                        {11, 13}, {13, 15}, {12, 14}, {14, 16}
                    };

                    for (const auto& pose : poses) {
                        // Draw skeleton connections
                        for (const auto& conn : SKELETON_CONNECTIONS) {
                            if (conn.first < (int)pose.keypoints.size() && conn.second < (int)pose.keypoints.size()) {
                                const auto& kp1 = pose.keypoints[conn.first];
                                const auto& kp2 = pose.keypoints[conn.second];
                                if (kp1.confidence > 0.5f && kp2.confidence > 0.5f) {
                                    cv::line(out_image, 
                                             cv::Point(kp1.x * scale_x, kp1.y * scale_y), 
                                             cv::Point(kp2.x * scale_x, kp2.y * scale_y), 
                                             cv::Scalar(0, 255, 255), 2);
                                }
                            }
                        }

                        // Draw keypoints (excluding face landmarks: index 0 to 4)
                        for (int k = 5; k < (int)pose.keypoints.size(); ++k) {
                            const auto& kpt = pose.keypoints[k];
                            if (kpt.confidence > 0.5f) {
                                cv::circle(out_image, 
                                           cv::Point(kpt.x * scale_x, kpt.y * scale_y), 
                                           4, cv::Scalar(0, 0, 255), -1);
                            }
                        }
                    }

                    // Calculate FPS
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(now - last_time_1).count();
                    last_time_1 = now;
                    double fps = (elapsed > 0) ? (1.0 / elapsed) : 0.0;

                    if (_show_info) {
                        auto now_sys = std::chrono::system_clock::now();
                        auto time_t_now = std::chrono::system_clock::to_time_t(now_sys);
                        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_sys.time_since_epoch()) % 1000;
                        std::tm tm_now;
                        localtime_r(&time_t_now, &tm_now);
                        char time_str[64];
                        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_now);
                        char ms_str[8];
                        snprintf(ms_str, sizeof(ms_str), "%03d", (int)ms.count());
                        std::string datetime_str = std::string(time_str) + "." + ms_str;

                        char fps_str[32];
                        snprintf(fps_str, sizeof(fps_str), "%.1f", fps);

                        cv::putText(out_image, datetime_str, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        cv::putText(out_image, fps_str, cv::Point(540, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }

                    /* 4. Encode as JPEG */
                    std::vector<uchar> jpeg_buf;
                    if (cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        
                        /* 5. Construct metadata tags */
                        json tag;
                        tag["width"] = out_image.cols;
                        tag["height"] = out_image.rows;
                        tag["type"] = out_image.type();
                        tag["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                        tag["cam_channel"] = 1;
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

                    auto proc_end = std::chrono::high_resolution_clock::now();
                    double proc_elapsed = std::chrono::duration<double, std::milli>(proc_end - proc_start).count();
                    logger::info("[{}] Stream 1 inference worker thread loop execution time: {} ms", getName(), proc_elapsed);
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

                logger::info("[{}] Received Stream 2 Image - Size: {}x{}, Channels: {}, Type: {}", getName(), image.cols, image.rows, image.channels(), image.type());

                try {
                    auto proc_start = std::chrono::high_resolution_clock::now();

                    /* 1. Run YOLO11-Face detection */
                    std::vector<cv::Rect> bboxes = _face_detector->process(image, _nms_threshold, _padding_w, _padding_h);

                    /* 3. Resize if target monitor resolution is defined */
                    cv::Mat out_image;
                    if (_has_target_resolution && (_target_width != image.cols || _target_height != image.rows)) {
                        cv::resize(image, out_image, cv::Size(_target_width, _target_height), 0, 0, cv::INTER_LINEAR);
                    } else {
                        out_image = image;
                    }

                    float scale_x = (float)out_image.cols / image.cols;
                    float scale_y = (float)out_image.rows / image.rows;

                    /* 2. Draw bounding boxes on the resized out_image */
                    for (const auto& box : bboxes) {
                        cv::Rect scaled_box(
                            box.x * scale_x,
                            box.y * scale_y,
                            box.width * scale_x,
                            box.height * scale_y
                        );
                        cv::rectangle(out_image, scaled_box, cv::Scalar(0, 255, 0), 2);
                    }

                    // Calculate FPS
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(now - last_time_2).count();
                    last_time_2 = now;
                    double fps = (elapsed > 0) ? (1.0 / elapsed) : 0.0;

                    if (_show_info) {
                        auto now_sys = std::chrono::system_clock::now();
                        auto time_t_now = std::chrono::system_clock::to_time_t(now_sys);
                        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_sys.time_since_epoch()) % 1000;
                        std::tm tm_now;
                        localtime_r(&time_t_now, &tm_now);
                        char time_str[64];
                        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_now);
                        char ms_str[8];
                        snprintf(ms_str, sizeof(ms_str), "%03d", (int)ms.count());
                        std::string datetime_str = std::string(time_str) + "." + ms_str;

                        char fps_str[32];
                        snprintf(fps_str, sizeof(fps_str), "%.1f", fps);

                        cv::putText(out_image, datetime_str, cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        cv::putText(out_image, fps_str, cv::Point(380, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }

                    /* 4. Encode as JPEG */
                    std::vector<uchar> jpeg_buf;
                    if (cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        
                        /* 5. Construct metadata tags */
                        json tag;
                        tag["width"] = out_image.cols;
                        tag["height"] = out_image.rows;
                        tag["type"] = out_image.type();
                        tag["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                        tag["cam_channel"] = 2;
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

                    auto proc_end = std::chrono::high_resolution_clock::now();
                    double proc_elapsed = std::chrono::duration<double, std::milli>(proc_end - proc_start).count();
                    logger::info("[{}] Stream 2 inference worker thread loop execution time: {} ms", getName(), proc_elapsed);
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

