#include "osm.monolithic.inference_v2.hpp"
#include <flame/log.hpp>
#include <dep/json.hpp>
#include <chrono>

using json = nlohmann::json;

/* create component instance */
static osm_monolithic_inference_v2* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new osm_monolithic_inference_v2(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

osm_monolithic_inference_v2::osm_monolithic_inference_v2() {
}

bool osm_monolithic_inference_v2::onInit(){
    try{
        const json& parameters = getProfile()->parameters();
        
        _show_info = parameters.value("show_info", true);
        logger::info("[{}] Show info parameter: {}", getName(), _show_info);

        _vertical_flip = parameters.value("vertical_flip", false);
        logger::info("[{}] Vertical flip parameter: {}", getName(), _vertical_flip);

        /* Model parameters & use flags */
        std::string face_det_model_path = "/home/iae-vc/dev/flame_osm/bin/x86_64/models/yolo11n-face.torchscript";
        int face_det_gpu_id = 0;

        _use_face_det = true;
        _use_face_analysis_e2e = true;
        _use_body_pose = true;

        auto get_json_float = [](const json& j, const std::vector<std::string>& keys, float default_val) -> float {
            for (const auto& k : keys) {
                if (j.contains(k)) {
                    if (j[k].is_number()) {
                        return j[k].get<float>();
                    } else if (j[k].is_string()) {
                        try {
                            return std::stof(j[k].get<std::string>());
                        } catch (...) {}
                    }
                }
            }
            return default_val;
        };

        if (parameters.contains("face_detection")) {
            const auto& fd_params = parameters["face_detection"];
            _use_face_det = fd_params.value("use", _use_face_det);
            face_det_model_path = fd_params.value("model_path", face_det_model_path);
            face_det_gpu_id = fd_params.value("gpu_id", face_det_gpu_id);
            _nms_threshold = get_json_float(fd_params, {"nms", "iou", "nms_thresh", "nms_threshold", "iou_thresh"}, _nms_threshold);
            _conf_threshold = get_json_float(fd_params, {"conf", "threshold", "conf_thresh", "conf_threshold", "confidence"}, _conf_threshold);
            _padding_scale = get_json_float(fd_params, {"padding_scale", "pad_scale", "scale"}, _padding_scale);
            _vis_face_det = fd_params.value("visualize", true);

            /* POI (Point of Interest) parameters */
            _use_poi = fd_params.value("use_poi", fd_params.value("use_roi", _use_poi));
            _poi_visualize = fd_params.value("poi_visualize", fd_params.value("roi_visualize", true));
            _poi_dist = get_json_float(fd_params, {"poi_dist", "dist", "poi_distance"}, _poi_dist);
            if (fd_params.contains("poi") && fd_params["poi"].is_array() && fd_params["poi"].size() >= 2) {
                _poi_x = fd_params["poi"][0].get<int>();
                _poi_y = fd_params["poi"][1].get<int>();
            } else if (fd_params.contains("roi") && fd_params["roi"].is_array() && fd_params["roi"].size() >= 2) {
                _poi_x = fd_params["roi"][0].get<int>();
                _poi_y = fd_params["roi"][1].get<int>();
            }

            if (fd_params.contains("padding") && fd_params["padding"].is_array() && fd_params["padding"].size() == 2) {
                _padding_w = fd_params["padding"][0].get<float>();
                _padding_h = fd_params["padding"][1].get<float>();
                logger::info("[{}] Loaded face detection padding: w={}, h={}", getName(), _padding_w, _padding_h);
            }
            logger::info("[{}] Face detection configured: model={}, gpu={}, conf={:.3f}, nms={:.3f}, padding_scale={:.3f}, use_poi={}, poi=[{}, {}], poi_dist={:.1f}, poi_visualize={}",
                         getName(), face_det_model_path, face_det_gpu_id, _conf_threshold, _nms_threshold, _padding_scale, _use_poi, _poi_x, _poi_y, _poi_dist, _poi_visualize);
        }

        std::string face_analysis_model_path = "/home/iae-vc/dev/flame_osm/bin/x86_64/models/dad_3dheads_e2e.torchscript";
        int face_analysis_gpu_id = 0;
        if (parameters.contains("face_analysis_e2e")) {
            const auto& fa_params = parameters["face_analysis_e2e"];
            _use_face_analysis_e2e = fa_params.value("use", _use_face_analysis_e2e);
            face_analysis_model_path = fa_params.value("model_path", face_analysis_model_path);
            face_analysis_gpu_id = fa_params.value("gpu_id", face_analysis_gpu_id);
            _vis_face_analysis_e2e = fa_params.value("visualize", true);
            _vis_landmarks_68 = fa_params.value("vis_landmarks_68", false);
            _vis_landmarks_191 = fa_params.value("vis_landmarks_191", true);
            _vis_head_pose = fa_params.value("vis_head_pose", true);
            _vis_square_box = fa_params.value("vis_square_box", true);
            _vis_head_mesh = fa_params.value("vis_head_mesh", false);
        }

        std::string body_model_path = "/home/iae-vc/dev/flame_osm/bin/x86_64/models/yolo26m-pose.torchscript";
        int body_gpu_id = 0;
        if (parameters.contains("body_pose_estimation")) {
            const auto& bp_params = parameters["body_pose_estimation"];
            _use_body_pose = bp_params.value("use", _use_body_pose);
            body_model_path = bp_params.value("model_path", body_model_path);
            body_gpu_id = bp_params.value("gpu_id", body_gpu_id);
            _vis_body_pose = bp_params.value("visualize", true);
        }

        std::string readiness_model_path = "/home/iae-vc/dev/flame_osm/bin/x86_64/models/iae_dms_251212.torchscript";
        int readiness_gpu_id = 1;
        float dr_threshold = 0.5f;
        float dr_readiness_low = 0.2f;
        float dr_readiness_moderate = 0.5f;
        float dr_readiness_high = 1.0f;
        if (parameters.contains("driver_readiness_estimation")) {
            const auto& dr_params = parameters["driver_readiness_estimation"];
            _use_driver_readiness = dr_params.value("use", _use_driver_readiness);
            readiness_model_path = dr_params.value("model_path", readiness_model_path);
            readiness_gpu_id = dr_params.value("gpu_id", readiness_gpu_id);
            _vis_driver_readiness = dr_params.value("visualize", true);
            dr_threshold = dr_params.value("threshold", dr_threshold);
            dr_readiness_low = dr_params.value("readiness_low", dr_readiness_low);
            dr_readiness_moderate = dr_params.value("readiness_moderate", dr_readiness_moderate);
            dr_readiness_high = dr_params.value("readiness_high", dr_readiness_high);
        }

        double ref_yaw = 0.0;
        double ref_pitch = 0.0;
        double sigma_yaw = 15.0;
        double sigma_pitch = 10.0;
        double t_window = 2.0;
        double readiness_low = 0.2;
        double readiness_moderate = 0.6;
        double readiness_high = 1.0;
        if (parameters.contains("driver_readiness_estimation_logical")) {
            const auto& drl_params = parameters["driver_readiness_estimation_logical"];
            _use_driver_readiness_logical = drl_params.value("use", _use_driver_readiness_logical);
            _vis_driver_readiness_logical = drl_params.value("visualize", true);
            ref_yaw = drl_params.value("ref_yaw", ref_yaw);
            ref_pitch = drl_params.value("ref_pitch", ref_pitch);
            sigma_yaw = drl_params.value("sigma_yaw", sigma_yaw);
            sigma_pitch = drl_params.value("sigma_pitch", sigma_pitch);
            t_window = drl_params.value("t_window", t_window);
            readiness_low = drl_params.value("readiness_low", readiness_low);
            readiness_moderate = drl_params.value("readiness_moderate", readiness_moderate);
            readiness_high = drl_params.value("readiness_high", readiness_high);
        }

        // Mutual exclusion of DMS estimators: deep learning has priority
        if (_use_driver_readiness) {
            _use_driver_readiness_logical = false;
            logger::info("[{}] Driver readiness estimation (deep learning) is enabled. Forcing logical readiness estimation to false.", getName());
        }

        /* Blink detection parameters */
        if (parameters.contains("blink_detection")) {
            const auto& bd_params = parameters["blink_detection"];
            _use_blink_detection = bd_params.value("use", _use_blink_detection);
            _vis_blink_detection = bd_params.value("visualize", _vis_blink_detection);
        }

        /* Stream configuration */
        if (parameters.contains("use_image_stream") && parameters["use_image_stream"].is_array()) {
            for (const auto& stream_id : parameters["use_image_stream"]) {
                if (stream_id.get<int>() == 1) _enable_stream_1 = true;
                if (stream_id.get<int>() == 2) _enable_stream_2 = true;
            }
        } else {
            _enable_stream_1 = true;
        }

        /* Output resolution from dataport */
        json dataport_cfg = getProfile()->dataPort();
        if (dataport_cfg.contains("image_stream_1_processed_monitor")) {
            const auto& monitor_cfg = dataport_cfg["image_stream_1_processed_monitor"];
            if (monitor_cfg.contains("resolution")) {
                _target_width = monitor_cfg["resolution"].value("width", 800);
                _target_height = monitor_cfg["resolution"].value("height", 450);
                _has_target_resolution = true;
                logger::info("[{}] Found target output resolution: {}x{}", getName(), _target_width, _target_height);
            }
        }

        /* Initialize Face Detector */
        if (_use_face_det) {
            _face_detector = std::make_unique<face_detection>();
            if (!_face_detector->loadModel(face_det_model_path, face_det_gpu_id)) {
                logger::error("[{}] Failed to load face detection model: {}", getName(), face_det_model_path);
                return false;
            }
        }

        /* Initialize DAD-3DHeads E2E Face Analyzer */
        if (_use_face_analysis_e2e) {
            _face_analyzer_e2e = std::make_unique<face_analysis_e2e>();
            if (!_face_analyzer_e2e->loadModel(face_analysis_model_path, face_analysis_gpu_id)) {
                logger::error("[{}] Failed to load DAD-3DHeads E2E model: {}", getName(), face_analysis_model_path);
                return false;
            }
        }

        /* Initialize Body Pose Estimator */
        if (_use_body_pose) {
            _body_pose_estimator = std::make_unique<body_pose_estimation>();
            if (!_body_pose_estimator->loadModel(body_model_path, body_gpu_id)) {
                logger::error("[{}] Failed to load body pose estimation model: {}", getName(), body_model_path);
                return false;
            }
        }

        /* Initialize Driver Readiness Estimators */
        if (_use_driver_readiness) {
            _driver_readiness_estimator = std::make_unique<driver_readiness_estimation>();
            if (!_driver_readiness_estimator->loadModel(readiness_model_path, readiness_gpu_id)) {
                logger::error("[{}] Failed to load driver readiness estimation model: {}", getName(), readiness_model_path);
                return false;
            }
            _driver_readiness_estimator->setParameters(dr_threshold, dr_readiness_low, dr_readiness_moderate, dr_readiness_high);
        }

        if (_use_driver_readiness_logical) {
            _driver_readiness_logical_estimator = std::make_unique<driver_readiness_estimation_logical>();
            _driver_readiness_logical_estimator->setParameters(
                ref_yaw, ref_pitch, sigma_yaw, sigma_pitch, t_window,
                readiness_low, readiness_moderate, readiness_high
            );
        }

        /* Initialize Blink Detection Analyzer */
        if (_use_blink_detection) {
            _blink_analyzer = std::make_unique<blink_analysis_component>();
            nlohmann::json bd_params;
            if (parameters.contains("blink_detection")) {
                bd_params = parameters["blink_detection"];
            }
            if (!_blink_analyzer->init(bd_params)) {
                logger::warn("[{}] Blink detection model init failed. Disabling blink detection.", getName());
                _use_blink_detection = false;
                _blink_analyzer.reset();
            } else {
                logger::info("[{}] Blink detection analyzer initialized successfully", getName());
            }
        }

        /* Start background worker thread */
        _worker_stop.store(false);
        _worker_finished.store(false);
        _inference_worker = std::thread(&osm_monolithic_inference_v2::_inference_process, this);
    }
    catch(const std::exception& e){
        logger::error("[{}] Exception during onInit : {}", getName(), e.what());
        return false;
    }

    return true;
}

void osm_monolithic_inference_v2::onLoop(){
    /* nothing in onLoop */
}

void osm_monolithic_inference_v2::onClose(){
    logger::info("[{}] Closing osm.monolithic.inference_v2 component", getName());

    _worker_stop.store(true);
    if (_inference_worker.joinable()) {
        auto start_wait = std::chrono::steady_clock::now();
        while (!_worker_finished.load() && 
               std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_wait).count() < 1000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (!_worker_finished.load()) {
            logger::warn("[{}] Inference worker thread did not stop in time, canceling...", getName());
            pthread_cancel(_inference_worker.native_handle());
        }
        _inference_worker.join();
    }
    logger::info("[{}] Monolithic inference V2 worker thread stopped", getName());

    if (_face_detector) {
        _face_detector.reset();
        logger::info("[{}] Face detector instance successfully released", getName());
    }

    if (_face_analyzer_e2e) {
        _face_analyzer_e2e.reset();
        logger::info("[{}] Face analyzer E2E instance successfully released", getName());
    }

    if (_body_pose_estimator) {
        _body_pose_estimator.reset();
        logger::info("[{}] Body pose estimator instance successfully released", getName());
    }

    if (_driver_readiness_estimator) {
        _driver_readiness_estimator.reset();
        logger::info("[{}] Driver readiness estimator instance successfully released", getName());
    }

    if (_driver_readiness_logical_estimator) {
        _driver_readiness_logical_estimator.reset();
        logger::info("[{}] Driver readiness logical estimator instance successfully released", getName());
    }

    if (_blink_analyzer) {
        _blink_analyzer.reset();
        logger::info("[{}] Blink analyzer instance successfully released", getName());
    }
}

void osm_monolithic_inference_v2::onData(flame::component::ZData& data){
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

                if (_vertical_flip) {
                    cv::flip(cloned_img, cloned_img, 1); // 좌우 반전
                }

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

cv::Mat osm_monolithic_inference_v2::getLatestImage1() {
    std::lock_guard<std::mutex> lock(_img_mutex_1);
    return _latest_image_1.clone();
}

cv::Mat osm_monolithic_inference_v2::getLatestImage2() {
    std::lock_guard<std::mutex> lock(_img_mutex_2);
    return _latest_image_2.clone();
}

void osm_monolithic_inference_v2::draw_readiness_graph(cv::Mat& image, int x, int y, int width, int height) {
    std::vector<std::pair<double, double>> time_score_pairs;
    auto now = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(_history_mutex);
        while (!_readiness_history.empty()) {
            double age = std::chrono::duration<double>(now - _readiness_history.front().first).count();
            if (age > 10.0) {
                _readiness_history.pop_front();
            } else {
                break;
            }
        }

        for (const auto& item : _readiness_history) {
            double age = std::chrono::duration<double>(now - item.first).count();
            time_score_pairs.push_back({age, item.second});
        }
    }

    if (x < 0 || y < 0 || x + width > image.cols || y + height > image.rows) {
        return;
    }

    cv::Rect bg_rect(x, y, width, height);
    cv::Mat overlay;
    image.copyTo(overlay);
    cv::rectangle(overlay, bg_rect, cv::Scalar(20, 20, 20), cv::FILLED);
    cv::addWeighted(overlay, 0.6, image, 0.4, 0, image);
    cv::rectangle(image, bg_rect, cv::Scalar(80, 80, 80), 1);

    int margin_l = 30;
    int margin_r = 10;
    int margin_t = 10;
    int margin_b = 15;

    int plot_w = width - margin_l - margin_r;
    int plot_h = height - margin_t - margin_b;

    int plot_x0 = x + margin_l;
    int plot_y0 = y + margin_t;

    cv::line(image, cv::Point(plot_x0, plot_y0 + plot_h), cv::Point(plot_x0 + plot_w, plot_y0 + plot_h), cv::Scalar(150, 150, 150), 1);
    cv::line(image, cv::Point(plot_x0, plot_y0), cv::Point(plot_x0, plot_y0 + plot_h), cv::Scalar(150, 150, 150), 1);

    cv::putText(image, "1.0", cv::Point(x + 2, plot_y0 + 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(180, 180, 180), 1);
    cv::putText(image, "0.0", cv::Point(x + 2, plot_y0 + plot_h), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(180, 180, 180), 1);
    cv::putText(image, "Readiness (10s)", cv::Point(plot_x0 + 5, plot_y0 - 2), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 255), 1);

    if (time_score_pairs.size() < 2) {
        return;
    }

    std::vector<cv::Point> pts;
    for (const auto& pair : time_score_pairs) {
        double age = pair.first;
        double score = std::clamp(pair.second, 0.0, 1.0);

        int px = plot_x0 + static_cast<int>((1.0 - (age / 10.0)) * plot_w);
        int py = plot_y0 + plot_h - static_cast<int>(score * plot_h);

        px = std::clamp(px, plot_x0, plot_x0 + plot_w);
        py = std::clamp(py, plot_y0, plot_y0 + plot_h);
        pts.push_back(cv::Point(px, py));
    }

    for (size_t i = 1; i < pts.size(); ++i) {
        double s = time_score_pairs[i].second;
        cv::Scalar line_col = cv::Scalar(0, 0, 255);
        if (s > 0.5) line_col = cv::Scalar(0, 255, 0);
        else if (s > 0.2) line_col = cv::Scalar(0, 255, 255);

        cv::line(image, pts[i - 1], pts[i], line_col, 2, cv::LINE_AA);
    }
}

void osm_monolithic_inference_v2::_inference_process() {
    logger::info("[{}] Inference worker thread started (V2)", getName());

    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 100};
    auto last_time_1 = std::chrono::high_resolution_clock::now();
    uint64_t frame_count = 0;
    auto last_idle_warning = std::chrono::steady_clock::now();

    while (!_worker_stop.load()) {
        if (_enable_stream_1) {
            cv::Mat image = getLatestImage1();
            if (!image.empty()) {
                // Clear cache
                {
                    std::lock_guard<std::mutex> lock(_img_mutex_1);
                    _latest_image_1.release();
                }

                frame_count++;
                auto t_frame_start = std::chrono::high_resolution_clock::now();

                try {
                    head_pose::PoseResult last_pose;
                    bool has_pose = false;
                    face_analysis::FaceAnalysisResult face_res;

                    /* 1. Run YOLO11-Face detection */
                    auto t_det_start = std::chrono::high_resolution_clock::now();
                    std::vector<FaceBox> detected_faces;
                    if (_use_face_det && _face_detector) {
                        detected_faces = _face_detector->detect(image, _conf_threshold, _nms_threshold, _padding_scale);
                        if (_use_poi) {
                            int best_face_idx = -1;
                            double min_dist = std::numeric_limits<double>::max();

                            for (size_t i = 0; i < detected_faces.size(); ++i) {
                                const auto& face = detected_faces[i];
                                double cx = face.bbox.x + face.bbox.width / 2.0;
                                double cy = face.bbox.y + face.bbox.height / 2.0;
                                double dx = cx - _poi_x;
                                double dy = cy - _poi_y;
                                double dist = std::sqrt(dx * dx + dy * dy);

                                if (dist <= _poi_dist) {
                                    if (dist < min_dist) {
                                        min_dist = dist;
                                        best_face_idx = static_cast<int>(i);
                                    }
                                }
                            }

                            if (best_face_idx >= 0) {
                                detected_faces = { detected_faces[best_face_idx] };
                            } else {
                                detected_faces.clear();
                            }
                        } else if (_max_faces > 0 && (int)detected_faces.size() > _max_faces) {
                            detected_faces.resize(_max_faces);
                        }
                    }
                    double det_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_det_start).count();

                    std::vector<cv::Rect> bboxes;
                    bboxes.reserve(detected_faces.size());
                    for (const auto& f : detected_faces) {
                        bboxes.push_back(f.bbox);
                    }

                    /* 2. Run DAD-3DHeads E2E Face Analysis (End-to-End FLAME 3DMM + 68/191 Landmarks + 3D Pose) */
                    double fa_ms = 0.0;
                    std::vector<face_analysis::FaceAnalysisResult> face_results;
                    if (_use_face_analysis_e2e && _face_analyzer_e2e && !detected_faces.empty()) {
                        auto t_fa_start = std::chrono::high_resolution_clock::now();
                        for (const auto& f : detected_faces) {
                            auto res = _face_analyzer_e2e->process(image, f.bbox, f.score);
                            if (res.valid) {
                                face_results.push_back(res);
                            }
                        }
                        fa_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_fa_start).count();
                        if (!face_results.empty()) {
                            face_res = face_results[0];
                            last_pose = face_res.pose;
                            has_pose = face_res.pose.success;
                        }
                    }

                    /* 2.5 Run Blink Detection (BlinkLinMulT) */
                    double blink_ms = 0.0;
                    blink_analysis::DetectionResult blink_res;
                    if (_use_blink_detection && _blink_analyzer && !bboxes.empty()) {
                        auto t_blink_start = std::chrono::high_resolution_clock::now();
                        // Build head pose euler array from face analysis result
                        std::array<float, 3> euler = {0.0f, 0.0f, 0.0f};
                        float ear = 0.0f;
                        if (has_pose) {
                            euler[0] = static_cast<float>(last_pose.euler[0]); // pitch
                            euler[1] = static_cast<float>(last_pose.euler[1]); // yaw
                            euler[2] = static_cast<float>(last_pose.euler[2]); // roll
                        }
                        int64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        blink_res = _blink_analyzer->process(image, bboxes[0], euler, ear, ts);
                        blink_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_blink_start).count();
                    }

                    /* 3. Run Body Pose Estimation */
                    double pose_ms = 0.0;
                    std::vector<body_pose::PoseResult> poses;
                    if (_use_body_pose && _body_pose_estimator && !bboxes.empty()) {
                        auto t_pose_start = std::chrono::high_resolution_clock::now();
                        std::vector<body_pose::PoseResult> all_poses = _body_pose_estimator->process(image, 0.5f, 0.45f);

                        // Match face bbox with body pose using nose keypoint (index 0)
                        int selected_bbox_idx = -1;
                        int selected_pose_idx = -1;
                        int max_area = -1;

                        for (int bi = 0; bi < (int)bboxes.size(); ++bi) {
                            const cv::Rect& bbox = bboxes[bi];
                            int area = bbox.width * bbox.height;
                            for (int pi = 0; pi < (int)all_poses.size(); ++pi) {
                                const auto& pose = all_poses[pi];
                                if (!pose.keypoints.empty()) {
                                    float nose_x = pose.keypoints[0].x;
                                    float nose_y = pose.keypoints[0].y;
                                    if (bbox.contains(cv::Point2f(nose_x, nose_y)) && area > max_area) {
                                        max_area = area;
                                        selected_bbox_idx = bi;
                                        selected_pose_idx = pi;
                                    }
                                }
                            }
                        }

                        if (selected_bbox_idx >= 0) {
                            bboxes = { bboxes[selected_bbox_idx] };
                            poses  = { all_poses[selected_pose_idx] };
                        } else if (!all_poses.empty()) {
                            poses = { all_poses[0] };
                        }
                        pose_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_pose_start).count();
                    }

                    /* 4. Prepare Output Mat for Visualization */
                    cv::Mat out_image;
                    if (_has_target_resolution && (_target_width != image.cols || _target_height != image.rows)) {
                        cv::resize(image, out_image, cv::Size(_target_width, _target_height), 0, 0, cv::INTER_LINEAR);
                    } else {
                        out_image = image.clone();
                    }

                    float scale_x = static_cast<float>(out_image.cols) / static_cast<float>(image.cols);
                    float scale_y = static_cast<float>(out_image.rows) / static_cast<float>(image.rows);

                    // Visualize POI (Point of Interest) & distance threshold circle
                    if (_use_poi && _poi_visualize) {
                        int spoi_x = static_cast<int>(_poi_x * scale_x);
                        int spoi_y = static_cast<int>(_poi_y * scale_y);
                        // int spoi_r = static_cast<int>(_poi_dist * ((scale_x + scale_y) * 0.5f));

                        // Draw POI distance threshold circle
                        //cv::circle(out_image, cv::Point(spoi_x, spoi_y), spoi_r, cv::Scalar(0, 165, 255), 1, cv::LINE_AA);

                        // Draw POI center crosshair marker
                        cv::drawMarker(out_image, cv::Point(spoi_x, spoi_y), cv::Scalar(0, 165, 255), cv::MARKER_CROSS, 16, 1, cv::LINE_AA);
                        // cv::circle(out_image, cv::Point(spoi_x, spoi_y), 4, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
                    }

                    // Visualize DAD-3DHeads E2E Results (1:1 Box with Score, 191 Landmarks, 3D Pose Axis, Info Panel)
                    if (_use_face_analysis_e2e && _vis_face_analysis_e2e && _face_analyzer_e2e && !face_results.empty()) {
                        for (const auto& res : face_results) {
                            if (!res.valid) continue;
                            if (scale_x != 1.0f || scale_y != 1.0f) {
                                face_analysis::FaceAnalysisResult scaled_res = res;
                                scaled_res.square_bbox.x = static_cast<int>(scaled_res.square_bbox.x * scale_x);
                                scaled_res.square_bbox.y = static_cast<int>(scaled_res.square_bbox.y * scale_y);
                                scaled_res.square_bbox.width = static_cast<int>(scaled_res.square_bbox.width * scale_x);
                                scaled_res.square_bbox.height = static_cast<int>(scaled_res.square_bbox.height * scale_y);
                                scaled_res.center.x *= scale_x;
                                scaled_res.center.y *= scale_y;
                                scaled_res.scale_size = static_cast<float>(std::min(scaled_res.square_bbox.width, scaled_res.square_bbox.height));
                                scaled_res.pose.nose_tip_2d.x *= scale_x;
                                scaled_res.pose.nose_tip_2d.y *= scale_y;

                                for (auto& pt : scaled_res.landmarks_68) { pt.x *= scale_x; pt.y *= scale_y; }
                                for (auto& pt : scaled_res.landmarks_191) { pt.x *= scale_x; pt.y *= scale_y; }
                                for (auto& pt : scaled_res.projected_vertices) { pt.x *= scale_x; pt.y *= scale_y; }

                                _face_analyzer_e2e->drawResult(out_image, scaled_res, _vis_landmarks_68, _vis_landmarks_191, _vis_head_pose, _vis_square_box, _vis_head_mesh);
                            } else {
                                _face_analyzer_e2e->drawResult(out_image, res, _vis_landmarks_68, _vis_landmarks_191, _vis_head_pose, _vis_square_box, _vis_head_mesh);
                            }
                        }

                        // Draw Info Panel at top-left matching demo_e2e_test.py
                        if (_vis_head_pose) {
                            face_analysis_e2e::drawInfoPanel(out_image, face_results[0].pose, static_cast<int>(face_results.size()));
                        }
                    } else if (_use_face_det && _vis_face_det && !bboxes.empty()) {
                        for (const auto& box : bboxes) {
                            cv::Rect scaled_box(
                                static_cast<int>(box.x * scale_x),
                                static_cast<int>(box.y * scale_y),
                                static_cast<int>(box.width * scale_x),
                                static_cast<int>(box.height * scale_y)
                            );
                            cv::rectangle(out_image, scaled_box, cv::Scalar(0, 255, 128), 2);
                        }
                    }

                    // Visualize Body Pose
                    if (_use_body_pose && _vis_body_pose && !poses.empty()) {
                        static const std::vector<std::pair<int, int>> skeleton_pairs = {
                            {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
                            {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15},
                            {12, 14}, {14, 16}
                        };

                        for (const auto& pose : poses) {
                            for (const auto& pair : skeleton_pairs) {
                                if (pair.first < (int)pose.keypoints.size() && pair.second < (int)pose.keypoints.size()) {
                                    const auto& kp1 = pose.keypoints[pair.first];
                                    const auto& kp2 = pose.keypoints[pair.second];
                                    if (kp1.confidence > 0.5f && kp2.confidence > 0.5f) {
                                        cv::line(out_image, 
                                                 cv::Point(static_cast<int>(kp1.x * scale_x), static_cast<int>(kp1.y * scale_y)), 
                                                 cv::Point(static_cast<int>(kp2.x * scale_x), static_cast<int>(kp2.y * scale_y)), 
                                                 cv::Scalar(0, 255, 255), 2);
                                    }
                                }
                            }

                            for (size_t k = 5; k < pose.keypoints.size(); ++k) {
                                const auto& kpt = pose.keypoints[k];
                                if (kpt.confidence > 0.5f) {
                                    cv::circle(out_image, 
                                               cv::Point(static_cast<int>(kpt.x * scale_x), static_cast<int>(kpt.y * scale_y)), 
                                               4, cv::Scalar(0, 0, 255), -1);
                                }
                            }
                        }
                    }

                    // Visualize Blink Detection Results
                    if (_use_blink_detection && _vis_blink_detection && _blink_analyzer && !bboxes.empty()) {
                        cv::Rect scaled_face(
                            static_cast<int>(bboxes[0].x * scale_x),
                            static_cast<int>(bboxes[0].y * scale_y),
                            static_cast<int>(bboxes[0].width * scale_x),
                            static_cast<int>(bboxes[0].height * scale_y)
                        );
                        _blink_analyzer->drawResult(out_image, scaled_face, blink_res);
                    }

                    /* 5. Run Driver Readiness Estimation (Torch-based, if enabled) */
                    driver_readiness::ReadinessResult readiness_res;
                    if (_use_driver_readiness && _driver_readiness_estimator) {
                        readiness_res = _driver_readiness_estimator->process(poses, last_pose, has_pose, out_image.cols);
                        if (readiness_res.is_ready) {
                            std::lock_guard<std::mutex> lock(_history_mutex);
                            _readiness_history.push_back({std::chrono::steady_clock::now(), static_cast<double>(readiness_res.confidence)});
                        }
                    }

                    /* 6. Run Driver Readiness Estimation (Rule-based Logical, if enabled) */
                    driver_readiness_logical::LogicalReadinessResult logical_res;
                    if (_use_driver_readiness_logical && _driver_readiness_logical_estimator) {
                        logical_res = _driver_readiness_logical_estimator->process(last_pose, has_pose);
                        if (logical_res.valid) {
                            std::lock_guard<std::mutex> lock(_history_mutex);
                            _readiness_history.push_back({std::chrono::steady_clock::now(), logical_res.readiness_score});
                        }
                    }

                    // Render readiness score changes graph
                    if ((_use_driver_readiness && _vis_driver_readiness) || (_use_driver_readiness_logical && _vis_driver_readiness_logical)) {
                        int graph_w = 400;
                        int graph_h = 65;
                        int graph_x = out_image.cols - graph_w - 10;
                        int graph_y = out_image.rows - graph_h - 10;
                        draw_readiness_graph(out_image, graph_x, graph_y, graph_w, graph_h);
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

                        cv::putText(out_image, datetime_str, cv::Point(std::max(10, out_image.cols - 270), 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        cv::putText(out_image, fps_str, cv::Point(out_image.cols - 60, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }

                    /* 7. Encode as JPEG */
                    std::vector<uchar> jpeg_buf;
                    if (cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        
                        /* 8. Construct metadata tags */
                        json tag;
                        tag["width"] = out_image.cols;
                        tag["height"] = out_image.rows;
                        tag["type"] = out_image.type();
                        tag["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                        tag["cam_channel"] = 1;
                        tag["fps"] = fps;

                        if (_use_driver_readiness && _driver_readiness_estimator && readiness_res.is_ready) {
                            tag["dms_dl_class"] = readiness_res.predicted_class;
                            tag["dms_dl_confidence"] = readiness_res.confidence;
                            tag["dms_dl_attention_score"] = readiness_res.confidence;
                            tag["dms_dl_category"] = readiness_res.category;
                        }
                        if (_use_driver_readiness_logical && _driver_readiness_logical_estimator) {
                            tag["dms_logical_readiness"] = logical_res.readiness_score;
                            tag["dms_logical_category"] = logical_res.category;
                        }
                        if (_use_blink_detection && _blink_analyzer) {
                            tag["blink_prob"] = blink_res.blink_prob;
                            tag["blink_is_blinking"] = blink_res.is_blinking;
                            tag["blink_count"] = blink_res.blink_count;
                            tag["blink_perclos"] = blink_res.perclos;
                        }

                        /* 9. Send multipart message */
                        flame::component::ZData out_msg;
                        out_msg.from = "image_stream_1_processed_monitor";
                        out_msg.meta = tag.dump();
                        out_msg.addmem(jpeg_buf.data(), jpeg_buf.size());

                        if (!dispatch("image_stream_1_processed_monitor", out_msg)) {
                            logger::warn("[{}] Failed to dispatch processed image 1", getName());
                        }
                    }

                    // Calculate total processing time for this frame
                    double total_frame_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_frame_start).count();

                    if (!bboxes.empty() && face_res.valid) {
                        logger::info("[{}] [Frame #{}] Total: {:.1f}ms ({:.1f} FPS) | Det: {:.1f}ms ({} faces) | E2E: {:.1f}ms [P:{:.1f}, Y:{:.1f}, R:{:.1f}] | Blink: {:.1f}ms [prob:{:.2f}, blinks:{}, perclos:{:.2f}] | Pose: {:.1f}ms",
                                     getName(), frame_count, total_frame_ms, fps,
                                     det_ms, bboxes.size(),
                                     fa_ms, last_pose.euler[0], last_pose.euler[1], last_pose.euler[2],
                                     blink_ms, blink_res.blink_prob, blink_res.blink_count, blink_res.perclos,
                                     pose_ms);
                    } else if (!bboxes.empty()) {
                        logger::info("[{}] [Frame #{}] Total: {:.1f}ms ({:.1f} FPS) | Det: {:.1f}ms ({} faces) | E2E: {:.1f}ms (invalid) | Blink: {:.1f}ms | Pose: {:.1f}ms",
                                     getName(), frame_count, total_frame_ms, fps,
                                     det_ms, bboxes.size(), fa_ms, blink_ms, pose_ms);
                    } else {
                        logger::info("[{}] [Frame #{}] Total: {:.1f}ms ({:.1f} FPS) | Det: {:.1f}ms (0 faces, skipped)",
                                     getName(), frame_count, total_frame_ms, fps, det_ms);
                    }
                }
                catch (const std::exception& e) {
                    logger::error("[{}] Error in inference worker loop: {}", getName(), e.what());
                }
            } else {
                auto now_idle = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now_idle - last_idle_warning).count() >= 3000) {
                    logger::info("[{}] Waiting for input frames on image_stream_1... (Processed {} frames so far)", getName(), frame_count);
                    last_idle_warning = now_idle;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    _worker_finished.store(true);
}
