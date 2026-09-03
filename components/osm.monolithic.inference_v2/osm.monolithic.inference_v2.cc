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

        if (parameters.contains("face_detection")) {
            const auto& fd_params = parameters["face_detection"];
            _use_face_det = fd_params.value("use", _use_face_det);
            face_det_model_path = fd_params.value("model_path", face_det_model_path);
            face_det_gpu_id = fd_params.value("gpu_id", face_det_gpu_id);
            _nms_threshold = fd_params.value("nms", _nms_threshold);
            _vis_face_det = fd_params.value("visualize", true);
            _use_roi = fd_params.value("use_roi", false);
            _roi_visualize = fd_params.value("roi_visualize", true);
            if (fd_params.contains("roi") && fd_params["roi"].is_array()) {
                if (fd_params["roi"].size() == 4) {
                    _roi_x1 = fd_params["roi"][0].get<int>();
                    _roi_y1 = fd_params["roi"][1].get<int>();
                    _roi_x2 = fd_params["roi"][2].get<int>();
                    _roi_y2 = fd_params["roi"][3].get<int>();
                }
            }
            if (fd_params.contains("padding") && fd_params["padding"].is_array() && fd_params["padding"].size() == 2) {
                _padding_w = fd_params["padding"][0].get<float>();
                _padding_h = fd_params["padding"][1].get<float>();
                logger::info("[{}] Loaded face detection padding: w={}, h={}", getName(), _padding_w, _padding_h);
            }
        }

        std::string face_analysis_model_path = "/home/iae-vc/dev/flame_osm/bin/x86_64/models/dad_3dheads_e2e.torchscript";
        int face_analysis_gpu_id = 0;
        if (parameters.contains("face_analysis_e2e")) {
            const auto& fa_params = parameters["face_analysis_e2e"];
            _use_face_analysis_e2e = fa_params.value("use", _use_face_analysis_e2e);
            face_analysis_model_path = fa_params.value("model_path", face_analysis_model_path);
            face_analysis_gpu_id = fa_params.value("gpu_id", face_analysis_gpu_id);
            _vis_face_analysis_e2e = fa_params.value("visualize", true);
            _vis_landmarks_68 = fa_params.value("vis_landmarks_68", true);
            _vis_landmarks_191 = fa_params.value("vis_landmarks_191", false);
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

        /* Start background worker thread */
        _worker_stop.store(false);
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
    _worker_stop.store(true);
    if (_inference_worker.joinable()) {
        _inference_worker.join();
    }
    logger::info("[{}] Monolithic inference V2 worker thread stopped", getName());
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

    while (!_worker_stop.load()) {
        if (_enable_stream_1) {
            cv::Mat image = getLatestImage1();
            if (!image.empty()) {
                // Clear cache
                {
                    std::lock_guard<std::mutex> lock(_img_mutex_1);
                    _latest_image_1.release();
                }

                try {
                    head_pose::PoseResult last_pose;
                    bool has_pose = false;
                    face_analysis::FaceAnalysisResult face_res;

                    /* 1. Run YOLO11-Face detection */
                    std::vector<cv::Rect> bboxes;
                    if (_use_face_det && _face_detector) {
                        bboxes = _face_detector->process(image, _nms_threshold, _padding_w, _padding_h);
                        if (_use_roi) {
                            std::vector<cv::Rect> filtered_bboxes;
                            for (const auto& box : bboxes) {
                                int cx = box.x + box.width / 2;
                                int cy = box.y + box.height / 2;
                                if (cx >= _roi_x1 && cx <= _roi_x2 && cy >= _roi_y1 && cy <= _roi_y2) {
                                    filtered_bboxes.push_back(box);
                                }
                            }
                            bboxes = filtered_bboxes;
                        }
                    }

                    /* 2. Run DAD-3DHeads E2E Face Analysis (End-to-End FLAME 3DMM + 68/191 Landmarks + 3D Pose) */
                    if (_use_face_analysis_e2e && _face_analyzer_e2e && !bboxes.empty()) {
                        face_res = _face_analyzer_e2e->process(image, bboxes[0]);
                        if (face_res.valid) {
                            last_pose = face_res.pose;
                            has_pose = face_res.pose.success;
                        }
                    }

                    /* 3. Run Body Pose Estimation */
                    std::vector<body_pose::PoseResult> poses;
                    if (_use_body_pose && _body_pose_estimator && !bboxes.empty()) {
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

                    // Visualize Face BBox
                    if (_use_face_det && _vis_face_det && !bboxes.empty()) {
                        for (const auto& box : bboxes) {
                            cv::Rect scaled_box(
                                static_cast<int>(box.x * scale_x),
                                static_cast<int>(box.y * scale_y),
                                static_cast<int>(box.width * scale_x),
                                static_cast<int>(box.height * scale_y)
                            );
                            cv::rectangle(out_image, scaled_box, cv::Scalar(0, 255, 0), 2);
                        }
                    }

                    // Visualize ROI
                    if (_use_roi && _roi_visualize) {
                        cv::Rect scaled_roi(
                            static_cast<int>(_roi_x1 * scale_x),
                            static_cast<int>(_roi_y1 * scale_y),
                            static_cast<int>((_roi_x2 - _roi_x1) * scale_x),
                            static_cast<int>((_roi_y2 - _roi_y1) * scale_y)
                        );
                        cv::rectangle(out_image, scaled_roi, cv::Scalar(0, 165, 255), 2);
                    }

                    // Visualize DAD-3DHeads E2E Results (68/191 Landmarks, 3D Pose, 1:1 Square Box)
                    if (_use_face_analysis_e2e && _vis_face_analysis_e2e && _face_analyzer_e2e && face_res.valid) {
                        if (scale_x != 1.0f || scale_y != 1.0f) {
                            face_analysis::FaceAnalysisResult scaled_res = face_res;
                            scaled_res.square_bbox.x = static_cast<int>(scaled_res.square_bbox.x * scale_x);
                            scaled_res.square_bbox.y = static_cast<int>(scaled_res.square_bbox.y * scale_y);
                            scaled_res.square_bbox.width = static_cast<int>(scaled_res.square_bbox.width * scale_x);
                            scaled_res.square_bbox.height = static_cast<int>(scaled_res.square_bbox.height * scale_y);
                            scaled_res.pose.nose_tip_2d.x *= scale_x;
                            scaled_res.pose.nose_tip_2d.y *= scale_y;

                            for (auto& pt : scaled_res.landmarks_68) { pt.x *= scale_x; pt.y *= scale_y; }
                            for (auto& pt : scaled_res.landmarks_191) { pt.x *= scale_x; pt.y *= scale_y; }
                            for (auto& pt : scaled_res.projected_vertices) { pt.x *= scale_x; pt.y *= scale_y; }

                            _face_analyzer_e2e->drawResult(out_image, scaled_res, _vis_landmarks_68, _vis_landmarks_191, _vis_head_pose, _vis_square_box, _vis_head_mesh);
                        } else {
                            _face_analyzer_e2e->drawResult(out_image, face_res, _vis_landmarks_68, _vis_landmarks_191, _vis_head_pose, _vis_square_box, _vis_head_mesh);
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

                    // Draw Head Pose text box at bottom-left corner
                    if (has_pose && _vis_head_pose) {
                        double pitch = last_pose.euler[0];
                        double yaw = last_pose.euler[1];
                        double roll = last_pose.euler[2];

                        char txt_pitch[64], txt_yaw[64], txt_roll[64];
                        snprintf(txt_pitch, sizeof(txt_pitch), "Pitch : %.1f", pitch);
                        snprintf(txt_yaw, sizeof(txt_yaw), "Yaw   : %.1f", yaw);
                        snprintf(txt_roll, sizeof(txt_roll), "Roll  : %.1f", roll);

                        int font_face = cv::FONT_HERSHEY_SIMPLEX;
                        double font_scale = 0.5;
                        int thickness = 1;
                        int baseline = 0;

                        cv::Size s1 = cv::getTextSize(txt_pitch, font_face, font_scale, thickness, &baseline);
                        cv::Size s2 = cv::getTextSize(txt_yaw, font_face, font_scale, thickness, &baseline);
                        cv::Size s3 = cv::getTextSize(txt_roll, font_face, font_scale, thickness, &baseline);
                        int max_w = std::max({s1.width, s2.width, s3.width});

                        int box_w = max_w + 20;
                        int box_h = 65;
                        int start_x = 10;
                        int start_y = out_image.rows - box_h - 10;

                        cv::Rect bg_box(start_x, start_y, box_w, box_h);
                        cv::Mat overlay;
                        out_image.copyTo(overlay);
                        cv::rectangle(overlay, bg_box, cv::Scalar(0, 0, 0), cv::FILLED);
                        cv::addWeighted(overlay, 0.5, out_image, 0.5, 0, out_image);
                        cv::rectangle(out_image, bg_box, cv::Scalar(255, 255, 255), 1);

                        cv::putText(out_image, txt_pitch, cv::Point(start_x + 10, start_y + 18), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
                        cv::putText(out_image, txt_yaw,   cv::Point(start_x + 10, start_y + 38), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
                        cv::putText(out_image, txt_roll,  cv::Point(start_x + 10, start_y + 58), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
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

                        cv::putText(out_image, datetime_str, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        cv::putText(out_image, fps_str, cv::Point(out_image.cols - 60, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
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

                        /* 9. Send multipart message */
                        flame::component::ZData out_msg;
                        out_msg.from = "image_stream_1_processed_monitor";
                        out_msg.meta = tag.dump();
                        out_msg.addmem(jpeg_buf.data(), jpeg_buf.size());

                        if (!dispatch("image_stream_1_processed_monitor", out_msg)) {
                            logger::warn("[{}] Failed to dispatch processed image 1", getName());
                        }
                    }
                }
                catch (const std::exception& e) {
                    logger::error("[{}] Error in inference worker loop: {}", getName(), e.what());
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
