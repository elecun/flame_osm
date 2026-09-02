
#include "video.file.grabber.hpp"
#include <flame/log.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cmath>

using namespace flame;
using namespace std;
using namespace cv;

/* create component instance */
static video_file_grabber* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new video_file_grabber(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

bool video_file_grabber::onInit(){
    try {
        _worker_stop.store(false);

        /* read profile */
        json parameters = getProfile()->parameters();

        /* load video files */
        _video_files.clear();
        if (parameters.contains("video_files") && parameters["video_files"].is_array()) {
            for (const auto& item : parameters["video_files"]) {
                if (item.is_string()) {
                    string p = item.get<string>();
                    if (!p.empty()) {
                        _video_files.push_back(p);
                    }
                }
            }
        } else if (parameters.contains("camera") && parameters["camera"].is_array()) {
            for (const auto& cam : parameters["camera"]) {
                if (cam.contains("file") && cam["file"].is_string()) {
                    _video_files.push_back(cam["file"].get<string>());
                }
            }
        }

        if (_video_files.empty()) {
            logger::error("[{}] No video files specified in 'video_files' parameter", getName());
            return false;
        }

        logger::info("[{}] {} video file(s) defined in parameters.", getName(), _video_files.size());
        for (size_t i = 0; i < _video_files.size(); ++i) {
            logger::info("[{}]   Video [{}]: {}", getName(), i + 1, _video_files[i]);
        }

        /* setup pipeline flags */
        _use_image_stream.store(parameters.value("use_image_stream", true));
        _fault_reset.store(parameters.value("fault_reset", false));
        _fault_limit.store(parameters.value("fault_limit", 3));

        /* undistortion setup */
        _enable_undistort = parameters.value("undistort", false);
        if (_enable_undistort && parameters.contains("camera_calibration")) {
            json calib_json;
            std::string calib_file_path;

            if (parameters["camera_calibration"].is_string()) {
                calib_file_path = parameters["camera_calibration"].get<std::string>();
                if (std::filesystem::exists(calib_file_path)) {
                    try {
                        std::ifstream f(calib_file_path);
                        calib_json = json::parse(f);
                    } catch (const std::exception& e) {
                        logger::error("[{}] Failed to parse calibration JSON file '{}': {}", getName(), calib_file_path, e.what());
                    }
                } else {
                    logger::warn("[{}] Calibration file not found: '{}'", getName(), calib_file_path);
                }
            } else if (parameters["camera_calibration"].is_object()) {
                calib_json = parameters["camera_calibration"];
            }

            if (!calib_json.empty()) {
                if (calib_json.contains("camera_matrix") && calib_json["camera_matrix"].is_array()) {
                    _camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                    const auto& cm = calib_json["camera_matrix"];
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            _camera_matrix.at<double>(r, c) = cm[r][c].get<double>();
                        }
                    }
                } else if (calib_json.contains("intrinsic") && calib_json["intrinsic"].is_array()) {
                    _camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                    const auto& intr = calib_json["intrinsic"];
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            _camera_matrix.at<double>(r, c) = intr[r][c].get<double>();
                        }
                    }
                }

                if (calib_json.contains("distortion_coefficients") && calib_json["distortion_coefficients"].is_array()) {
                    const auto& dist = calib_json["distortion_coefficients"];
                    int dist_size = dist.size();
                    _dist_coeffs = cv::Mat(1, dist_size, CV_64F);
                    for (int i = 0; i < dist_size; ++i) {
                        _dist_coeffs.at<double>(0, i) = dist[i].get<double>();
                    }
                } else if (calib_json.contains("distortion") && calib_json["distortion"].is_array()) {
                    const auto& dist = calib_json["distortion"];
                    int dist_size = dist.size();
                    _dist_coeffs = cv::Mat(1, dist_size, CV_64F);
                    for (int i = 0; i < dist_size; ++i) {
                        _dist_coeffs.at<double>(0, i) = dist[i].get<double>();
                    }
                }

                int img_w = 1920, img_h = 1080;
                if (calib_json.contains("image_size")) {
                    img_w = calib_json["image_size"].value("width", img_w);
                    img_h = calib_json["image_size"].value("height", img_h);
                }

                if (!_camera_matrix.empty() && !_dist_coeffs.empty()) {
                    cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
                        _camera_matrix, _dist_coeffs, cv::Size(img_w, img_h), 0.0, cv::Size(img_w, img_h)
                    );
                    cv::initUndistortRectifyMap(
                        _camera_matrix, _dist_coeffs, cv::Mat(),
                        new_camera_matrix, cv::Size(img_w, img_h),
                        CV_32FC1, _map1, _map2
                    );
                    logger::info("[{}] Loaded camera calibration parameters successfully. Resolution: {}x{}", getName(), img_w, img_h);
                }
            } else {
                _enable_undistort = false;
            }
        }

        /* read profile dataport */
        json dataport_cfg = getProfile()->dataPort();
        _output_ports = {"image_stream_1", "image_stream_2"};

        for (const auto& portname : _output_ports) {
            if (dataport_cfg.contains(portname)) {
                const auto& port_cfg = dataport_cfg[portname];
                if (port_cfg.contains("rotation")) {
                    _port_rotations[portname] = port_cfg["rotation"].get<string>();
                    logger::info("[{}] Port '{}' rotation configured: {}", getName(), portname, _port_rotations[portname]);
                }
            }
            _last_capture_times[portname] = chrono::high_resolution_clock::now();
            _dispatch_workers[portname] = thread(&video_file_grabber::_dispatch_task, this, portname);
        }

        /* start grabbing worker */
        _grab_worker = thread(&video_file_grabber::_grab_task, this);

    } catch (const std::exception& e) {
        logger::error("[{}] Exception in onInit: {}", getName(), e.what());
        return false;
    }

    return true;
}

void video_file_grabber::onLoop(){
    /* nothing loop */
}

void video_file_grabber::onClose(){
    /* stop worker */
    _worker_stop.store(true);

    /* notify all dispatch workers */
    for (auto& [port, cv] : _queue_cvs) {
        cv.notify_all();
    }

    /* stop grabbing thread */
    if (_grab_worker.joinable()) {
        _grab_worker.join();
        logger::debug("[{}] grab worker stopped", getName());
    }

    /* stop dispatch workers */
    for (auto& [port, worker] : _dispatch_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    _dispatch_workers.clear();
    logger::debug("[{}] all dispatch workers stopped", getName());

    /* close video capture */
    if (_video_capture && _video_capture->isOpened()) {
        _video_capture->release();
    }
}

void video_file_grabber::onData(flame::component::ZData& data){
    /* reserved function */
}

void video_file_grabber::_grab_task(){
    _current_video_idx = 0;

    auto open_video_file = [this](size_t idx) -> bool {
        if (_video_capture && _video_capture->isOpened()) {
            _video_capture->release();
        }
        if (idx >= _video_files.size()) return false;

        const string& file_path = _video_files[idx];

        if (!std::filesystem::exists(file_path)) {
            logger::error("[{}] Video file not found at path: '{}'", getName(), file_path);
            return false;
        }

        _video_capture = make_unique<cv::VideoCapture>(file_path);
        if (!_video_capture->isOpened()) {
            logger::error("[{}] Failed to open video file at path: '{}'", getName(), file_path);
            return false;
        }

        _video_fps = _video_capture->get(cv::CAP_PROP_FPS);
        if (_video_fps <= 0.0 || std::isnan(_video_fps) || std::isinf(_video_fps)) {
            _video_fps = 30.0;
        }
        _total_frames = static_cast<int>(_video_capture->get(cv::CAP_PROP_FRAME_COUNT));
        _frame_width = static_cast<int>(_video_capture->get(cv::CAP_PROP_FRAME_WIDTH));
        _frame_height = static_cast<int>(_video_capture->get(cv::CAP_PROP_FRAME_HEIGHT));

        logger::info("[{}] Loaded video [{}/{}]: '{}' ({}x{}, FPS: {:.2f}, Total frames: {})",
                     getName(), idx + 1, _video_files.size(), file_path,
                     _frame_width, _frame_height, _video_fps, _total_frames);
        return true;
    };

    unsigned long frame_count = 0;

    while (!_worker_stop.load()) {
        // If capture is not currently open, try to open the current video file
        if (!_video_capture || !_video_capture->isOpened()) {
            if (!open_video_file(_current_video_idx)) {
                _capture_fault.store(true);
                int current_faults = ++_fault_count;
                logger::warn("[{}] Capture fault: Video file unavailable. (Current faults: {})", getName(), current_faults);

                if (_use_image_stream.load()) {
                    for (const auto& portname : _output_ports) {
                        int cam_channel = (portname == "image_stream_1" ? 1 : 2);
                        auto msg = make_shared<flame::component::ZData>();
                        json tag;
                        auto now = chrono::high_resolution_clock::now();
                        tag["fps"] = 0.0;
                        tag["height"] = 0;
                        tag["width"] = 0;
                        tag["type"] = 0;
                        tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
                        tag["cam_channel"] = cam_channel;
                        tag["capture_fault"] = true;

                        msg->from = portname;
                        msg->meta = tag.dump();
                        msg->addmem(nullptr, 0);

                        {
                            lock_guard<mutex> lock(_queue_mtxs[portname]);
                            if (_dispatch_queues[portname].size() < _max_queue_size) {
                                _dispatch_queues[portname].push(msg);
                            } else {
                                _dispatch_queues[portname].pop();
                                _dispatch_queues[portname].push(msg);
                            }
                        }
                        _queue_cvs[portname].notify_one();
                    }
                }

                this_thread::sleep_for(chrono::seconds(1));
                continue;
            }
        }

        auto frame_start = chrono::high_resolution_clock::now();
        double current_fps = (_video_fps > 0.0) ? _video_fps : 30.0;
        auto frame_duration = chrono::microseconds(static_cast<int64_t>(1000000.0 / current_fps));

        try {
            cv::Mat captured;
            bool success = false;

            if (_video_capture && _video_capture->isOpened()) {
                success = _video_capture->read(captured);
            }

            if (!success || captured.empty()) {
                logger::info("[{}] End of video file reached: '{}'", getName(), _video_files[_current_video_idx]);
                _current_video_idx = (_current_video_idx + 1) % _video_files.size();
                logger::info("[{}] Switching to next video [{}/{}]: '{}'", getName(), _current_video_idx + 1, _video_files.size(), _video_files[_current_video_idx]);

                if (_video_capture && _video_capture->isOpened()) {
                    _video_capture->release();
                }

                if (!open_video_file(_current_video_idx)) {
                    _capture_fault.store(true);
                    this_thread::sleep_for(chrono::seconds(1));
                    continue;
                }

                // Update frame duration for the new video
                current_fps = (_video_fps > 0.0) ? _video_fps : 30.0;
                frame_duration = chrono::microseconds(static_cast<int64_t>(1000000.0 / current_fps));

                // Read first frame of next video
                success = _video_capture->read(captured);
                if (!success || captured.empty()) {
                    this_thread::sleep_for(chrono::milliseconds(100));
                    continue;
                }
            }

            if (!captured.empty()) {
                _capture_fault.store(false);

                if (_use_image_stream.load()) {
                    for (const auto& portname : _output_ports) {
                        cv::Mat frame_out = captured.clone();
                        int cam_channel = (portname == "image_stream_1" ? 1 : 2);

                        // 1. Rotate if configured for this port
                        if (_port_rotations.find(portname) != _port_rotations.end()) {
                            cv::Mat rotated;
                            string rot_type = _port_rotations[portname];
                            if (rot_type == "ccw") {
                                cv::rotate(frame_out, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
                                frame_out = rotated;
                            } else if (rot_type == "cw") {
                                cv::rotate(frame_out, rotated, cv::ROTATE_90_CLOCKWISE);
                                frame_out = rotated;
                            } else if (rot_type == "180") {
                                cv::rotate(frame_out, rotated, cv::ROTATE_180);
                                frame_out = rotated;
                            }
                        }

                        // 2. Undistort if enabled
                        if (_enable_undistort && !_map1.empty() && !_map2.empty()) {
                            cv::Mat undistorted;
                            cv::remap(frame_out, undistorted, _map1, _map2, cv::INTER_LINEAR);
                            frame_out = undistorted;
                        }

                        // 3. Construct message
                        auto msg = make_shared<flame::component::ZData>();
                        json tag;
                        auto now = chrono::high_resolution_clock::now();
                        chrono::duration<double> elapsed = now - _last_capture_times[portname];
                        _last_capture_times[portname] = now;

                        tag["fps"] = (elapsed.count() > 0) ? 1.0 / elapsed.count() : current_fps;
                        tag["height"] = frame_out.rows;
                        tag["width"] = frame_out.cols;
                        tag["type"] = frame_out.type();
                        tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
                        tag["cam_channel"] = cam_channel;
                        tag["capture_fault"] = false;
                        tag["file_index"] = _current_video_idx;

                        msg->from = portname;
                        msg->meta = tag.dump();
                        msg->addmem(frame_out.data, frame_out.total() * frame_out.elemSize());

                        // 4. Push to dispatch queue
                        {
                            lock_guard<mutex> lock(_queue_mtxs[portname]);
                            if (_dispatch_queues[portname].size() < _max_queue_size) {
                                _dispatch_queues[portname].push(msg);
                            } else {
                                _dispatch_queues[portname].pop();
                                _dispatch_queues[portname].push(msg);
                            }
                        }
                        _queue_cvs[portname].notify_one();
                    }
                }

                frame_count++;
            }

        } catch (const cv::Exception& e) {
            logger::debug("[{}] CV Exception: {}", getName(), e.what());
        } catch (const std::exception& e) {
            logger::error("[{}] Exception in grab task: {}", getName(), e.what());
        }

        // Frame rate control
        auto frame_end = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::microseconds>(frame_end - frame_start);
        if (elapsed < frame_duration) {
            this_thread::sleep_for(frame_duration - elapsed);
        }
    }

    logger::debug("[{}] Stopped grab task..", getName());
}

void video_file_grabber::_dispatch_task(string portname) {
    logger::debug("[{}] Started dispatch task for port {}", getName(), portname);

    while (!_worker_stop.load()) {
        try {
            shared_ptr<flame::component::ZData> msg = nullptr;
            {
                unique_lock<mutex> lock(_queue_mtxs[portname]);
                _queue_cvs[portname].wait(lock, [this, &portname] {
                    return !_dispatch_queues[portname].empty() || _worker_stop.load();
                });

                if (_worker_stop.load() && _dispatch_queues[portname].empty()) break;

                if (!_dispatch_queues[portname].empty()) {
                    msg = _dispatch_queues[portname].front();
                    _dispatch_queues[portname].pop();
                }
            }

            if (msg) {
                if (!dispatch(msg->from, *msg)) {
                    logger::warn("[{}] Failed to dispatch image to port {}", getName(), msg->from);
                }
            }
        } catch (const std::exception& e) {
            if (!_worker_stop.load()) {
                logger::error("[{}] Exception in dispatch task for port {}: {}", getName(), portname, e.what());
            }
            break;
        }
    }
    logger::debug("[{}] Stopped dispatch task for port {}", getName(), portname);
}

