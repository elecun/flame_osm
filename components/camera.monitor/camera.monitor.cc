#include "camera.monitor.hpp"

#include <flame/log.hpp>
#include <dep/json.hpp>

using json = nlohmann::json;

/* create component instance */
static camera_monitor* _instance = nullptr;
flame::component::Object* Create()
{
    if (!_instance)
        _instance = new camera_monitor();
    return _instance;
}
void Release()
{
    if (_instance) {
        delete _instance;
        _instance = nullptr;
    }
}

camera_monitor::camera_monitor() {}

bool camera_monitor::onInit()
{
    try {
        _stop_threads.store(false);

        /* Read dataport config to obtain per-monitor-port resolution */
        json dataport_cfg = getProfile()->dataPort();

        auto load_resolution = [&](const string& monitor_portname) {
            MonitorResolution res;
            if (dataport_cfg.contains(monitor_portname)) {
                const auto& port_cfg = dataport_cfg[monitor_portname];
                
                if (port_cfg.contains("resolution")) {
                    const auto& r = port_cfg["resolution"];
                    if (r.contains("width") && r.contains("height")) {
                        res.has_resolution = true;
                        res.width  = r["width"].get<int>();
                        res.height = r["height"].get<int>();
                        logger::info("[{}] Monitor port '{}' target resolution: {}x{}", getName(), monitor_portname, res.width, res.height);
                    }
                }
            }
            _monitor_resolutions[monitor_portname] = res;
        };

        load_resolution("image_stream_1_monitor");
        load_resolution("image_stream_2_monitor");

        // Pre-allocate channels before starting threads
        _channels["image_stream_1"] = make_unique<StreamChannel>();
        _channels["image_stream_2"] = make_unique<StreamChannel>();

        // Start threads for image_stream_1 and image_stream_2
        _monitor_threads["image_stream_1"] = thread(&camera_monitor::_monitor_task, this, "image_stream_1", "image_stream_1_monitor");
        _monitor_threads["image_stream_2"] = thread(&camera_monitor::_monitor_task, this, "image_stream_2", "image_stream_2_monitor");

        logger::info("[{}] Monitoring component initialized with threads.", getName());
    } catch (const std::exception& e) {
        logger::error("[{}] Initialization exception: {}", getName(), e.what());
        return false;
    }
    return true;
}

void camera_monitor::onLoop() {}

void camera_monitor::onClose()
{
    try {
        _stop_threads.store(true);

        // Notify all queue condition variables to wake up threads
        for (auto& [name, ch] : _channels) {
            if (ch) {
                ch->cv.notify_all();
            }
        }

        // Join monitor threads
        for (auto& [name, t] : _monitor_threads) {
            if (t.joinable()) {
                t.join();
                logger::debug("[{}] Joined thread for {}", getName(), name);
            }
        }
        _monitor_threads.clear();

        // Clear all queues
        for (auto& [name, ch] : _channels) {
            if (ch) {
                lock_guard<mutex> lock(ch->mtx);
                queue<shared_ptr<flame::component::ZData>> empty_q;
                swap(ch->q, empty_q);
            }
        }

        logger::info("[{}] Component successfully closed.", getName());
    } catch (const std::exception& e) {
        logger::error("[{}] Error on close: {}", getName(), e.what());
    }
}

void camera_monitor::onData(flame::component::ZData& data)
{
    try {
        const string& portname = data.from;

        auto it = _channels.find(portname);
        if (it != _channels.end() && it->second) {
            auto msg = make_shared<flame::component::ZData>(std::move(data));

            /* push to channel queue */
            {
                lock_guard<mutex> lock(it->second->mtx);
                if (it->second->q.size() < _max_queue_size) {
                    it->second->q.push(msg);
                } else {
                    it->second->q.pop();
                    it->second->q.push(msg);
                }
            }
            it->second->cv.notify_one();
        }
    } catch (const std::exception& e) {
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}

void camera_monitor::_monitor_task(string stream_name, string monitor_portname)
{
    logger::debug("[{}] Started monitor task for {} -> {}", getName(), stream_name, monitor_portname);

    /* Retrieve the target resolution for this monitor port */
    const MonitorResolution& target_res = _monitor_resolutions[monitor_portname];

    /* Retrieve channel pointer */
    StreamChannel* ch = nullptr;
    auto it = _channels.find(stream_name);
    if (it != _channels.end()) {
        ch = it->second.get();
    }
    if (!ch) {
        logger::error("[{}] Stream channel '{}' not found in _channels!", getName(), stream_name);
        return;
    }

    /* JPEG encoding parameters */
    vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};

    while (!_stop_threads.load()) {
        try {
            shared_ptr<flame::component::ZData> msg = nullptr;
            {
                unique_lock<mutex> lock(ch->mtx);
                ch->cv.wait(lock, [this, ch] {
                    return !ch->q.empty() || _stop_threads.load();
                });

                if (_stop_threads.load() && ch->q.empty())
                    break;

                if (!ch->q.empty()) {
                    msg = ch->q.front();
                    ch->q.pop();
                }
            }

            if (msg) {
                auto start_time = chrono::high_resolution_clock::now();

                if (msg->size() >= 2) {
                    zmq::message_t tag_msg = msg->pop();
                    zmq::message_t img_msg = msg->pop();

                    /* ---- Parse the tag JSON to reconstruct cv::Mat ---- */
                    json tag;
                    try {
                        string tag_str(static_cast<char*>(tag_msg.data()), tag_msg.size());
                        tag = json::parse(tag_str);
                    } catch (const std::exception& e) {
                        logger::warn("[{}] Failed to parse tag JSON: {}", getName(), e.what());
                        continue;
                    }

                    int src_height = tag.value("height", 0);
                    int src_width  = tag.value("width", 0);
                    int src_type   = tag.value("type", CV_8UC3);

                    /* ---- Handle capture fault or empty image states ---- */
                    bool is_fault = false;
                    if (tag.contains("capture_fault") && tag["capture_fault"].get<bool>()) {
                        is_fault = true;
                    }
                    if (src_height <= 0 || src_width <= 0 || img_msg.size() == 0 || img_msg.data() == nullptr) {
                        is_fault = true;
                    }

                    cv::Mat src_image;
                    if (!is_fault) {
                        size_t elem_size = CV_ELEM_SIZE(src_type);
                        size_t raw_expected_size = static_cast<size_t>(src_height) * src_width * elem_size;

                        const unsigned char* data_ptr = static_cast<const unsigned char*>(img_msg.data());
                        size_t data_size = img_msg.size();

                        // Check if buffer is JPEG compressed (starts with SOI marker 0xFF 0xD8)
                        if (data_size >= 2 && data_ptr[0] == 0xFF && data_ptr[1] == 0xD8) {
                            cv::Mat raw_buf(1, data_size, CV_8UC1, const_cast<unsigned char*>(data_ptr));
                            src_image = cv::imdecode(raw_buf, cv::IMREAD_COLOR);
                            if (src_image.empty()) {
                                is_fault = true;
                            }
                        } else if (data_size >= raw_expected_size) {
                            // Raw uncompressed cv::Mat buffer
                            src_image = cv::Mat(src_height, src_width, src_type, const_cast<unsigned char*>(data_ptr)).clone();
                        } else {
                            logger::warn("[{}] [{}] Image buffer size mismatch (got: {}, expected at least: {})",
                                         getName(), stream_name, data_size, raw_expected_size);
                            is_fault = true;
                        }
                    }

                    if (is_fault || src_image.empty()) {
                        string out_tag_str = tag.dump();
                        flame::component::ZData out_msg;
                        out_msg.from = monitor_portname;
                        out_msg.meta = out_tag_str;
                        out_msg.addmem(nullptr, 0);

                        if (!dispatch(monitor_portname, out_msg)) {
                            logger::warn("[{}] Failed to dispatch fault state message to port {}", getName(), monitor_portname);
                        }
                        continue;
                    }

                    /* ---- Resize if a target resolution is configured ---- */
                    cv::Mat out_image;
                    json out_tag = tag; // copy tag for potential modification

                    if (target_res.has_resolution &&
                        (target_res.width != src_image.cols || target_res.height != src_image.rows)) {
                        cv::resize(src_image, out_image,
                                   cv::Size(target_res.width, target_res.height),
                                   0, 0, cv::INTER_LINEAR);
                        out_tag["width"]  = target_res.width;
                        out_tag["height"] = target_res.height;
                    } else {
                        out_image = src_image;
                    }

                    /* ---- JPEG encoding ---- */
                    vector<uchar> jpeg_buf;
                    if (!cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        logger::warn("[{}] Failed to JPEG encode image for {}", getName(), monitor_portname);
                        continue;
                    }

                    /* ---- Build output ZData multipart message ---- */
                    string out_tag_str = out_tag.dump();

                    flame::component::ZData out_msg;
                    out_msg.from = monitor_portname;
                    out_msg.meta = out_tag_str;
                    out_msg.addmem(jpeg_buf.data(), jpeg_buf.size()); // Payload data frame

                    if (!dispatch(monitor_portname, out_msg)) {
                        logger::warn("[{}] Failed to dispatch image to port {}", getName(), monitor_portname);
                    }
                }

                auto end_time = chrono::high_resolution_clock::now();
                chrono::duration<double, std::milli> elapsed = end_time - start_time;
                // logger::debug("[{}] [{}] Processing loop time: {:.3f} ms", getName(), stream_name, elapsed.count());
            }
        } catch (const std::exception& e) {
            if (!_stop_threads.load()) {
                logger::error("[{}] Exception in monitor task for {}: {}", getName(), stream_name, e.what());
            }
            break;
        }
    }


    logger::debug("[{}] Stopped monitor task for {}", getName(), stream_name);
}
