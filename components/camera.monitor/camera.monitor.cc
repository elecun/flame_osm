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
                        logger::info("[{}] Monitor port '{}' target resolution: {}x{}",
                                     getName(), monitor_portname, res.width, res.height);
                    }
                }
            }
            _monitor_resolutions[monitor_portname] = res;
        };

        load_resolution("image_stream_1_monitor");
        load_resolution("image_stream_2_monitor");

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
        for (auto& [name, cv] : _queue_cvs) {
            cv.notify_all();
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
        for (auto& [name, q] : _monitor_queues) {
            queue<shared_ptr<flame::component::ZData>> empty_q;
            swap(q, empty_q);
        }

        logger::info("[{}] Component successfully closed.", getName());
    } catch (const std::exception& e) {
        logger::error("[{}] Error on close: {}", getName(), e.what());
    }
}

void camera_monitor::onData(flame::component::ZData& data)
{
    try {
        string portname = data.from;

        if (portname == "image_stream_1" || portname == "image_stream_2") {
            // Move received ZData frames into a new shared_ptr<ZData>
            auto msg = make_shared<flame::component::ZData>(std::move(data));
            msg->from = data.from;
            msg->meta = data.meta;

            /* push to channel queue */
            {
                lock_guard<mutex> lock(_queue_mtxs[portname]);
                if (_monitor_queues[portname].size() < _max_queue_size) {
                    _monitor_queues[portname].push(msg);
                } else {
                    _monitor_queues[portname].pop();
                    _monitor_queues[portname].push(msg);
                }
            }
            _queue_cvs[portname].notify_one();
            // logger::info("[{}] Received data from port: {}, frames count: {}", getName(), portname, msg->size());
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

    /* JPEG encoding parameters */
    vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};

    while (!_stop_threads.load()) {
        try {
            shared_ptr<flame::component::ZData> msg = nullptr;
            {
                unique_lock<mutex> lock(_queue_mtxs[stream_name]);
                _queue_cvs[stream_name].wait(lock, [this, stream_name] {
                    return !_monitor_queues[stream_name].empty() || _stop_threads.load();
                });

                if (_stop_threads.load() && _monitor_queues[stream_name].empty())
                    break;

                if (!_monitor_queues[stream_name].empty()) {
                    msg = _monitor_queues[stream_name].front();
                    _monitor_queues[stream_name].pop();
                }
            }

            if (msg) {
                auto start_time = chrono::high_resolution_clock::now();

                if (msg->size() >= 2) {
                    zmq::message_t tag_msg = msg->pop();
                    zmq::message_t img_msg = msg->pop();

                    /* ---- Parse the tag JSON to reconstruct cv::Mat ---- */
                    string tag_str(static_cast<char*>(tag_msg.data()), tag_msg.size());
                    json tag = json::parse(tag_str);

                    int src_height = tag["height"].get<int>();
                    int src_width  = tag["width"].get<int>();
                    int src_type   = tag["type"].get<int>();

                    /* Wrap raw buffer into cv::Mat (no copy) */
                    cv::Mat src_image(src_height, src_width, src_type,
                                     img_msg.data());

                    /* ---- Resize if a target resolution is configured ---- */
                    cv::Mat out_image;
                    json out_tag = tag; // copy tag for potential modification

                    if (target_res.has_resolution &&
                        (target_res.width != src_width || target_res.height != src_height)) {
                        cv::resize(src_image, out_image,
                                   cv::Size(target_res.width, target_res.height),
                                   0, 0, cv::INTER_LINEAR);
                        out_tag["width"]  = target_res.width;
                        out_tag["height"] = target_res.height;
                    } else {
                        out_image = src_image.clone(); // clone to ensure contiguous memory for encoding
                    }

                    /* ---- JPEG encoding ---- */
                    vector<uchar> jpeg_buf;
                    if (!cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                        logger::warn("[{}] Failed to JPEG encode image for {}", getName(), monitor_portname);
                        continue;
                    }

                    /* ---- Build output ZData multipart message ---- */
                    // Frame layout expected by camera.py:
                    //   Frame 0: topic (monitor portname string)
                    //   Frame 1: tag JSON string (metadata)
                    //   Frame 2: JPEG encoded image data
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
                logger::debug("[{}] [{}] Processing loop time: {:.3f} ms", getName(), stream_name, elapsed.count());
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
