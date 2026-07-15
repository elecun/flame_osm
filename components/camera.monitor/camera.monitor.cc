#include "camera.monitor.hpp"

#include <flame/log.hpp>

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



        // Start threads for image_stream_0 and image_stream_2
        _monitor_threads["image_stream_0"] = thread(&camera_monitor::_monitor_task, this, "image_stream_0", "image_stream_1_monitor");
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

        if (portname == "image_stream_0" || portname == "image_stream_2") {
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


                    // Forward the exact frames to the monitor port using a new ZData
                    flame::component::ZData out_msg;
                    out_msg.from = monitor_portname;
                    out_msg.meta = string(static_cast<char*>(tag_msg.data()), tag_msg.size());
                    out_msg.addmem(tag_msg.data(), tag_msg.size());
                    out_msg.addmem(img_msg.data(), img_msg.size());

                    if (!dispatch(monitor_portname, out_msg)) {
                        logger::warn("[{}] Failed to dispatch image to port {}", getName(), monitor_portname);
                    }
                }

                auto end_time = chrono::high_resolution_clock::now();
                chrono::duration<double, std::milli> elapsed = end_time - start_time;
                logger::info("[{}] [{}] Processing loop time: {:.3f} ms", getName(), stream_name, elapsed.count());
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
