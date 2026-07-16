/**
 * @file camera.monitor.hpp
 * @author Byunghun Hwang (you@domain.com)
 * @brief Camera Monitoring Service Component (1:1 use only)
 * @version 0.1
 * @date 2026-05-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef FLAME_CAMERA_MONITOR_HPP_INCLUDED
#define FLAME_CAMERA_MONITOR_HPP_INCLUDED

#include <atomic>
#include <condition_variable>
#include <flame/component/object.hpp>
#include <mutex>

#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace flame::component;

/**
 * @brief Per-monitor-port resolution configuration.
 *        If has_resolution is false the original image size is kept.
 */
struct MonitorResolution {
    bool has_resolution{false};
    int  width{0};
    int  height{0};
};

class camera_monitor : public flame::component::Object {
public:
    camera_monitor();
    virtual ~camera_monitor() = default;

    /* default interface functions */
    bool onInit() override;
    void onLoop() override;
    void onClose() override;
    void onData(flame::component::ZData& data) override;

private:
    /* thread task */
    void _monitor_task(string stream_name, string monitor_portname);

private:
    unordered_map<string, chrono::time_point<chrono::high_resolution_clock>> _last_received_times;
    mutex _mtx;

    /* Thread and Queue management */
    atomic<bool> _stop_threads{false};
    unordered_map<string, thread> _monitor_threads;
    unordered_map<string, queue<shared_ptr<flame::component::ZData>>> _monitor_queues;
    unordered_map<string, mutex> _queue_mtxs;
    unordered_map<string, condition_variable> _queue_cvs;
    const size_t _max_queue_size = 5;

    /* Per-monitor-port target resolution (keyed by monitor portname) */
    unordered_map<string, MonitorResolution> _monitor_resolutions;
};

EXPORT_COMPONENT_API

#endif