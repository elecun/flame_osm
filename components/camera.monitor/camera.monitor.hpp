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

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <vector>
#include <unordered_map>
#include <chrono>


using namespace std;
using namespace flame::component;

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
        unordered_map<string, chrono::time_point<chrono::high_resolution_clock>> _last_received_times;
        mutex _mtx;
};

EXPORT_COMPONENT_API

#endif