/**
 * @file solectrix.camera.grabber.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Frame Grabber with Solectrix proFRAME 3.0 + TI954 Adapter
 * @version 0.1
 * @date 2025-05-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_SOLECTRIX_CAMERA_GRABBER_HPP_INCLUDED
#define FLAME_SOLECTRIX_CAMERA_GRABBER_HPP_INCLUDED

#include <flame/component/object.hpp>
#include "sxpf_grabber.hpp"
#include <vector>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class solectrix_camera_grabber : public flame::component::object {
    public:
        solectrix_camera_grabber() = default;
        virtual ~solectrix_camera_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;


    private:
        /* grabber tasks */
        void _grab_task(json parameters);

    private:
        /* grabbing worker */
        thread _grab_worker;

        /* flags */
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream { false };
        atomic<bool> _use_image_stream_monitoring { false };


}; /* class */

EXPORT_COMPONENT_API


 #endif