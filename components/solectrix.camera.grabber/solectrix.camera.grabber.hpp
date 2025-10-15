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
#include <vector>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "sxpf_grabber.hpp"

using namespace std;
using namespace cv;
using namespace flame::component;

class solectrix_camera_grabber : public flame::component::object {
    public:
        solectrix_camera_grabber() = default;
        virtual ~solectrix_camera_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message(const message_t& msg) override;

        /* device control functions */
        bool open_device(int endpoint_id = 0, int channel_id = 4, uint32_t decode_csi2_datatype = 0x1e, int left_shift = 8);
        void close_device();
        
        /* grab function */
        cv::Mat grab();

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

        /* grabber device */
        unique_ptr<sxpf_grabber> _grabber_handle;
        
        /* device state */
        sxpf_hdl _fg { 0 };
        HWAITSXPF _devfd { 0 };
        bool _device_opened { false };
        int _endpoint_id { 0 };
        int _channel_id { 4 };
        uint32_t _decode_csi2_datatype { 0x1e };
        int _left_shift { 8 };


}; /* class */

EXPORT_COMPONENT_API


 #endif