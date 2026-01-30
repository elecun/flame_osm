/**
 * @file uvc.camera.grabber.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief UVC Camera Component using opencv uvc camera interface
 * @version 0.1
 * @date 2025-04-03
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_UVC_CAMERA_GRABBER_HPP_INCLUDED
#define FLAME_UVC_CAMERA_GRABBER_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <flame/common/zpipe.hpp>

using namespace std;

class uvc_camera_grabber : public flame::component::object {
    public:
        uvc_camera_grabber() = default;
        virtual ~uvc_camera_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message(const flame::component::message_t& msg) override;

    private:
        /* grabber tasks */
        void _grab_task(int camera_id, json camera_param);

        /* private function */
        vector<string> find_available_camera(int n_max=10, const string prefix="/dev/video");

    private:
        /* grabbing worker */
        unordered_map<int, thread> _grab_worker;
        
        /* zpipe */
        std::shared_ptr<flame::pipe::ZPipe> _pipe;
        std::map<int, std::shared_ptr<flame::pipe::AsyncZSocket>> _pub_sockets;

        /* flag */
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream_monitoring { false };
        atomic<bool> _use_image_stream { false };
        atomic<double> _rotation_cw { 0.0 };
        mutex _calibration_mtx;
        cv::Mat _map1, _map2;
        atomic<bool> _use_undistortion { false };

}; /* class */

EXPORT_COMPONENT_API


#endif