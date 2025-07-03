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

using namespace std;

class uvc_camera_grabber : public flame::component::object {
    public:
        uvc_camera_grabber() = default;
        virtual ~uvc_camera_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;

    private:
        /* grabber tasks */
        void _grab_task(json camera_param);

        /* private function */
        vector<string> find_available_camera(int n_max=10, const string prefix="/dev/video");

    private:
        /* grabbing worker */
        unordered_map<int, thread> _grab_worker;

        /* flag */
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream_monitoring { false };
        atomic<bool> _use_image_stream { false };
        

}; /* class */

EXPORT_COMPONENT_API


#endif