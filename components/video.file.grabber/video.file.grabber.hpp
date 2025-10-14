/**
 * @file video.file.grabber.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief frame grabber for video file interface
 * @version 0.1
 * @date 2025-07-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */


#ifndef FLAME_VIDEO_FILE_GRABBER_HPP_INCLUDED
#define FLAME_VIDEO_FILE_GRABBER_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <memory>
#include <atomic>
#include <functional>
#include <unordered_map>

using namespace std;
using namespace cv;

class video_file_grabber : public flame::component::object {
    public:
        video_file_grabber() = default;
        virtual ~video_file_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message(const message_t& msg) override;

    private:
        /* task processing by action invoker */
        void _action_invoke_listener_proc(json parameters);
        void _action_proc(json args);

    private:
        /* rpc-like api functions */
        void api_start_grab(const json& args);  /* start frame grab */
        void api_stop_grab(const json& args);   /* stop frame grab */

    private:
        /* action thread */
        atomic<bool> _action_working { false }; /* action thread termination control */
        thread _invoked_action_thread;

    private:
        /* action_apis */
        unordered_map<string, function<void(json)>> api_table;

        /* video capture */
        cv::VideoCapture _cap;

        /* flag */
        thread _action_invoke_listener;
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream { false };
        

}; /* class */

EXPORT_COMPONENT_API


#endif