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


#ifndef FLAMEVIDEO_FILE_GRABBER_HPP_INCLUDED
#define FLAMEVIDEO_FILE_GRABBER_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <memory>
#include <atomic>
#include <functional>
#include <unordered_map>

using namespace std;

class video_file_grabber : public flame::component::object {
    public:
        video_file_grabber() = default;
        virtual ~video_file_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;

    private:
        /* task processing by action invoker */
        void _action_invoke_listener_proc(json parameters);

    private:
        void api_load_video(const json& args);

    private:
        /* action_apis */
        unordered_map<string, function<void(json)>> api_table;

        /* video capture */
        cv::VideoCapture _cap;

        /* flag */
        thread _action_invoke_listener;
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream_monitoring { false };
        atomic<bool> _use_image_stream { false };
        

}; /* class */

EXPORT_COMPONENT_API


#endif