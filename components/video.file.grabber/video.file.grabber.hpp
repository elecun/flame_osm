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
using namespace flame::component;

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
        /* grabber tasks */
        void _grab_task(json parameters);

    private:
        /* grabbing worker */
        thread _grab_worker;

        /* flags */
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream { false };
        atomic<bool> _use_image_stream_monitoring { false };

        /* video capture device */
        unique_ptr<cv::VideoCapture> _video_capture;

}; /* class */

EXPORT_COMPONENT_API


#endif