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
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <atomic>
#include <chrono>
#include <string>

using namespace std;
using namespace cv;
using namespace flame::component;

class video_file_grabber : public flame::component::Object {
    public:
        video_file_grabber() = default;
        virtual ~video_file_grabber() = default;

        /* default interface functions */
        bool onInit() override;
        void onLoop() override;
        void onClose() override;
        void onData(flame::component::ZData& data) override;

    private:
        /* grabber tasks */
        void _grab_task();
        void _dispatch_task(string portname);

    private:
        /* grabbing worker */
        thread _grab_worker;
        unordered_map<string, thread> _dispatch_workers;
        unordered_map<string, queue<shared_ptr<flame::component::ZData>>> _dispatch_queues;
        unordered_map<string, mutex> _queue_mtxs;
        unordered_map<string, condition_variable> _queue_cvs;
        const size_t _max_queue_size = 5;

        /* flags */
        atomic<bool> _worker_stop { false };
        atomic<bool> _use_image_stream { false };
        atomic<bool> _fault_reset { false };
        atomic<bool> _capture_fault { false };
        atomic<int> _fault_limit { 3 };
        atomic<int> _fault_count { 0 };

        /* timings & ports */
        unordered_map<string, chrono::time_point<chrono::high_resolution_clock>> _last_capture_times;
        unordered_map<string, string> _port_rotations;
        vector<string> _output_ports;

        /* video playback */
        vector<string> _video_files;
        size_t _current_video_idx { 0 };
        unique_ptr<cv::VideoCapture> _video_capture;
        double _video_fps { 30.0 };
        int _total_frames { 0 };
        int _frame_width { 0 };
        int _frame_height { 0 };

        /* undistortion parameters */
        bool _enable_undistort { false };
        cv::Mat _camera_matrix;
        cv::Mat _dist_coeffs;
        cv::Mat _map1, _map2;

}; /* class */

EXPORT_COMPONENT_API

#endif