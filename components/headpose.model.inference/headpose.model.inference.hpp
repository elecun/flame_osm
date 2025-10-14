/**
 * @file headpose.model.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Headpose estimation using MediaPipe
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_HEADPOSE_MODEL_INFERENCE_HPP_INCLUDED
#define FLAME_HEADPOSE_MODEL_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <map>
#include <unordered_map>
#include <vector>
#include <thread>
#include <string>
#include <atomic>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <zmq.hpp>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/graphs/face_mesh/face_mesh_desktop_live.h>
#include <mutex>
#include <condition_variable>
#include <queue>


class headpose_model_inference : public flame::component::object {
public:
    headpose_model_inference() = default;
    virtual ~headpose_model_inference() = default;

    /* default interface functions */
    bool on_init() override;
    void on_loop() override;
    void on_close() override;
    void on_message(const message_t& msg) override;

private:
    /* ZMQ related */
    zmq::context_t* zmq_context_;
    zmq::socket_t* zmq_socket_;
    std::string zmq_endpoint_;

    /* MediaPipe related */
    std::unique_ptr<mediapipe::CalculatorGraph> graph_;
    mediapipe::CalculatorGraphConfig graph_config_;

    /* Threading */
    std::thread processing_thread_;
    std::atomic<bool> is_running_;
    std::mutex image_mutex_;
    std::condition_variable image_cv_;
    std::queue<cv::Mat> image_queue_;

    /* 3D Model points for head pose estimation */
    std::vector<cv::Point3f> model_points_;

    /* Camera intrinsic parameters */
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    /* Methods */
    void init_mediapipe();
    void init_3d_model_points();
    void processing_loop();
    cv::Vec3d calculate_head_pose(const std::vector<cv::Point2f>& landmarks);
    void print_landmarks_and_pose(const std::vector<cv::Point2f>& landmarks, const cv::Vec3d& pose);

}; /* class */

EXPORT_COMPONENT_API

#endif