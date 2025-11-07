/**
 * @file kps.holistic.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Mediapipe Holistic Model (Face Mesh + Body Pose + Hands Key points)
 * @version 0.1
 * @date 2025-10-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_KPS_HOLISITC_MODEL_INFERENCE_HPP_INCLUDED
#define FLAME_HOLISITC_MODEL_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>


class kps_holistic_inference : public flame::component::object {
public:
    kps_holistic_inference() = default;
    virtual ~kps_holistic_inference() = default;

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