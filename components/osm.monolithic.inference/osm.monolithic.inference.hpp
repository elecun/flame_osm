/**
 * @file osm.monolithic.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief OSM Monolithic Inference Component
 * @version 0.1
 * @date 2026-07-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef FLAME_OSM_MONOLITHIC_INFERENCE_HPP_INCLUDED
#define FLAME_OSM_MONOLITHIC_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <opencv2/opencv.hpp>
#include "face_detection.hpp"
#include "body_pose_estimation.hpp"
#include "face_landmark_2d.hpp"

using namespace std;
using namespace flame::component;

class osm_monolithic_inference : public flame::component::Object {
    public:
        osm_monolithic_inference();
        virtual ~osm_monolithic_inference() = default;

        /* default interface functions */
        bool onInit() override;
        void onLoop() override;
        void onClose() override;
        void onData(flame::component::ZData& data) override;

        /* Thread-safe Image Getters */
        cv::Mat getLatestImage1();
        cv::Mat getLatestImage2();

    private:
        /* Inference worker thread loop */
        void _inference_process();

    private:
        /* Latest Images Caching */
        cv::Mat _latest_image_1;
        cv::Mat _latest_image_2;

        /* Mutexes for Thread Safety */
        std::mutex _img_mutex_1;
        std::mutex _img_mutex_2;

        /* Face Detector Instance */
        std::unique_ptr<face_detection> _face_detector;

        /* Body Pose Estimator Instance */
        std::unique_ptr<body_pose_estimation> _body_pose_estimator;

        /* Face Landmark 2D Instance */
        std::unique_ptr<face_landmark_2d> _face_landmark_2d;

        /* Thread Control */
        std::thread _inference_worker;
        std::atomic<bool> _worker_stop{false};

        /* Monitor port configuration */
        int _target_width = 800;
        int _target_height = 450;
        bool _has_target_resolution = false;
        bool _enable_stream_1 = false;
        bool _enable_stream_2 = false;
        float _nms_threshold = 0.45f;
        float _padding_w = 0.0f;
        float _padding_h = 0.0f;
        bool _show_info = true;
};

EXPORT_COMPONENT_API

#endif
