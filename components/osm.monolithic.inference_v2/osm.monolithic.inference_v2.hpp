/**
 * @file osm.monolithic.inference_v2.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief OSM Monolithic Inference Component V2 (with DAD-3DHeads E2E)
 * @version 0.2
 * @date 2026-09-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef FLAME_OSM_MONOLITHIC_INFERENCE_V2_HPP_INCLUDED
#define FLAME_OSM_MONOLITHIC_INFERENCE_V2_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <deque>
#include <opencv2/opencv.hpp>
#include "face_detection.hpp"
#include "face_analysis_e2e.hpp"
#include "body_pose_estimation.hpp"
#include "driver_readiness_estimation.hpp"
#include "driver_readiness_estimation_logical.hpp"

using namespace std;
using namespace flame::component;

class osm_monolithic_inference_v2 : public flame::component::Object {
    public:
        osm_monolithic_inference_v2();
        virtual ~osm_monolithic_inference_v2() = default;

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

        /* Face Detector Instance (YOLO) */
        std::unique_ptr<face_detection> _face_detector;

        /* DAD-3DHeads E2E Face Analysis Instance */
        std::unique_ptr<face_analysis_e2e> _face_analyzer_e2e;

        /* Body Pose Estimator Instance */
        std::unique_ptr<body_pose_estimation> _body_pose_estimator;

        /* Driver Readiness Estimator Instances */
        std::unique_ptr<driver_readiness_estimation> _driver_readiness_estimator;
        std::unique_ptr<driver_readiness_estimation_logical> _driver_readiness_logical_estimator;

        /* Model Execution Flags */
        bool _use_face_det{true};
        bool _use_face_analysis_e2e{true};
        bool _use_body_pose{true};
        bool _use_driver_readiness{false};
        bool _use_driver_readiness_logical{true};

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
        bool _vertical_flip = false;

        /* Visualization Flags */
        bool _vis_face_det{true};
        bool _vis_face_analysis_e2e{true};
        bool _vis_landmarks_68{true};
        bool _vis_landmarks_191{false};
        bool _vis_head_pose{true};
        bool _vis_square_box{true};
        bool _vis_head_mesh{false};
        bool _vis_body_pose{true};
        bool _vis_driver_readiness{true};
        bool _vis_driver_readiness_logical{true};

        /* ROI configuration */
        bool _use_roi{false};
        bool _roi_visualize{true};
        int _roi_x1{0};
        int _roi_y1{0};
        int _roi_x2{0};
        int _roi_y2{0};

        /* DMS Score History for Visualization Graph */
        std::deque<std::pair<std::chrono::steady_clock::time_point, double>> _readiness_history;
        std::mutex _history_mutex;
        void draw_readiness_graph(cv::Mat& image, int x, int y, int width, int height);
};

EXPORT_COMPONENT_API

#endif
