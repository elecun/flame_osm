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
#include <opencv2/opencv.hpp>

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
        /* Latest Images Caching */
        cv::Mat _latest_image_1;
        cv::Mat _latest_image_2;

        /* Mutexes for Thread Safety */
        std::mutex _img_mutex_1;
        std::mutex _img_mutex_2;
};

EXPORT_COMPONENT_API

#endif
