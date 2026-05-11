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

using namespace std;
using namespace cv;
using namespace flame::component;


class kps_holistic_inference : public flame::component::Object {
public:
    kps_holistic_inference() = default;
    virtual ~kps_holistic_inference() = default;

    /* default interface functions */
    bool onInit() override;
    void onLoop() override;
    void onClose() override;
    void onData(flame::component::ZData& data) override;

private:
    

}; /* class */

EXPORT_COMPONENT_API

#endif