/**
 * @file fpdlink.camera.grabber.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief TI FPD Link III camera grabber (with TI954 Deserializer)
 * @version 0.1
 * @date 2025-05-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_FPDLINK_CAMERA_GRABBER_HPP_INCLUDED
#define FLAME_FPDLINK_CAMERA_GRABBER_HPP_INCLUDED

#include <flame/component/object.hpp>

using namespace std;

class fpdlink_camera_grabber : public flame::component::object {
    public:
        fpdlink_camera_grabber() = default;
        virtual ~fpdlink_camera_grabber() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;

}; /* class */

EXPORT_COMPONENT_API


 #endif