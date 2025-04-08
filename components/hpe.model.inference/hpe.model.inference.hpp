/**
 * @file hpe.model.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Human Pose Estimation Model Inference Component
 * @version 0.1
 * @date 2025-04-03
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_HPE_MODLE_INFERENCE_HPP_INCLUDED
#define FLAME_HPE_MODLE_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>

class hpe_model_inference : public flame::component::object {
    public:
    hpe_model_inference() = default;
        virtual ~hpe_model_inference() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;

}; /* class */

EXPORT_COMPONENT_API

#endif