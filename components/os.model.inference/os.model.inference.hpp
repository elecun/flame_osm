/**
 * @file os.model.inference.hpp
 * @author Byunghun Hwang
 * @brief Occupant Status Analyzer
 * @version 0.1
 * @date 2025-08-20
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_OS_MODEL_INFERENCE_HPP_INCLUDED
#define FLAME_OS_MODEL_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>

using namespace std;

class os_model_inference : public flame::component::object {
public:
    os_model_inference() = default;
    virtual ~os_model_inference() = default;

    /* default interface functions */
    bool on_init() override;
    void on_loop() override;
    void on_close() override;
    void on_message() override;

}; /* class */

EXPORT_COMPONENT_API

#endif