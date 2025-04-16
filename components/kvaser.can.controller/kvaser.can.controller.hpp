/**
 * @file kvaser.can.controller.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Kvaser CAN Controller Component
 * @version 0.1
 * @date 2025-04-03
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_KVASER_CAN_CONTROLLER_HPP_INCLUDED
#define FLAME_KVASER_CAN_CONTROLLER_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <canlib.h>

class kvaser_can_controller : public flame::component::object {
public:
kvaser_can_controller() = default;
    virtual ~kvaser_can_controller() = default;

    /* default interface functions */
    bool on_init() override;
    void on_loop() override;
    void on_close() override;
    void on_message() override;

private:
    canHandle _handle { canINVALID_HANDLE };
    int _can_channels = {0};

}; /* class */

EXPORT_COMPONENT_API

#endif