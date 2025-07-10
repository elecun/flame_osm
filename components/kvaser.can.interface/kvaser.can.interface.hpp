/**
 * @file kvaser.can.interface.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Kvaser CAN Control Interface
 * @version 0.1
 * @date 2025-07-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_KVASER_CAN_INTERFACE_HPP_INCLUDED
#define FLAME_KVASER_CAN_INTERFACE_HPP_INCLUDED

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
extern "C" {
    #include <canlib.h>
}

class kvaser_can_interface : public flame::component::object {
public:
kvaser_can_interface() = default;
    virtual ~kvaser_can_interface() = default;

    /* default interface functions */
    bool on_init() override;
    void on_loop() override;
    void on_close() override;
    void on_message() override;

private: /* private functions */
    void _can_ch0_rcv_task();

private:
    /* worker related */
    thread _can_ch0_rcv_worker; /* can channel 0 receiver */
    atomic<bool> _worker_stop { false };

    /* CAN device related */
    canHandle _can_handle { canINVALID_HANDLE };
    int _can_channels = {0};

}; /* class */

EXPORT_COMPONENT_API

#endif