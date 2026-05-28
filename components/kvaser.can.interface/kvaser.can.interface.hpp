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
#include <mutex>

extern "C" {
    #include <canlib.h>
}

// DMS configuration and status enums
enum class DMSEnable : uint8_t {
    DISABLE = 0x0,
    ENABLE = 0x1
};

enum class DMSState : uint8_t {
    INIT = 0x0,
    INACTIVE = 0x1,
    ACTIVE = 0x2,
    FAULT = 0x3
};

enum class DMSDriverReadiness : uint8_t {
    UNKNOWN = 0x0,
    HIGH = 0x1,
    MODERATE = 0x2,
    LOW = 0x3
};

class kvaser_can_interface : public flame::component::Object {
public:
    kvaser_can_interface() = default;
    virtual ~kvaser_can_interface() = default;

    /* default interface functions */
    bool onInit() override;
    void onLoop() override;
    void onClose() override;
    void onData(flame::component::ZData& data) override;

    /* thread-safe getters and setters for shared variables */
    void set_dms_enable(DMSEnable val);
    DMSEnable get_dms_enable();

    void set_dms_state(DMSState val);
    DMSState get_dms_state();

    void set_dms_readiness(DMSDriverReadiness val);
    DMSDriverReadiness get_dms_readiness();

private: /* private functions */
    void _can_ch0_rcv_task();
    void _can_tx_task();

private:
    /* worker related */
    std::thread _can_ch0_rcv_worker; /* can channel 0 receiver */
    std::thread _can_tx_worker;      /* periodic CAN transmitter */
    std::atomic<bool> _worker_stop { false };

    /* CAN device related */
    canHandle _can_handle { canINVALID_HANDLE };
    int _can_channels = {0};

    /* shared states protected by mutex */
    std::mutex _vars_mutex;
    DMSEnable _dms_enable { DMSEnable::ENABLE }; // CMD_DMS_1000ms: Init value is 1 (Enable)
    DMSState _dms_state { DMSState::INIT };       // STS_DMS_1000ms: Init value is 0 (Init)
    DMSDriverReadiness _dms_readiness { DMSDriverReadiness::UNKNOWN }; // STS_DMS_1000ms: Init value is 0 (Unknown)

}; /* class */

EXPORT_COMPONENT_API

#endif