#include "kvaser.can.interface.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <thread>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace flame;
using namespace std;

/* create component instance */
static kvaser_can_interface* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new kvaser_can_interface(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool kvaser_can_interface::onInit(){

    try{

        /* read profile */
        json parameters = getProfile()->parameters();

        /* CAN device initialization */
        canInitializeLibrary();
        
        /* find channels */
        canStatus _status { canOK };
        _status = canGetNumberOfChannels(&_can_channels);
        if(_status != canOK){
            char err[512] = {0,};
            canGetErrorText((canStatus)_status, err, sizeof(err));
            logger::error("[{}] Cannot find CAN channel. Error code : {}", getName(), err);
            return false;
        }
        logger::debug("[{}] Found {} CAN channel(s) ", getName(), _can_channels);

        /* show channel info. */
        for(int i=0; i<_can_channels; i++){
            char device_name[128] = {0};
            int ch_on_card = 0;

            // read channel name
            _status = canGetChannelData(i, canCHANNELDATA_DEVDESCR_ASCII, device_name, sizeof(device_name));
            if (_status != canOK) {
                logger::warn("[{}] Channel {} name retrieval failed. Error code : {}", getName(), i, (int)_status);
            }

            // read can channel
            _status = canGetChannelData(i, canCHANNELDATA_CHAN_NO_ON_CARD, &ch_on_card, sizeof(ch_on_card));
            if (_status != canOK) {
                ch_on_card = -1;
                logger::warn("[{}] Channel {} on card retrieval failed. Error code : {}", getName(), i, (int)_status);
            }
        
            logger::debug("[{}] CAN Channel {}: Device Name: {}\tDevice Channel: {}", getName(), i, device_name, ch_on_card);
        }

        /* use channel */
        int ch = parameters.value("use_channel", 0);
        // Open the channel as CAN FD
        _can_handle = canOpenChannel(ch, canOPEN_CAN_FD | canOPEN_ACCEPT_VIRTUAL);
        if(_can_handle < 0){
            char err_buf[100];
            canGetErrorText((canStatus)_can_handle, err_buf, sizeof(err_buf));
            logger::error("[{}] Error opening channel: {}", getName(), err_buf);
            return false;
        }
        else{
            // 1. Set Arbitration Phase Bitrate (Nominal Phase)
            unsigned long bitrate = parameters.value("can_bitrate", 500000);
            double nominal_sp = parameters.value("can_bitrate_sample_point", 80.0) / 100.0;
            unsigned int total_tq_nom = (bitrate <= 500000) ? 80 : 40;
            unsigned int tseg1_nom = static_cast<unsigned int>(std::round(total_tq_nom * nominal_sp)) - 1;
            unsigned int tseg2_nom = total_tq_nom - 1 - tseg1_nom;
            unsigned int sjw_nom = tseg2_nom;

            logger::info("Nominal bitrate: {} bps", bitrate);
            logger::info("Nominal sample point: {}%", nominal_sp * 100.0);
            logger::info("Nominal total TQ: {}", total_tq_nom);
            logger::info("Nominal tseg1: {}", tseg1_nom);
            logger::info("Nominal tseg2: {}", tseg2_nom);
            logger::info("Nominal SJW: {}", sjw_nom);

            canStatus nom_status = canSetBusParams(_can_handle, bitrate, tseg1_nom, tseg2_nom, sjw_nom, 1, 0);
            if (nom_status != canOK) {
                char err[512] = {0,};
                canGetErrorText(nom_status, err, sizeof(err));
                logger::error("[{}] Failed to set Nominal bitrate {} bps with sample point {}%: {}", 
                              getName(), bitrate, nominal_sp * 100.0, err);
                return false;
            }

            // 2. Set Data Phase Bitrate for CAN FD
            unsigned long fd_data_bitrate = parameters.value("can_fd_data_bitrate", 1000000);
            double data_sp = parameters.value("can_fd_data_bitrate_sample_point", 75.0) / 100.0;
            unsigned int total_tq_data = (fd_data_bitrate <= 2000000) ? 40 : 20;
            unsigned int tseg1_data = static_cast<unsigned int>(std::round(total_tq_data * data_sp)) - 1;
            unsigned int tseg2_data = total_tq_data - 1 - tseg1_data;
            unsigned int sjw_data = tseg2_data;

            canStatus fd_status = canSetBusParamsFd(_can_handle, fd_data_bitrate, tseg1_data, tseg2_data, sjw_data);
            if(fd_status != canOK) {
                char err[512] = {0,};
                canGetErrorText(fd_status, err, sizeof(err));
                logger::error("[{}] Failed to set CAN FD data bitrate {} bps with sample point {}%: {}", 
                              getName(), fd_data_bitrate, data_sp * 100.0, err);
                return false;
            }

            logger::info("Data bitrate : {} bps", fd_data_bitrate);
            logger::info("Data sample point : {}%", data_sp * 100.0);
            logger::info("Data total TQ : {}", total_tq_data);
            logger::info("Data tseg1 : {}", tseg1_data);
            logger::info("Data tseg2 : {}", tseg2_data);
            logger::info("Data SJW : {}", sjw_data);

            _status = canBusOn(_can_handle);
            if(_status != canOK){
                char err[512] = {0,};
                canGetErrorText(_status, err, sizeof(err));
                logger::error("[{}] Failed to go bus ON : {}", getName(), err);
                return false;
            }
            logger::debug("[{}] CAN FD channel {} opened & set arbitration bitrate {} bps (SP: {}%), data bitrate {} bps (SP: {}%)", 
                          getName(), ch, bitrate, nominal_sp * 100.0, fd_data_bitrate, data_sp * 100.0);

            /* start background workers */
            bool enable_rx_task = parameters.value("enable_rx_task", true);
            if (enable_rx_task) {
                _worker_stop.store(false);
                logger::info("[{}] Listen for CAN FD frames...", getName());
                _can_ch0_rcv_worker = thread(&kvaser_can_interface::_can_ch0_rcv_task, this);
            }
        }   
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void kvaser_can_interface::onLoop(){
    if(_can_handle < 0){
        return;
    }

    DMSEnable enable;
    DMSState state;
    DMSDriverReadiness readiness;
    {
        std::lock_guard<std::mutex> lock(_vars_mutex);
        enable = _dms_enable;
        state = _dms_state;
        readiness = _dms_readiness;
    }

    if(enable == DMSEnable::ENABLE) {
        // Pack DMS_State and DMS_DriverPresent (readiness) into 0x220 message
        // STS_DMS_1000ms (0x220):
        // - DMS_State: Start Bit 0, Len 2
        // - DMS_DriverPresent: Start Bit 2, Len 2
        unsigned char tx_data[8] = {0};
        tx_data[0] = (static_cast<uint8_t>(state) & 0x03) | 
                     ((static_cast<uint8_t>(readiness) & 0x03) << 2);

        unsigned int flags = canMSG_STD | canFDMSG_FDF | canFDMSG_BRS;
        canStatus stat = canWrite(_can_handle, 0x220, tx_data, 8, flags);
        if (stat != canOK) {
            char err_buf[100];
            canGetErrorText(stat, err_buf, sizeof(err_buf));
            logger::error("[{}] Error sending periodic CAN message 0x220: {}", getName(), err_buf);
        } else {
            logger::debug("[{}] Sent periodic CAN FD message 0x220 (State: {}, Readiness: {})", getName(), static_cast<int>(state), static_cast<int>(readiness));
        }
    }
}

void kvaser_can_interface::onClose(){
    
    /* stop workers */
    _worker_stop.store(true);
    if(_can_ch0_rcv_worker.joinable()) {
        _can_ch0_rcv_worker.join();
    }

    /* close CAN */
    if(_can_handle >= 0) {
        canBusOff(_can_handle);
        canClose(_can_handle);
        _can_handle = canINVALID_HANDLE;
    }
    canUnloadLibrary();
    logger::info("[{}] CAN Interface successfully closed.", getName());
}

void kvaser_can_interface::onData(flame::component::ZData& data){
    try {
        string portname = data.from;
        if (portname == "can_ch1_in") {
            if (data.size() > 0) {
                zmq::message_t msg = data.pop();
                string payload(static_cast<char*>(msg.data()), msg.size());
                
                if (!payload.empty() && (payload.front() == '{' || payload.front() == '[')) {
                    json j = json::parse(payload);
                    
                    // Check if it is a request to update shared variables
                    bool is_var_update = false;
                    {
                        std::lock_guard<std::mutex> lock(_vars_mutex);
                        if (j.contains("dms_enable")) {
                            int val = j["dms_enable"].get<int>();
                            _dms_enable = (val == 0) ? DMSEnable::DISABLE : DMSEnable::ENABLE;
                            logger::info("[{}] Updated dms_enable via onData: {}", getName(), val);
                            is_var_update = true;
                        }
                        if (j.contains("dms_state")) {
                            int val = j["dms_state"].get<int>();
                            _dms_state = static_cast<DMSState>(val);
                            logger::info("[{}] Updated dms_state via onData: {}", getName(), val);
                            is_var_update = true;
                        }
                        if (j.contains("dms_readiness")) {
                            int val = j["dms_readiness"].get<int>();
                            _dms_readiness = static_cast<DMSDriverReadiness>(val);
                            logger::info("[{}] Updated dms_readiness via onData: {}", getName(), val);
                            is_var_update = true;
                        }
                    }

                    // Alternatively, check if it's a request to transmit a custom CAN frame
                    if (!is_var_update && j.contains("id") && j.contains("data")) {
                        unsigned int id = j["id"].get<unsigned int>();
                        vector<uint8_t> vec_data = j["data"].get<vector<uint8_t>>();
                        unsigned int dlc = j.value("dlc", (unsigned int)vec_data.size());
                        unsigned int flags = j.value("flags", (unsigned int)(canMSG_STD | canFDMSG_FDF | canFDMSG_BRS));
                        
                        unsigned char tx_buf[64] = {0};
                        size_t tx_len = std::min(vec_data.size(), sizeof(tx_buf));
                        std::copy(vec_data.begin(), vec_data.begin() + tx_len, tx_buf);
                        
                        canStatus stat = canWrite(_can_handle, id, tx_buf, dlc, flags);
                        if (stat != canOK) {
                            char err_buf[100];
                            canGetErrorText(stat, err_buf, sizeof(err_buf));
                            logger::error("[{}] Error sending custom CAN message via onData: {}", getName(), err_buf);
                        } else {
                            logger::debug("[{}] Sent custom CAN message {} via onData", getName(), id);
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}

void kvaser_can_interface::_can_ch0_rcv_task(){

    try{
        while(!_worker_stop.load()){
            long id;
            unsigned char data[64]; // Sized 64 bytes for CAN FD payloads
            unsigned int dlc;
            unsigned int flags;
            unsigned long time;

            canStatus stat = canRead(_can_handle, &id, data, &dlc, &flags, &time);
            if(stat == canOK) {
                std::ostringstream oss;
                for(unsigned int i=0; i < dlc && i < 64; ++i){
                    oss << std::hex << std::uppercase << std::setfill('0') << std::setw(2) << static_cast<int>(data[i]) << " ";
                }
                logger::debug("[{}] ID({}) | DLC({}) | Flags({}) | Data({})", getName(), id, dlc, flags, oss.str());

                // Read and update enable/disable signal
                // CMD_DMS_1000ms: ID 0x120, Signal: DMS_Enable (Start Bit: 0, Len: 1)
                if(id == 0x120) {
                    if (dlc >= 1) {
                        uint8_t dms_enable_val = data[0] & 0x01;
                        {
                            std::lock_guard<std::mutex> lock(_vars_mutex);
                            _dms_enable = (dms_enable_val == 0) ? DMSEnable::DISABLE : DMSEnable::ENABLE;
                        }
                        logger::info("[{}] Received CMD_DMS_1000ms: DMS_Enable set to {}", getName(), (dms_enable_val ? "ENABLE" : "DISABLE"));
                    } else {
                        logger::warn("[{}] Received CMD_DMS_1000ms but DLC is {}", getName(), dlc);
                    }
                }

                // ISC_01_10ms: ID 0x100
                if(id == 0x100) {
                    if (dlc >= 8) {
                        uint8_t master_status = data[0] & 0x03;
                        uint8_t operating_case = (data[0] >> 2) & 0x0F;
                        uint8_t fault_clear_req = (data[0] >> 7) & 0x01;
                        uint8_t dms_enable = data[1] & 0x01;
                        uint8_t dms_state = (data[1] >> 1) & 0x03;
                        uint8_t dms_driver_present = (data[1] >> 3) & 0x03;

                        logger::info("[{}] Received ISC_01_10ms: Raw: {}| MasterStatus: {}, CASE: {}, FaultClear: {}, DMS_Enable: {}, DMS_State: {}, DMS_DriverPresent: {}", 
                                     getName(), oss.str(), master_status, operating_case, fault_clear_req, dms_enable, dms_state, dms_driver_present);
                    } else {
                        logger::warn("[{}] Received ISC_01_10ms but DLC is {}", getName(), dlc);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } /* end while */

        logger::info("[{}] Receiver thread successfully stopped.", getName());
    }
    catch(const std::out_of_range& e){
        logger::error("[{}] Invalid parameter access in receiver task", getName());
    }
    catch(const zmq::error_t& e){
        logger::error("[{}] Pipeline Error in receiver task: {}", getName(), e.what());
    }
    catch(const json::exception& e){
        logger::error("[{}] Data Parse Error in receiver task: {}", getName(), e.what());
    }

}


/* Thread-safe getters and setters implementation */
void kvaser_can_interface::set_dms_enable(DMSEnable val) {
    std::lock_guard<std::mutex> lock(_vars_mutex);
    _dms_enable = val;
}

DMSEnable kvaser_can_interface::get_dms_enable() {
    std::lock_guard<std::mutex> lock(_vars_mutex);
    return _dms_enable;
}

void kvaser_can_interface::set_dms_state(DMSState val) {
    std::lock_guard<std::mutex> lock(_vars_mutex);
    _dms_state = val;
}

DMSState kvaser_can_interface::get_dms_state() {
    std::lock_guard<std::mutex> lock(_vars_mutex);
    return _dms_state;
}

void kvaser_can_interface::set_dms_readiness(DMSDriverReadiness val) {
    std::lock_guard<std::mutex> lock(_vars_mutex);
    _dms_readiness = val;
}

DMSDriverReadiness kvaser_can_interface::get_dms_readiness() {
    std::lock_guard<std::mutex> lock(_vars_mutex);
    return _dms_readiness;
}