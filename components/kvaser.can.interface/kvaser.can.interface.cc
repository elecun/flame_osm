
#include "kvaser.can.interface.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static kvaser_can_interface* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new kvaser_can_interface(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool kvaser_can_interface::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* CAN device initialization */
        canInitializeLibrary();
        
        /* find channels */
        canStatus _status { canOK };
        _status = canGetNumberOfChannels(&_can_channels);
        if(_status != canOK){
            char err[512] = {0,};
            canGetErrorText((canStatus)_status, err, sizeof(err));
            logger::error("[{}] Cannot find CAN channel. Error code : {}", get_name(), err);
            return false;
        }
        logger::debug("[{}] Found {} CAN channel(s) ", get_name(), _can_channels);

        /* show channel info. */
        for(int i=0; i<_can_channels; i++){
            char device_name[128] = {0};
            int ch_on_card = 0;

            // read channel name
            _status = canGetChannelData(i, canCHANNELDATA_DEVDESCR_ASCII, device_name, sizeof(device_name));
            if (_status != canOK) {
                logger::warn("[{}] Channel {} name retrieval failed. Error code : {}", get_name(), i, (int)_status);
            }
            string str_device_name(device_name);

            // read can channel
            _status = canGetChannelData(i, canCHANNELDATA_CHAN_NO_ON_CARD, &ch_on_card, sizeof(ch_on_card));
            if (_status != canOK) {
                ch_on_card = -1;
                logger::warn("[{}] Channel {} on card retrieval failed. Error code : {}", get_name(), i, (int)_status);
            }
        
            logger::debug("[{}] CAN Channel {}: Device Name: {}\tDevice Channel: {}", get_name(), i, device_name, ch_on_card);
   
        }

        /* use channel */
        int ch = parameters.value("use_channel", 0);
        _can_handle = canOpenChannel(ch, canOPEN_ACCEPT_VIRTUAL);
        if(_can_handle<0){
            char err_buf[100];
            canGetErrorText((canStatus)_can_handle, err_buf, sizeof(err_buf));
            logger::error("[{}] Error : {}", get_name(), err_buf);
            return false;
        }
        else{
            unsigned long bitrate = parameters.value("can_bitrate", 500000);
            switch(bitrate){
                case 1000000: canSetBusParams(_can_handle, canBITRATE_1M, 0, 0, 0, 0, 0); break;
                case 500000: canSetBusParams(_can_handle, canBITRATE_500K, 0, 0, 0, 0, 0); break;
                case 250000: canSetBusParams(_can_handle, canBITRATE_250K, 0, 0, 0, 0, 0); break;
                case 125000: canSetBusParams(_can_handle, canBITRATE_125K, 0, 0, 0, 0, 0); break;
                case 100000: canSetBusParams(_can_handle, canBITRATE_100K, 0, 0, 0, 0, 0); break;
                case 62000: canSetBusParams(_can_handle, canBITRATE_62K, 0, 0, 0, 0, 0); break;
                case 50000: canSetBusParams(_can_handle, canBITRATE_50K, 0, 0, 0, 0, 0); break;
                case 83000: canSetBusParams(_can_handle, canBITRATE_83K, 0, 0, 0, 0, 0); break;
                case 10000: canSetBusParams(_can_handle, canBITRATE_10K, 0, 0, 0, 0, 0); break;
                default:
                    canSetBusParams(_can_handle, canBITRATE_1M, 0, 0, 0, 0, 0);
            }
            _status = canBusOn(_can_handle);
            if(_status!=canOK){
                char err[512] = {0,};
                canGetErrorText(_status, err, sizeof(err));
                logger::error("[{}] Failed to go bus ON : {}", get_name(), err);
                return false;
            }
            logger::debug("[{}] CAN channel {} opened & set BUS bitrate {} bps", get_name(), ch, bitrate);

            /* start channel 0 receiver */
            logger::info("[{}] Listen for CAN frames...", get_name());
            _can_ch0_rcv_worker = thread(&kvaser_can_interface::_can_ch0_rcv_task, this);
            
        }   
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void kvaser_can_interface::on_loop(){
  
        unsigned int id = 0x123;                   // Example CAN ID (Standard)
        unsigned char data[8] = {0};             // Data payload (8 bytes)
        unsigned int dlc = 8;                    // Data Length Code (number of bytes to send)
        unsigned int flags = canMSG_STD;         // Message flags (Standard ID)
        // Use canMSG_EXT for Extended ID

        // Prepare some changing data
        data[0] = 0xAA;
        data[1] = 0xBB;
        data[2] = 0x00; // Send loop counter in the 3rd byte
        data[3] = 0xCC;
        data[4] = 0xDD;
        data[5] = 0xEE;
        data[6] = 0xFF;
        data[7] = 0x11;

        logger::info("[{}] Sending CAN message, ID=0x{}, DLC={}", get_name(), id, dlc);

        // Send the message
        // canWrite() returns immediately after queuing the message.
        // canWriteWait() waits until the message is sent or a timeout occurs.
        // Using canWrite() for this example.
        canStatus stat = canWrite(_can_handle, id, data, dlc, flags);
        if (stat != canOK) {
            char err_buf[100];
            canGetErrorText(stat, err_buf, sizeof(err_buf));
            string str_err(err_buf, sizeof(err_buf));
            logger::error("[{}] Error sending CAN message: {}", get_name(), str_err);
        } else {
            logger::info("[{}] CAN message sent successfully", get_name());
        }
 
}


void kvaser_can_interface::on_close(){
    
    /* close CAN */
    canBusOff(_can_handle);
    canClose(_can_handle);
    canUnloadLibrary();


}

void kvaser_can_interface::on_message(){
    
}


void kvaser_can_interface::_can_ch0_rcv_task(){

    try{
        json tag;
        while(!_worker_stop.load()){
            long id;
            unsigned char data[8];
            unsigned int dlc;
            unsigned int flags;
            unsigned long time;

            canStatus stat = canRead(_can_handle, &id, data, &dlc, &flags, &time);
            if(stat==canOK) {
                std::ostringstream oss;
                for(unsigned char d:data){
                    oss << std::hex << std::uppercase << std::setfill('0') << std::setw(2) << static_cast<int>(d) << " ";
                }
                logger::debug("[{}] ID({}) | DLC({}) | Data({})", get_name(), id, dlc, oss.str());

                /* for sample */
                // if(id==(0x180+_node_id)){
                //     int16_t temperature = data[0] | (data[1] << 8);
                //     int16_t slope_z = data[2] | (data[3] << 8);
                //     int16_t slope_y = data[4] | (data[5] << 8);

                //     double slope_z_deg = static_cast<double>(slope_z)*resolution;
                //     double slope_y_deg = static_cast<double>(slope_y)*resolution;

                //     logger::info("[{}] Y({:.3f}), Z({:.3f}), Temp({})", get_name(), slope_y_deg, slope_z_deg, to_string(temperature));
                // }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));


        } /* end while */

        /* realse */
        canBusOff(_can_handle);
        canClose(_can_handle);
        canUnloadLibrary();
        logger::info("[{}] Close Device", get_name());
    }
    catch(const std::out_of_range& e){
        logger::error("[{}] Invalid parameter access", get_name());
    }
    catch(const zmq::error_t& e){
        logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
    }
    catch(const json::exception& e){
        logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
    }

}