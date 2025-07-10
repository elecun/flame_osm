
#include "kvaser.can.controller.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static kvaser_can_controller* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new kvaser_can_controller(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool kvaser_can_controller::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* CAN device initialization */
        canInitializeLibrary();
        
        /* find channels */
        canStatus _status { canOK };
        _status = canGetNumberOfChannels(&_can_channels);
        if(_status != canOK){
            logger::error("[{}] Cannot find CAN channel. Error code : {}", get_name(), (int)_status);
            return false;
        }
        logger::info("[{}] Found {} CAN channel(s) ", get_name(), _can_channels);

        /* show channel info. */
        logger::info("---------------------------------------------------------------------");
        for(int i=0; i<_can_channels; i++){
            char device_name[128] = {0};
            int ch_on_card = 0;

            // read channel name
            _status = canGetChannelData(i, canCHANNELDATA_DEVDESCR_ASCII, device_name, sizeof(device_name));
            if (_status != canOK) {
                logger::warn("Channel {} name retrieval failed. Error code : {}", i, (int)_status);
            }
            string str_device_name(device_name);

            // read can channel
            _status = canGetChannelData(i, canCHANNELDATA_CHAN_NO_ON_CARD, &ch_on_card, sizeof(ch_on_card));
            if (_status != canOK) {
                ch_on_card = -1;
                logger::warn("Channel {} on card retrieval failed. Error code : {}", i, (int)_status);
            }
        
            logger::info("CAN Channel {}: Device Name: {}\tDevice Channel: {}", i, device_name, ch_on_card);
   
        }
        logger::info("---------------------------------------------------------------------");

        /* use channel */
        if(parameters.contains("use_channel")){
            int ch = parameters.value("use_channel", 0);
            _handle = canOpenChannel(ch, canOPEN_ACCEPT_VIRTUAL);
            if(_handle<0){
                logger::error("[{}] Cannot open channel {}. Error code : {}", get_name(), ch, (int)_handle);

                char err_buf[100];
                canGetErrorText((canStatus)_handle, err_buf, sizeof(err_buf));
                logger::error("[{}] Error : {}", get_name(), err_buf);

                return false;
            }
            else{
                unsigned long bitrate = parameters.value("can_bitrate", 1000000);
                switch(bitrate){
                    case 1000000: canSetBusParams(_handle, canBITRATE_1M, 0, 0, 0, 0, 0); break;
                    case 500000: canSetBusParams(_handle, canBITRATE_500K, 0, 0, 0, 0, 0); break;
                    case 250000: canSetBusParams(_handle, canBITRATE_250K, 0, 0, 0, 0, 0); break;
                    case 125000: canSetBusParams(_handle, canBITRATE_125K, 0, 0, 0, 0, 0); break;
                    case 100000: canSetBusParams(_handle, canBITRATE_100K, 0, 0, 0, 0, 0); break;
                    case 62000: canSetBusParams(_handle, canBITRATE_62K, 0, 0, 0, 0, 0); break;
                    case 50000: canSetBusParams(_handle, canBITRATE_50K, 0, 0, 0, 0, 0); break;
                    case 83000: canSetBusParams(_handle, canBITRATE_83K, 0, 0, 0, 0, 0); break;
                    case 10000: canSetBusParams(_handle, canBITRATE_10K, 0, 0, 0, 0, 0); break;
                    default:
                        canSetBusParams(_handle, canBITRATE_1M, 0, 0, 0, 0, 0);
                }
                _status = canBusOn(_handle);
                if(_status<0){
                    logger::error("[{}] Cannot set channel {} to bus on. Error code : {}", get_name(), ch, (int)_status);
                    return false;
                }
                else{
                    logger::info("[{}] CAN channel {} set to bus on", get_name(), ch);
                }

                logger::info("[{}] CAN channel {} opened & set BUS bitrate {} bps", get_name(), ch, bitrate);
                
            }
        }
        
        
    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }

    return true;
}

void kvaser_can_controller::on_loop(){

  
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
        canStatus stat = canWrite(_handle, id, data, dlc, flags);
        if (stat != canOK) {
            char err_buf[100];
            canGetErrorText(stat, err_buf, sizeof(err_buf));
            string str_err(err_buf, sizeof(err_buf));
            logger::error("[{}] Error sending CAN message: {}", get_name(), str_err);
        } else {
            logger::info("[{}] CAN message sent successfully", get_name());
        }
 
}


void kvaser_can_controller::on_close(){
    
    /* close CAN */
    canBusOff(_handle);
    canClose(_handle);
    canUnloadLibrary();


}

void kvaser_can_controller::on_message(){
    
}
