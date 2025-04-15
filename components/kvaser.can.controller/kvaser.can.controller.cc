
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

        canInitializeLibrary();
        
        /* find channels */
        int n_channels = 0;
        _status = canGetNumberOfChannels(&n_channels);

        std::cout << "Found " << n_channels << " Kvaser CAN channel(s):" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(5) << "Ch"
              << std::setw(25) << "Device Name"
              << std::setw(15) << "Serial No."
              << std::setw(15) << "EAN/UPC"
              << std::setw(10) << "Ch On Card"
              << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    // 3. 각 채널 정보 가져오기 및 출력
    for (int i = 0; i < n_channels; ++i) {
        char deviceName[128] = {0};
        char eanUpcStr[20] = {0}; // EAN/UPC는 문자열로도 가져올 수 있음
        long serialNumber = 0;
        long long eanUpcNum = 0;
        int channelOnCard = 0;
        unsigned int devFlags = 0;

        // 채널 이름 가져오기
        _status = canGetChannelData(i, canCHANNELDATA_DEVDESCR_ASCII, deviceName, sizeof(deviceName));
        if (_status != canOK) {
           // 이름 가져오기 실패 시 대체 이름 사용 또는 에러 처리
           snprintf(deviceName, sizeof(deviceName), "Unknown/Error");
           // printCanlibError(_status); // 각 항목 실패 시 에러 출력 원하면 주석 해제
        }

        // 시리얼 번호 가져오기 (정수형)
        _status = canGetChannelData(i, canCHANNELDATA_CARD_SERIAL_NO, &serialNumber, sizeof(serialNumber));
         if (_status != canOK) {
            serialNumber = 0; // 실패 시 0으로 설정
            // printCanlibError(_status);
        }

        // EAN/UPC 번호 가져오기 (long long)
        // 주의: EAN/UPC는 하드웨어에 따라 없을 수도 있음
        _status = canGetChannelData(i, canCHANNELDATA_CARD_UPC_NO, &eanUpcNum, sizeof(eanUpcNum));
        if (_status != canOK) {
            eanUpcNum = 0; // 실패 시 0으로 설정
            // printCanlibError(_status);
            snprintf(eanUpcStr, sizeof(eanUpcStr), "N/A");
        } else {
            snprintf(eanUpcStr, sizeof(eanUpcStr), "%lld", eanUpcNum);
        }

        // 카드 상의 채널 번호 가져오기 (멀티 채널 장치용)
        _status = canGetChannelData(i, canCHANNELDATA_CHAN_NO_ON_CARD, &channelOnCard, sizeof(channelOnCard));
        if (_status != canOK) {
            channelOnCard = -1; // 실패 시 -1로 설정
            // printCanlibError(_status);
        }

        // 장치 플래그 가져오기 (가상 장치 여부 등 확인 가능)
        // _status = canGetChannelData(i, canCHANNELDATA_DEVICE_FLAGS, &devFlags, sizeof(devFlags));
        // if (_status == canOK && (devFlags & canDEVICE_FLAG_VIRTUAL)) {
        //     // 가상 장치임을 표시할 수 있음
        // }


        // 정보 출력
        std::cout << std::left << std::setw(5) << i
                  << std::setw(25) << deviceName
                  << std::setw(15) << serialNumber
                  << std::setw(15) << eanUpcStr
                  << std::setw(10) << channelOnCard
                  << std::endl;
    }

    std::cout << "------------------------------------------------------------------" << std::endl;

        
    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }

    return true;
}

void kvaser_can_controller::on_loop(){

    
}


void kvaser_can_controller::on_close(){
    
    /* close CAN */
    canClose(_handle);
    canUnloadLibrary();


}

void kvaser_can_controller::on_message(){
    
}
