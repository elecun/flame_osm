#include "camera.monitor.hpp"
#include <flame/log.hpp>

/* create component instance */
static camera_monitor* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new camera_monitor(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

camera_monitor::camera_monitor() {
}

bool camera_monitor::onInit(){
    return true;
}

void camera_monitor::onLoop(){
}

void camera_monitor::onClose(){
}

void camera_monitor::onData(flame::component::ZData& data){
    try {
        string portname = data.from;
        logger::info("[{}] Received data from port: {}, frames count: {}", getName(), portname, data.size());
    }
    catch(const std::exception& e){
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}
