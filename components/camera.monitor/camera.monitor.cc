#include "camera.monitor.hpp"
#include <flame/log.hpp>

/* create component instance */
static camera_monitor* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new camera_monitor(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

camera_monitor::camera_monitor() {
}

bool camera_monitor::onInit(){
    try{
        /* read profile */
        json dataport = getProfile()->dataPort();

        // for(auto& [portname, config] : dataport.items()){
        //     if(portname == "status") continue;

        //     /* Check if it's a subscriber type (socket_type: sub) */
        //     string socket_type = config.value("socket_type", "");
        //     if(socket_type == "sub"){
        //         logger::info("[{}] Setting up callback for port: {}", getName(), portname);
                
        //         auto port = getPort(portname);
        //         if(port){
        //             _last_received_times[portname] = chrono::high_resolution_clock::now();
        //             port->setMessageCallback([this, portname](flame::component::ZData& data){
        //                 this->_on_message(portname, data);
        //             });
        //         }
        //     }
        // }
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }
    catch(const std::exception& e){
        logger::error("[{}] Initialization Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void camera_monitor::onLoop(){
}

void camera_monitor::onClose(){
    logger::info("[{}] Component closing", getName());
}

void camera_monitor::onData(flame::component::ZData& data){

    logger::info("[{}] Received data with {} parts", getName(), data.size());
}

void camera_monitor::_on_message(const string& portname, flame::component::ZData& data){
    auto now = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = now - _last_received_times[portname];
    _last_received_times[portname] = now;

    double fps = (elapsed.count() > 0) ? 1.0 / elapsed.count() : 0.0;
    
    /* ZData received here already has topic stripped for SUB pattern */
    if(data.size() >= 1){
        string tag_str = data.popstr();
        // zmq::message_t image_data = data.pop();
        
        logger::info("[{}] Received data from {}: fps={:.2f}", getName(), portname, fps);
    }
}