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
    /* 
       Note: ZData for SUB sockets has the topic frame stripped by the engine.
       We need to identify the camera channel from the metadata tag.
    */
    try {
        if(data.size() >= 1){
            string tag_str = data.popstr();
            json tag = json::parse(tag_str);

            int cam_channel = tag.value("cam_channel", -1);
            if(cam_channel == -1) cam_channel = tag.value("cam_id", -1);

            if(cam_channel != -1){
                string channel_key = fmt::format("channel_{}", cam_channel);
                auto now = chrono::high_resolution_clock::now();

                lock_guard<mutex> lock(_mtx);
                if(_last_received_times.contains(channel_key)){
                    chrono::duration<double> elapsed = now - _last_received_times[channel_key];
                    double fps = (elapsed.count() > 0) ? 1.0 / elapsed.count() : 0.0;
                    logger::info("[{}] Received data from channel {}: fps={:.2f}", getName(), cam_channel, fps);
                }
                _last_received_times[channel_key] = now;
            }
        }
    }
    catch(const std::exception& e){
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}
