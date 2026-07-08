#include "osm.monolithic.inference.hpp"
#include <flame/log.hpp>

/* create component instance */
static osm_monolithic_inference* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new osm_monolithic_inference(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

osm_monolithic_inference::osm_monolithic_inference() {
}

bool osm_monolithic_inference::onInit(){
    try{
        json parameters = getProfile()->parameters();
        logger::info("[{}] Initialized osm.monolithic.inference component", getName());
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

void osm_monolithic_inference::onLoop(){
}

void osm_monolithic_inference::onClose(){
    logger::info("[{}] Closing osm.monolithic.inference component", getName());
}

void osm_monolithic_inference::onData(flame::component::ZData& data){
    try {
        if (data.size() >= 3) {
            std::string portname = data.popstr();
            std::string tag_str = data.popstr();
            zmq::message_t image_msg = data.pop();

            if (portname == "image_stream_1" || portname == "image_stream_2") {
                json tag = json::parse(tag_str);
                int height = tag["height"].get<int>();
                int width = tag["width"].get<int>();
                int type = tag["type"].get<int>();

                // Restore image Mat from payload
                cv::Mat raw_img(height, width, type, image_msg.data());
                cv::Mat cloned_img = raw_img.clone();

                if (portname == "image_stream_1") {
                    std::lock_guard<std::mutex> lock(_img_mutex_1);
                    _latest_image_1 = cloned_img;
                } else if (portname == "image_stream_2") {
                    std::lock_guard<std::mutex> lock(_img_mutex_2);
                    _latest_image_2 = cloned_img;
                }
            }
        }
    }
    catch (const std::exception& e) {
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}

cv::Mat osm_monolithic_inference::getLatestImage1() {
    std::lock_guard<std::mutex> lock(_img_mutex_1);
    return _latest_image_1.clone();
}

cv::Mat osm_monolithic_inference::getLatestImage2() {
    std::lock_guard<std::mutex> lock(_img_mutex_2);
    return _latest_image_2.clone();
}
