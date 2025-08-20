
#include "solectrix.camera.grabber.hpp"
#include <flame/log.hpp>

using namespace flame;
using namespace std;
using namespace cv;


/* create component instance */
static solectrix_camera_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new solectrix_camera_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool solectrix_camera_grabber::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* check parameters */
        if(!parameters.contains("camera") || !parameters["camera"].is_array()){
            logger::warn("[{}] Not found or Invalid 'camera' parameters. It must be valid.", get_name());
            return false;
        }
        logger::info("[{}] {} camera parameter is defined.", get_name(), parameters["camera"].size());

        /* setup pipeline */
        _use_image_stream.store(parameters.value("use_image_stream", false));
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));
        
        /* start grab task on thread */
        _grab_worker = thread(&solectrix_camera_grabber::_grab_task, this, parameters);
        
    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }

    return true;
}

void solectrix_camera_grabber::on_loop(){
    /* nothing loop */
    
}


void solectrix_camera_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop grabbing thread */
    if(_grab_worker.joinable()){
        _grab_worker.join();
        logger::debug("[{}] grabber is now successfully stopped", get_name());
    }

}

void solectrix_camera_grabber::on_message(){
    /* reserved function */
}

void solectrix_camera_grabber::_grab_task(json parameters){

    /* read channels for each camera */
    vector<int> channels;
    for(const auto& item: parameters["camera"]){
        if(item.contains("channel")){
            channels.push_back(item["channel"].get<int>());
        }
    }

    /* read port configurations */
    // string monitoring_portname = fmt::format("image_stream_monitor_{}", camera_id);
    // string stream_portname = fmt::format("image_stream_{}", camera_id);

    /* create grabber instance */
    unique_ptr<sxpf_grabber> _grabber = make_unique<sxpf_grabber>(channels, 1920, 1080);
    _grabber->open();


    int i = 0;
    while(!_worker_stop.load()){

        /* do grab */
        try{
            cv::Mat captured = _grabber->capture();
            cv::imwrite(fmt::format("./grabbed_{}.png", i++), captured);
        }
        catch(const cv::Exception& e){
            logger::debug("[{}] CV Exception {}", get_name(), e.what());
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
        }
        catch(const json::exception& e){
            logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
        }
        catch(const std::exception& e){
            logger::error("[{}] Standard Exception : {}", get_name(), e.what());
        }
    
    }

    /* close grabber instance */
    _grabber->close();

    logger::debug("[{}] Stopped grab task..", get_name());

    

}

