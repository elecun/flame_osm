
#include "solectrix.camera.grabber.hpp"
#include <flame/log.hpp>

using namespace flame;
using namespace std;


/* create component instance */
static solectrix_camera_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new solectrix_camera_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool solectrix_camera_grabber::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* create device instance */
        vector<int> channels;
        if(parameters.contains("use_channel"))
            channels = parameters["use_channel"].get<vector<int>>();
        else{
            logger::warn("[{}] Cannot found 'use_channel' parameter. It must be defined.", get_name());
            return false;
        }
        
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

    
}


void solectrix_camera_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

}

void solectrix_camera_grabber::on_message(){
    
}

void solectrix_camera_grabber::_grab_task(json parameters){

    vector<int> channels = parameters["use_channel"].get<vector<int>>();
    _grabber = make_unique<sxpf_grabber>(channels);

}

