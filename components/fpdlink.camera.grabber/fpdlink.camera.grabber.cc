
#include "fpdlink.camera.grabber.hpp"
#include <flame/log.hpp>

using namespace flame;
using namespace std;


/* create component instance */
static fpdlink_camera_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new fpdlink_camera_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool fpdlink_camera_grabber::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        
    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }

    return true;
}

void fpdlink_camera_grabber::on_loop(){

    
}


void fpdlink_camera_grabber::on_close(){

    

}

void fpdlink_camera_grabber::on_message(){
    
}

