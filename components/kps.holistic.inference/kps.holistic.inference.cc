
#include "kps.holistic.inference"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static headpose_model_inference* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new headpose_model_inference(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool headpose_model_inference::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        

        logger::info("[{}] Initialized successfully", get_name());

    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", get_name(), e.what());
        return false;
    }
    catch(std::exception& e){
        logger::error("[{}] Initialization Error : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void headpose_model_inference::on_loop(){
  
        
 
}


void headpose_model_inference::on_close(){
    try{
        
        logger::info("[{}] Closed successfully", get_name());
    }
    catch(std::exception& e){
        logger::error("[{}] Close Error : {}", get_name(), e.what());
    }
}

void headpose_model_inference::on_message(const message_t& msg){
    // Note: The 'msg' parameter is currently unused.
}
