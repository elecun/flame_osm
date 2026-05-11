
#include "kps.holistic.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static kps_holistic_inference* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new kps_holistic_inference(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool kps_holistic_inference::onInit(){

    try{

        /* read profile */
        json parameters = getProfile()->parameters();

        

        logger::info("[{}] Initialized successfully", getName());

    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }
    catch(std::exception& e){
        logger::error("[{}] Initialization Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void kps_holistic_inference::onLoop(){
  
        
 
}


void kps_holistic_inference::onClose(){
    try{
        
        logger::info("[{}] Closed successfully", getName());
    }
    catch(std::exception& e){
        logger::error("[{}] Close Error : {}", getName(), e.what());
    }
}

void kps_holistic_inference::onData(flame::component::ZData& data){
    // Note: The 'data' parameter is currently unused.
}
