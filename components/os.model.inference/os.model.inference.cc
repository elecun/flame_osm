
#include "os.model.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static os_model_inference* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new os_model_inference(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool os_model_inference::onInit(){

    try{

        /* read profile */
        json parameters = getProfile()->parameters();

        
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void os_model_inference::onLoop(){
  
        
 
}


void os_model_inference::onClose(){
    



}

void os_model_inference::onData(flame::component::ZData& data){
    
}
