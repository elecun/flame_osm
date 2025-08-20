
#include "os.model.inference.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static os_model_inference* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new os_model_inference(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool os_model_inference::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void os_model_inference::on_loop(){
  
        
 
}


void os_model_inference::on_close(){
    



}

void os_model_inference::on_message(){
    
}
