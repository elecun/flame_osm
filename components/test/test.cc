
#include "test.hpp"

using namespace flame;
using namespace std;

/* create component instance */
static test* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new test(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool test::on_init(){

    try{
        /* read profile */
        json parameters = get_profile()->parameters();

    }
    catch(json::exception& e){
        logger::error("[{}] Component profile read exception : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void test::on_loop(){

    logger::info("test::on_loop()");
}


void test::on_close(){

    
}

void test::on_message(){
    
}
