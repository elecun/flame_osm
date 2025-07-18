#include "video.file.grabber.hpp"
#include <flame/log.hpp>
#include <vector>
#include <list>
#include <string>

using namespace std;
using namespace cv;

void video_file_grabber::api_start_grab(const json& args){

    vector<string> requried = {"filepath"};
    for(const auto& arg:requried){
        if(!args.contains(arg)){
            logger::debug("[{}] Required arguments are missing...", get_name());
            break;
        }
    }

    /* start action thread */
    if(!_action_working.load()){
        _invoked_action_thread = thread(&video_file_grabber::_action_proc, this, args);
    }
    else{
        logger::warn("[{}] Already the Grab Action is performing... skip this invocation.", get_name());
    }

}

void video_file_grabber::api_stop_grab(const json& args){
    /* stop action working */
    _action_working.store(false);
    if(_invoked_action_thread.joinable()){
        _invoked_action_thread.join();
    }

    logger::debug("[{}] Grab Action is stoped by force", get_name());

}
