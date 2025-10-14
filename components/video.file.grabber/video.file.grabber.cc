
#include "video.file.grabber.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>

using namespace flame;
using namespace std;

/* create component instance */
static video_file_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new video_file_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool video_file_grabber::on_init(){

    try{
        /* read profile */
        json parameters = get_profile()->parameters();

        /* configure data for data pipelining  */
        _use_image_stream.store(parameters.value("use_image_stream", false));

        /* start api processing from action invoker */
        _action_invoke_listener = thread(&video_file_grabber::_action_invoke_listener_proc, this, parameters);

        /* rpc-like api registration */
        api_table["api_start_grab"] = [this](const json& args){ this->api_start_grab(args); };
        api_table["api_stop_grab"] = [this](const json& args){ this->api_stop_grab(args); };
    }
    catch(json::exception& e){
        logger::error("[{}] Component profile read exception : {}", get_name(), e.what());
        return false;
    }
    catch(cv::Exception::exception& e){
        logger::error("[{}] Device open exception : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void video_file_grabber::on_loop(){

    
}


void video_file_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop action invoke listener */
    if(_action_invoke_listener.joinable()){
        _action_invoke_listener.join();
        logger::debug("[{}] Grabber is successfully stopped", get_name());
    }

    /* stop grab action */
    _action_working.store(false);
    if(_invoked_action_thread.joinable()){
        _invoked_action_thread.join();
    }
}

void video_file_grabber::on_message(const message_t& msg){
    // Note: The 'msg' parameter is currently unused.
}

void video_file_grabber::_action_proc(json args){

    string video_file = args["filepath"].get<string>();
    int api_ref = static_cast<int>(CAP_FFMPEG);
    VideoCapture _cap(video_file, api_ref);

    /* check video capture */
    if(!_cap.isOpened()){
        logger::error("[{}] {} could not open", get_name(), video_file);
        return;
    }

    /* param */
    const string image_stream_port = "image_stream";
    const string image_stream_monitor_port = "image_stream_monitor";

    auto last_time = chrono::high_resolution_clock::now();

    /* loop action */
    _action_working.store(true);
    logger::debug("[{}] Action is now performing... ", get_name());
    while(_action_working.load()){
        try{
            
            /* capture raw frame */
            Mat raw_frame;
            _cap >> raw_frame;
            if(raw_frame.empty()){
                logger::warn("[{}] Grabbed video frame is empty", get_name());
                continue;
            }

            /* tags */
            json tag;
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - last_time;
            last_time = now;
            tag["fps"] = 1.0/elapsed.count();
            tag["height"] = raw_frame.rows;
            tag["width"] = raw_frame.cols;
            tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();

            /* send image */
            if(_use_image_stream.load()){

                /* image encoding */
                std::vector<unsigned char> serialized_image;
                cv::imencode(".jpg", raw_frame, serialized_image);

                if(get_port(image_stream_port)->handle()!=nullptr){
                    zmq::multipart_t msg_multipart_image;
                    msg_multipart_image.addstr(tag.dump());
                    msg_multipart_image.addmem(serialized_image.data(), serialized_image.size());
                    msg_multipart_image.send(*get_port(image_stream_port), ZMQ_DONTWAIT);
                    msg_multipart_image.clear();
                }
                else{
                    logger::warn("[{}] socket handle is not valid ", get_name());
                }
                serialized_image.clear();

            }
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
        }
    }

    /* final */
    _cap.release();
}


void video_file_grabber::_action_invoke_listener_proc(json paramters){

    logger::debug("[{}] Action invoke listener is now working...", get_name());
    while(!_worker_stop.load()){
        try{

            string portname = "action_invoke";
            zmq::multipart_t msg_multipart;
            bool success = msg_multipart.recv(*get_port(portname));
            if(success){
                string function = msg_multipart.popstr();
                string str_args = msg_multipart.popstr();
                auto json_args = json::parse(str_args);

                if(api_table.find(function) != api_table.end()){
                    api_table[function](json_args);
                }
                else {
                    logger::debug("[{}] '{}' does not support  ", get_name(), function);
                }
            }
        }
        catch(const cv::Exception& e){
            logger::error("[{}] CV Exception : {}", get_name(), e.err);
            logger::debug("[{}] {}", get_name(), e.what());
        }
        catch(const std::out_of_range& e){
            logger::error("[{}] Invalid parameter access", get_name());
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
        }
        catch(const json::exception& e){
            logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
        }

    }

}
