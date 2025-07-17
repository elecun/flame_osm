
#include "video.file.grabber.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>

using namespace flame;
using namespace std;
using namespace cv;

/* create component instance */
static video_file_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new video_file_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool video_file_grabber::on_init(){

    try{
        /* read profile */
        json parameters = get_profile()->parameters();

        /* start api processing from action invoker */
        _action_invoke_listener = thread(&video_file_grabber::_action_invoke_listener_proc, this, parameters);

        /* configure data for data pipelining  */
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));
        _use_image_stream.store(parameters.value("use_image_stream", false));

        /* api registration */
        api_table["api_load_video"] = [this](const json& args){ this->api_load_video(args); };
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

    /* stop grabbing */
    if(_action_invoke_listener.joinable()){
        _action_invoke_listener.join();
        logger::debug("[{}] Grabber is successfully stopped", get_name());
    }

}

void video_file_grabber::on_message(){
    
}

void video_file_grabber::api_load_video(const json& args){
    if(args.contains("filepath")){
        string video_file = args.value("filepath", "");
        logger::debug("[{}] File Path : {}", get_name(), video_file);

        if(_cap.isOpened()){
            _cap.release();
        }

        _cap.open(video_file);
    }
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
                    logger::debug("[{}] '{}' does not support  ", get_name());
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





    cv::VideoCapture _cap; /* camera capture */
    try{

        /* camera open */
        _cap.open(device, CAP_V4L2); /* for linux only */
        _cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        if(!_cap.isOpened()){
            CV_Error(cv::Error::StsError, fmt::format("Camera #{} cannot be opened.", camera_id));
        }

        /* read port configurations */
        string monitoring_portname = fmt::format("image_stream_monitor_{}", camera_id);
        string stream_portname = fmt::format("image_stream_{}", camera_id);

        json dataport_config = get_profile()->dataport();
        int monitoring_width = dataport_config.at(monitoring_portname).at("resolution").value("width", 480);
        int monitoring_height = dataport_config.at(monitoring_portname).at("resolution").value("height", 270);

        json tag;
        auto last_time = chrono::high_resolution_clock::now();
        logger::debug("[{}] Camera #{} grabbing is now working...", get_name(), camera_id);
        while(!_worker_stop.load()){

            /* capture from camera */
            Mat raw_frame;
            _cap >> raw_frame;
            if(raw_frame.empty()){
                logger::warn("[{}] Camera #{}({}) frame is empty", get_name(), camera_id, device);
                continue;
            }

            /* generate tag */
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - last_time;
            last_time = now;
            tag["fps"] = 1.0/elapsed.count();
            tag["camera_id"] = camera_id;
            tag["height"] = raw_frame.rows;
            tag["width"] = raw_frame.cols;
            tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();

            /* transfer original image to processs */
            if(_use_image_stream.load()){
                /* image encoding */
                std::vector<unsigned char> serialized_image;
                cv::imencode(".jpg", raw_frame, serialized_image);

                if(get_port(stream_portname)->handle()!=nullptr){
                    zmq::multipart_t msg_multipart_image;
                    msg_multipart_image.addstr(tag.dump());
                    msg_multipart_image.addmem(serialized_image.data(), serialized_image.size());
                    msg_multipart_image.send(*get_port(stream_portname), ZMQ_DONTWAIT);
                    msg_multipart_image.clear();
                }
                else{
                    logger::warn("[{}] {} socket handle is not valid ", get_name(), camera_id);
                }
                serialized_image.clear();
            }

            /* transfer small image for monitoring */
            if(_use_image_stream_monitoring.load()){
                
                cv::Mat monitor_image;
                cv::resize(raw_frame, monitor_image, cv::Size(monitoring_width, monitoring_height));
                std::vector<unsigned char> serialized_monitor_image;
                cv::imencode(".jpg", monitor_image, serialized_monitor_image);

                if(get_port(monitoring_portname)->handle()!=nullptr){
                    zmq::multipart_t msg_multipart;
                    msg_multipart.addstr(fmt::format("{}/image_stream_monitor_{}",get_name(), camera_id));
                    msg_multipart.addstr(tag.dump());
                    msg_multipart.addmem(serialized_monitor_image.data(), serialized_monitor_image.size());
                    msg_multipart.send(*get_port(monitoring_portname), ZMQ_DONTWAIT);
                    msg_multipart.clear();
                }
                serialized_monitor_image.clear();
            }

        } /* end while */

        /* realse */
        _cap.release();
        logger::info("[{}] Camera #{}({}) is released", get_name(), camera_id, device);
    }
    catch(const cv::Exception& e){
        logger::error("[{}] Camera #{} CV Exception : {}", get_name(), camera_id, e.err);
        logger::debug("[{}] {}", get_name(), e.what());
        _cap.release();
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
