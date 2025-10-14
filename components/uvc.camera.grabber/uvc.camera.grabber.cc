
#include "uvc.camera.grabber.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>

using namespace flame;
using namespace std;
using namespace cv;

/* create component instance */
static uvc_camera_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new uvc_camera_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool uvc_camera_grabber::on_init(){

    try{
        /* read profile */
        json parameters = get_profile()->parameters();

        /* set video capture instance */
        int auto_id = 1;
        if(parameters.contains("camera")){
            for(auto& dev:parameters["camera"]){
                int id = dev.value("id", auto_id++);
                dev["id"] = id; /* update camera id */

                /* assign grabber worker */
                _grab_worker[id] = thread(&uvc_camera_grabber::_grab_task, this, id, dev);        
            }
        }
        else {
            logger::warn("[{}] Cannot found camera(s) available", get_name());
            return false;
        }

        /* configure data for data pipelining  */
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));
        _use_image_stream.store(parameters.value("use_image_stream", false));
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

void uvc_camera_grabber::on_loop(){

    
}


void uvc_camera_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop grabbing */
    for_each(_grab_worker.begin(), _grab_worker.end(), [](auto& t) {
        if(t.second.joinable()){
            t.second.join();
            logger::debug("Camera #{} grabber is successfully stopped", t.first);
        }
    });
    _grab_worker.clear();

}

void uvc_camera_grabber::on_message(const message_t& msg){
    // Note: The 'msg' parameter is currently unused.
}

void uvc_camera_grabber::_grab_task(int camera_id, json camera_param){

    string device = camera_param.value("device", "");

    if(camera_id<0 || device.empty()){
        logger::warn("[{}] Undefined or Invalid Camera Configuration in the Component Profile.", get_name());
        return;
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
