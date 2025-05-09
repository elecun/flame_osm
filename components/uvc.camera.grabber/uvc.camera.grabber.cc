
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
        if(parameters.contains("cameras")){
            for(auto& dev:parameters["cameras"]){
                string device = dev["device"].get<string>();
                int id = dev["id"].get<int>();

                /* assign grabber worker */
                _grab_worker[id] = thread(&uvc_camera_grabber::_grab_task, this, id, device);
            }
        }

        /* realtime monitoring configure */
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));
        _use_image_stream.store(parameters.value("use_image_stream", false));

        logger::info("[{}] Use image stream monitoring : {}", get_name(), _use_image_stream_monitoring.load());
        logger::info("[{}] Use image stream : {}", get_name(), _use_image_stream.load());
    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }
    catch(cv::Exception::exception& e){
        logger::error("Device Error : {}", e.what());
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
            logger::info("Camera #{} is stopped", t.first);
        }
    });
    _grab_worker.clear();

}

void uvc_camera_grabber::on_message(){
    
}

void uvc_camera_grabber::_grab_task(int camera_id, string device){
    try{

        cv::VideoCapture _cap(device, CAP_V4L2); //for linux
        _cap.set(cv::CAP_PROP_BUFFERSIZE, 1); //minial buffer size
        if(!_cap.isOpened()){
            logger::error("[{}] Camera #{}({}) cannot be opened. please check the device.", get_name(), camera_id, device);
            return;
        }

        /* read port configurations */
        string monitoring_portname = fmt::format("image_stream_monitor_{}", camera_id);
        string stream_portname = fmt::format("image_stream_{}", camera_id);

        json dataport_config = get_profile()->dataport();
        string id_str = fmt::format("{}", camera_id);

        int monitoring_width = 0;
        int monitoring_height = 0;
        string monitoring_topic {""};
        try{
            if(dataport_config.contains(monitoring_portname)){

                monitoring_width = dataport_config.at(monitoring_portname).at("resolution").value("width", 640);
                monitoring_height = dataport_config.at(monitoring_portname).at("resolution").value("height", 480);
                monitoring_topic = fmt::format("{}/{}", get_name(), monitoring_portname);
                logger::info("[{}] Camera #{} monitoring image resolution : {}x{}", get_name(), camera_id, monitoring_width, monitoring_height);
            }
        }
        catch(const json::exception& e){
            logger::error("[{}] Camera #{}({}) monitoring image resolution error : {}", get_name(), camera_id, device, e.what());
        }

        /* grab */
        json tag;
        auto last_time = chrono::high_resolution_clock::now();
        while(!_worker_stop.load()){

            Mat raw_frame;
            _cap >> raw_frame;
            if(raw_frame.empty()){
                logger::warn("[{}] Camera #{}({}) frame is empty", get_name(), camera_id, device);
                continue;
            }

            /* generate captrued image tags */
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - last_time;
            last_time = now;
            tag["fps"] = 1.0/elapsed.count();
            tag["camera_id"] = camera_id;
            tag["height"] = raw_frame.rows;
            tag["width"] = raw_frame.cols;
            tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();

            /* send image to pipeline */
            if(_use_image_stream.load()){

                /* encode to jpg */
                std::vector<unsigned char> encoded_image;
                cv::imencode(".jpg", raw_frame, encoded_image);

                // data message
                pipe_data msg_image(encoded_image.data(), encoded_image.size());

                // push message
                zmq::multipart_t msg_multipart;
                msg_multipart.addstr(tag.dump());
                msg_multipart.addmem(msg_image.data(), msg_image.size());
                msg_multipart.send(*get_port(stream_portname), ZMQ_DONTWAIT);
            }

            /* send monitoring image to pipeline */
            if(_use_image_stream_monitoring.load()){

                // generate data message
                cv::Mat monitor_image;
                cv::resize(raw_frame, monitor_image, cv::Size(monitoring_width, monitoring_height));
                std::vector<unsigned char> encoded_monitor_image;;
                cv::imencode(".jpg", monitor_image, encoded_monitor_image);
                pipe_data msg_monitor_image(encoded_monitor_image.data(), encoded_monitor_image.size());

                // generate multipart message
                zmq::multipart_t msg_multipart;
                msg_multipart.addstr(monitoring_topic);
                msg_multipart.addstr(tag.dump());
                msg_multipart.addmem(msg_monitor_image.data(), msg_monitor_image.size());

                // publish message
                msg_multipart.send(*get_port(monitoring_portname), ZMQ_DONTWAIT);
            }


        }

        /* realse */
        _cap.release();
        logger::info("[{}] Camera #{}({}) is released", get_name(), camera_id, device);
        
    }
    catch(const cv::Exception::exception& e){
        logger::error("[{}] Camera #{} grabber has an error while grabbing", get_name(), camera_id);
    }
    catch(const zmq::error_t& e){
        logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
    }
    catch(const json::exception& e){
        logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
    }

}
