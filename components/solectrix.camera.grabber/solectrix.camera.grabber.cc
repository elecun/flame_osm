
#include "solectrix.camera.grabber.hpp"
#include <flame/log.hpp>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include "core_frame_processing.h"

using namespace flame;
using namespace std;
using namespace cv;


/* create component instance */
static solectrix_camera_grabber* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new solectrix_camera_grabber(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool solectrix_camera_grabber::onInit(){

    try{
        /* read profile */
        json parameters = getProfile()->parameters();

        /* check parameters */
        if(!parameters.contains("camera") || !parameters["camera"].is_array()){
            logger::warn("[{}] Not found or Invalid 'camera' parameters. It must be valid.", getName());
            return false;
        }
        logger::info("[{}] {} camera parameter is defined.", getName(), parameters["camera"].size());

        /* setup pipeline */
        _use_image_stream.store(parameters.value("use_image_stream", false));
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));

        /* device instance */
        _frame_grabber = make_unique<sxpf_grabber>(parameters);

        /* device initialization */
        if(!_frame_grabber->init()){
            logger::error("[{}] Failed to initialize frame grabber device", getName());
            return false;
        }

        /* device open */
        if(!_frame_grabber->open()){
            logger::error("[{}] Failed to open frame grabber device", getName());
            return false;
        }

        /* start worker */
        _grab_worker = thread(&solectrix_camera_grabber::_grab_task, this, parameters["camera"]);

    }
    catch(const std::exception& e){
        logger::error("[{}] Exception : {}", getName(), e.what());
        return false;
    }

    return true;
}

void solectrix_camera_grabber::onLoop(){
    /* nothing loop */
}

void solectrix_camera_grabber::onClose(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop grabbing thread */
    if(_grab_worker.joinable()){
        _grab_worker.join();
        logger::debug("[{}] grabber is now successfully stopped", getName());
    }

    /* device close */
    _frame_grabber->close();

}


void solectrix_camera_grabber::onData(flame::component::ZData& data){
    /* reserved function */
}

void solectrix_camera_grabber::_grab_task(json camera_parameters){

    auto last_time = chrono::high_resolution_clock::now();
    const auto& params = _frame_grabber->get_parameter_container();

    while(!_worker_stop.load()){
        auto loop_start = chrono::high_resolution_clock::now();

        /* do grab */
        try{
            pair<int, cv::Mat> captured_data = _frame_grabber->capture();
            int cam_id = captured_data.first;
            cv::Mat captured = captured_data.second;

            if (!captured.empty()) {
                /* find matching camera parameter by channel (cam_id) */
                string portname = "";
                string cam_name = "";

                for(const auto& p : params){
                    if(p.channel == cam_id){
                        portname = p.portname;
                        cam_name = p.name;
                        break;
                    }
                }

                if(portname.empty()){
                    logger::warn("[{}] Received frame from unknown cam_id: {}", getName(), cam_id);
                    continue;
                }

                /* image encoding and transmission */
                if(_use_image_stream.load()){
                    std::vector<unsigned char> serialized_image;
                    cv::imencode(".jpg", captured, serialized_image);

                    /* generate data meta tag */
                    json tag;
                    auto now = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = now - last_time;
                    last_time = now;

                    tag["fps"] = (elapsed.count() > 0) ? 1.0/elapsed.count() : 0.0;
                    tag["height"] = captured.rows;
                    tag["width"] = captured.cols;
                    tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
                    tag["cam_id"] = cam_id;

                    /* send data */
                    flame::component::ZData msg_multipart_image;
                    msg_multipart_image.addstr(portname);
                    msg_multipart_image.addstr(tag.dump());
                    msg_multipart_image.addmem(serialized_image.data(), serialized_image.size());
                    
                    /* publish monitoring image */
                    if(_use_image_stream_monitoring.load()){
                        string monitor_port = portname + "_monitor";
                        flame::component::ZData monitor_msg = msg_multipart_image.clone();
                        monitor_msg.pop(); // remove original portname topic
                        monitor_msg.pushstr(monitor_port); // push monitor_port as new topic
                        
                        if(!dispatch(monitor_port, monitor_msg)){
                            // logger::debug("[{}] monitoring port {} is not active", getName(), monitor_port);
                        }
                    }

                    if(!dispatch(portname, msg_multipart_image)){
                        logger::warn("[{}] socket handle is not valid for port {}", getName(), portname);
                    }
                    serialized_image.clear();
                }
            }
        }
        catch(const cv::Exception& e){
            logger::debug("[{}] CV Exception {}", getName(), e.what());
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] Piepeline Error : {}", getName(), e.what());
        }
        catch(const json::exception& e){
            logger::error("[{}] Data Parse Error : {}", getName(), e.what());
        }
        catch(const std::exception& e){
            logger::error("[{}] Standard Exception : {}", getName(), e.what());
        }

        auto loop_end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> loop_ms = loop_end - loop_start;
        logger::debug("[{}] Loop elapsed time: {:.2f} ms", getName(), loop_ms.count());
    
    }

    logger::debug("[{}] Stopped grab task..", getName());

}

bool solectrix_camera_grabber::open_device(int endpoint_id, int channel_id, uint32_t decode_csi2_datatype, int left_shift) {
    if (_device_opened) {
        logger::warn("[{}] Device already opened", getName());
        return true;
    }

    _endpoint_id = endpoint_id;
    _channel_id = channel_id;
    _decode_csi2_datatype = decode_csi2_datatype;
    _left_shift = left_shift;

    if(_frame_grabber->open()){
        _device_opened = true;
        logger::info("[{}] Device opened successfully (endpoint:{}, channel:{})", getName(), _endpoint_id, _channel_id);
        return true;
    }
    else {
        logger::error("[{}] Failed to initialize device", getName());
        return false;
    }
}

void solectrix_camera_grabber::close_device() {
    _frame_grabber->close();
    _device_opened = false;
    logger::info("[{}] Device closed", getName());
}

pair<int, Mat> solectrix_camera_grabber::grab() {
    if (!_device_opened) {
        logger::error("[{}] Device not opened. Call open_device() first", getName());
        return make_pair(-1, cv::Mat());
    }

    pair<int, Mat> result = _frame_grabber->capture();
    
    if(!result.second.empty()){
        logger::debug("[{}] Frame grabbed successfully from cam_id {}: {}x{}, channels: {}", getName(), result.first, result.second.cols, result.second.rows, result.second.channels());
    }
    else {
        logger::debug("[{}] No frame available");
    }

    return result;
}
