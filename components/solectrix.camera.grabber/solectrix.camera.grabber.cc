
#include "solectrix.camera.grabber.hpp"
#include <flame/log.hpp>
#include <fcntl.h>
#include <errno.h>
#include "core_frame_processing.h"

using namespace flame;
using namespace std;
using namespace cv;


/* create component instance */
static solectrix_camera_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new solectrix_camera_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool solectrix_camera_grabber::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* check parameters */
        if(!parameters.contains("camera") || !parameters["camera"].is_array()){
            logger::warn("[{}] Not found or Invalid 'camera' parameters. It must be valid.", get_name());
            return false;
        }
        logger::info("[{}] {} camera parameter is defined.", get_name(), parameters["camera"].size());

        /* setup pipeline */
        _use_image_stream.store(parameters.value("use_image_stream", false));
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));

        /* device instance */
        json camera_parameters = parameters["camera"];
        _grabber_handle = make_unique<sxpf_grabber>(parameters);

        /* device open */
        if(_grabber_handle->open()){

            /* start grab task on thread */
            _grab_worker = thread(&solectrix_camera_grabber::_grab_task, this, camera_parameters);

        }

    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }

    return true;
}

void solectrix_camera_grabber::on_loop(){
    /* nothing loop */
}


void solectrix_camera_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop grabbing thread */
    if(_grab_worker.joinable()){
        _grab_worker.join();
        logger::debug("[{}] grabber is now successfully stopped", get_name());
    }

    /* device close */
    _grabber_handle->close();

}

void solectrix_camera_grabber::on_message(const message_t& msg){
    /* reserved function */
}

void solectrix_camera_grabber::_grab_task(json parameters){

    while(!_worker_stop.load()){

        /* do grab */
        try{
            cv::Mat captured = _grabber_handle->capture();
            if (!captured.empty()) {
                logger::debug("[{}] Captured image: {}x{}, channels: {}", get_name(), captured.cols, captured.rows, captured.channels());
            }
        }
        catch(const cv::Exception& e){
            logger::debug("[{}] CV Exception {}", get_name(), e.what());
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
        }
        catch(const json::exception& e){
            logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
        }
        catch(const std::exception& e){
            logger::error("[{}] Standard Exception : {}", get_name(), e.what());
        }
    
    }

    logger::debug("[{}] Stopped grab task..", get_name());

}

bool solectrix_camera_grabber::open_device(int endpoint_id, int channel_id, uint32_t decode_csi2_datatype, int left_shift) {
    if (_device_opened) {
        logger::warn("[{}] Device already opened", get_name());
        return true;
    }
    
    _endpoint_id = endpoint_id;
    _channel_id = channel_id;
    _decode_csi2_datatype = decode_csi2_datatype;
    _left_shift = left_shift;
    
    // Use simplified device initialization
    bool success = initialize_device(_endpoint_id, _channel_id, &_fg, &_devfd);
    
    if (success) {
        _device_opened = true;
        logger::info("[{}] Device opened successfully (endpoint:{}, channel:{})", get_name(), _endpoint_id, _channel_id);
    } else {
        logger::error("[{}] Failed to initialize device", get_name());
    }
    
    return success;
}

void solectrix_camera_grabber::close_device() {
    if (!_device_opened || !_fg) {
        return;
    }
    
    // Use simplified cleanup
    cleanup_device(_fg);
    
    _fg = 0;
    _devfd = 0;
    _device_opened = false;
    
    logger::info("[{}] Device closed", get_name());
}

cv::Mat solectrix_camera_grabber::grab() {
    if (!_device_opened || !_fg) {
        logger::error("[{}] Device not opened. Call open_device() first", get_name());
        return cv::Mat();
    }
    
    cv::Mat result;
    
    // Use simplified wait and process function
    bool success = wait_and_process_frame(_fg, _devfd, result);
    
    if (success && !result.empty()) {
        logger::debug("[{}] Frame grabbed successfully: {}x{}, channels: {}", get_name(), result.cols, result.rows, result.channels());
    } else {
        logger::debug("[{}] No frame available", get_name());
    }
    
    return result;
}
