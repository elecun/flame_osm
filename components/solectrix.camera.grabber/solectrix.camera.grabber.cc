
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

    while(!_worker_stop.load()){

        /* do grab */
        try{
            cv::Mat captured = _frame_grabber->capture();
            if (!captured.empty()) {
                logger::debug("[{}] Captured image: {}x{}, channels: {}", getName(), captured.cols, captured.rows, captured.channels());

                /* image rotate (0=cw_90, 1=180, 2=ccw_90)*/
                int rotate_flag = -1;
                string portname = "";
                if(camera_parameters.is_array() && !camera_parameters.empty()){
                    rotate_flag = camera_parameters[0].value("rotate_flag", -1); // only single image
                    portname = camera_parameters[0].value("portname", "image_stream_1");
                }

                if(rotate_flag >= 0 && rotate_flag <= 2){
                    cv::rotate(captured, captured, rotate_flag);

                    /* image encoding */
                    std::vector<unsigned char> serialized_image;
                    cv::imencode(".jpg", captured, serialized_image);

                    /* generate data meta tag */
                    json tag;
                    auto now = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = now - last_time;
                    last_time = now;

                    tag["fps"] = 1.0/elapsed.count();
                    tag["height"] = captured.rows;
                    tag["width"] = captured.cols;
                    tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();

                    /* send data */
                    flame::component::ZData msg_multipart_image;
                    msg_multipart_image.addstr(portname);
                    msg_multipart_image.addstr(tag.dump());
                    msg_multipart_image.addmem(serialized_image.data(), serialized_image.size());
                    
                    if(!dispatch(portname, msg_multipart_image)){
                        logger::warn("[{}] socket handle is not valid ", getName());
                    }
                    serialized_image.clear();

                }

                /* publish monitoring image */
                if(_use_image_stream_monitoring.load()){

                    /* resize image */
                    

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

cv::Mat solectrix_camera_grabber::grab() {
    if (!_device_opened) {
        logger::error("[{}] Device not opened. Call open_device() first", getName());
        return cv::Mat();
    }

    cv::Mat result = _frame_grabber->capture();
    
    if(!result.empty()){
        logger::debug("[{}] Frame grabbed successfully: {}x{}, channels: {}", getName(), result.cols, result.rows, result.channels());
    }
    else {
        logger::debug("[{}] No frame available", getName());
    }

    return result;
}
