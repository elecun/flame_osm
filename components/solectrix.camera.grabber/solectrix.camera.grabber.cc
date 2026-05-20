
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

        /* set last capture times and start dispatch workers */
        for(const auto& cam_param : parameters["camera"]){
            int cam_channel = cam_param["channel"].get<int>();
            _last_capture_times[cam_channel] = chrono::high_resolution_clock::now();
            _dispatch_workers[cam_channel] = thread(&solectrix_camera_grabber::_dispatch_task, this, cam_channel);
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

    /* notify all dispatch workers */
    for(auto& [channel, cv] : _queue_cvs){
        cv.notify_all();
    }

    /* stop grabbing thread */
    if(_grab_worker.joinable()){
        _grab_worker.join();
        logger::debug("[{}] grabber is now successfully stopped", getName());
    }

    /* stop dispatch workers */
    for(auto& [channel, worker] : _dispatch_workers){
        if(worker.joinable()){
            worker.join();
        }
    }
    logger::debug("[{}] all dispatch workers are now successfully stopped", getName());

    /* device close */
    _frame_grabber->close();
}


void solectrix_camera_grabber::onData(flame::component::ZData& data){
    /* reserved function */
}

void solectrix_camera_grabber::_grab_task(json camera_parameters){

    while(!_worker_stop.load()){
        /* do grab */
        try{
            pair<int, cv::Mat> captured_data = _frame_grabber->capture();
            int cam_channel = captured_data.first;
            cv::Mat captured = captured_data.second;

            // If valid
            if(cam_channel>=0){
                if(_use_image_stream.load()){

                    // 1. encode image to jpg format
                    auto msg = make_shared<flame::component::ZData>();
                    vector<unsigned char> image_buf;
                    cv::imencode(".jpg", captured, image_buf);

                    //2. generate data meta tag
                    json tag;
                    auto now = chrono::high_resolution_clock::now();
                    chrono::duration<double> elapsed = now - _last_capture_times[cam_channel];
                    _last_capture_times[cam_channel] = now;

                    tag["fps"] = (elapsed.count() > 0) ? 1.0/elapsed.count() : 0.0;
                    tag["height"] = captured.rows;
                    tag["width"] = captured.cols;
                    tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
                    tag["cam_channel"] = cam_channel;

                    logger::debug("[{}] cam channel {}, fps : {}", getName(), cam_channel, tag["fps"].get<double>());

                    msg->from = fmt::format("image_stream_{}", cam_channel);
                    msg->meta = tag.dump();

                    // 3. populate multipart frames (first: metadata/tag, second: image data)
                    msg->addstr(msg->meta);
                    msg->addmem(image_buf.data(), image_buf.size());

                    /* push to channel queue */
                    {
                        lock_guard<mutex> lock(_queue_mtxs[cam_channel]);
                        if(_dispatch_queues[cam_channel].size() < _max_queue_size){
                            _dispatch_queues[cam_channel].push(msg);
                        }
                        else {
                            _dispatch_queues[cam_channel].pop();
                            _dispatch_queues[cam_channel].push(msg);
                        }
                    }
                    _queue_cvs[cam_channel].notify_one();
                }
            }
        }
        catch(const cv::Exception& e){
            logger::debug("[{}] CV Exception {}", getName(), e.what());
        }
        catch(const std::exception& e){
            logger::error("[{}] Standard Exception in grab task : {}", getName(), e.what());
        }
    }

    logger::debug("[{}] Stopped grab task..", getName());

}

void solectrix_camera_grabber::_dispatch_task(int channel){
    logger::debug("[{}] Started dispatch task for channel {}", getName(), channel);
    
    while(!_worker_stop.load()){
        try {
            shared_ptr<flame::component::ZData> msg = nullptr;
            {
                unique_lock<mutex> lock(_queue_mtxs[channel]);
                _queue_cvs[channel].wait(lock, [this, channel]{ return !_dispatch_queues[channel].empty() || _worker_stop.load(); });
                
                if(_worker_stop.load() && _dispatch_queues[channel].empty()) break;

                if(!_dispatch_queues[channel].empty()){
                    msg = _dispatch_queues[channel].front();
                    _dispatch_queues[channel].pop();
                }
            }

            if(msg){
                if(!dispatch(msg->from, *msg)){
                    logger::warn("[{}] Failed to dispatch image to port {}", getName(), msg->from);
                }
            }
        }
        catch(const std::exception& e){
            if(!_worker_stop.load()){
                logger::error("[{}] Exception in dispatch task for channel {}: {}", getName(), channel, e.what());
            }
            break;
        }
    }
    logger::debug("[{}] Stopped dispatch task for channel {}", getName(), channel);
}

bool solectrix_camera_grabber::open_device(int endpoint_id, int channel_id, uint32_t decode_csi2_datatype, int left_shift) {
    if (_device_opened) {
        logger::warn("[{}] Device already opened", getName());
        return true;
    }

    if(_frame_grabber->open()){
        _device_opened = true;
        logger::info("[{}] Device opened successfully (endpoint:{}, channel:{})", getName(), endpoint_id, channel_id);
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
