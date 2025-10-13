
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

void solectrix_camera_grabber::on_message(){
    /* reserved function */
}

void solectrix_camera_grabber::_grab_task(json parameters){

    while(!_worker_stop.load()){

        /* do grab */
        try{
            cv::Mat captured = _grabber_handle->capture2();
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
        logger::debug("[{}] Frame grabbed successfully: {}x{}, channels: {}", 
                    get_name(), result.cols, result.rows, result.channels());
    } else {
        logger::debug("[{}] No frame available", get_name());
    }
    
    return result;
}

cv::Mat solectrix_camera_grabber::_process_frame_data(sxpf_image_header_t* img_hdr) {
    cv::Mat result;
    
    if (!img_hdr || img_hdr->frame_size == 0) {
        return result;
    }
    
    uint8_t* image_data = (uint8_t*)(img_hdr + 1);
    int width = img_hdr->columns;  
    int height = img_hdr->rows;
    size_t available_data = img_hdr->frame_size - sizeof(sxpf_image_header_t);
    
    logger::debug("[{}] Processing frame: {}x{}, size={}, datatype=0x{:x}", 
                get_name(), width, height, available_data, _decode_csi2_datatype);
    
    // CSI-2 datatype processing (based on update_texture.cpp logic)
    if (_decode_csi2_datatype == 0x1e) { // YUV422 8-bit
        uint32_t bits_per_pixel, pixel_group_size;
        
        if (csi2_decode_datatype(_decode_csi2_datatype, &bits_per_pixel, &pixel_group_size)) {
            logger::error("[{}] Unsupported CSI-2 datatype: 0x{:x}", get_name(), _decode_csi2_datatype);
            return result;
        }
        
        // YUV datatype adjustment
        if (_decode_csi2_datatype >= 0x18 && _decode_csi2_datatype <= 0x1f) {
            bits_per_pixel /= 2; // For YUV, bits_per_pixel is not correct 
        }
        
        uint32_t word_count = available_data / 8;
        uint32_t x_size = word_count * 8 / bits_per_pixel;
        
        // Allocate buffer for decoded data
        std::vector<uint16_t> decoded_buffer(x_size * height);
        uint16_t* pdst = decoded_buffer.data();
        
        // Decode raw CSI-2 data
        uint32_t decoded_pix = csi2_decode_raw16(pdst, word_count * 8 / bits_per_pixel, 
                                               image_data, bits_per_pixel);
        
        if (decoded_pix == 0) {
            logger::error("[{}] Failed to decode CSI-2 data", get_name());
            return result;
        }
        
        logger::debug("[{}] Decoded {} pixels from CSI-2 data", get_name(), decoded_pix);
        
        // Apply left shift (based on update_texture.cpp left_shift logic)
        if (_left_shift > 0 && _left_shift <= 16) {
            for (uint32_t i = 0; i < decoded_pix; i++) {
                pdst[i] = pdst[i] << _left_shift;
            }
        }
        
        // Convert to OpenCV format - YUV422 UYVY
        // In YUV422, each pixel pair becomes one UYVY unit
        int yuv_width = decoded_pix / height / 2; // /2 because UYVY packs 2 pixels
        
        if (yuv_width > 0 && height > 0) {
            // Convert 16-bit to 8-bit for OpenCV
            cv::Mat decoded_16(height * yuv_width, 2, CV_16UC1, pdst);
            cv::Mat decoded_8;
            decoded_16.convertTo(decoded_8, CV_8UC1, 1.0/256.0);
            
            // Reshape to proper YUV422 format
            cv::Mat yuv_image = decoded_8.reshape(2, height); // 2 channels per row
            
            try {
                // Convert UYVY to RGB
                cv::cvtColor(yuv_image, result, cv::COLOR_YUV2RGB_UYVY);
                
                // Resize to target resolution if needed  
                if (result.cols != 1920 || result.rows != 1080) {
                    cv::Mat resized;
                    cv::resize(result, resized, cv::Size(1920, 1080));
                    result = resized;
                }
                
                logger::debug("[{}] Successfully converted YUV422 to RGB: {}x{}", 
                            get_name(), result.cols, result.rows);
                
            } catch (cv::Exception& e) {
                logger::warn("[{}] YUV conversion failed: {}, using grayscale", get_name(), e.what());
                
                // Fallback to grayscale
                cv::Mat gray = decoded_8.reshape(1, height * yuv_width);
                cv::resize(gray, result, cv::Size(1920, 1080));
            }
        }
    } else {
        // Fallback for other datatypes - direct copy
        int bytes_per_pixel = available_data / (width * height);
        
        if (bytes_per_pixel == 1) {
            result = cv::Mat(height, width, CV_8UC1);
        } else if (bytes_per_pixel == 2) {
            result = cv::Mat(height, width, CV_8UC2);
        } else if (bytes_per_pixel >= 3) {
            result = cv::Mat(height, width, CV_8UC3);
        } else {
            result = cv::Mat(height, width, CV_8UC1);
        }
        
        size_t copy_size = std::min(available_data, result.total() * result.elemSize());
        memcpy(result.data, image_data, copy_size);
    }
    
    return result;
}

