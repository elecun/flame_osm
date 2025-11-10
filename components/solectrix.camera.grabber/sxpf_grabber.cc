
#include "sxpf_grabber.hpp"
#include <flame/log.hpp>
#include <chrono>

#define NELEMENTS(ar)   (sizeof(ar) / sizeof((ar)[0]))

// sxpf_grabber::sxpf_grabber(vector<int> channels, int height, int width)
// :_channels(channels), _resolution_h(height), _resolution_w(width){
// }

sxpf_grabber::sxpf_grabber(json parameters){

    /* Found Grabber Card */
    int n_fg = sxpf_get_num_cards();
    if(n_fg>0)
        logger::info("(sxpf_grabber) {} frame grabber available", n_fg);
    else
        logger::error("(sxpf_grabber) Cannot found frame grabber available");
    

    /* apply from profile */
    json camera_parameters = parameters["camera"];

    

    if(camera_parameters.is_array()){
        for(const auto& camera: camera_parameters){
            try{
                _parameter_container.push_back(camera);
                logger::debug("(sxpf_grabber) applied camera parameter: {}", camera.dump());
            }
            catch(json::exception& e){
                logger::info("{sxpf_grabber} Imcompleted camera parameters");
            }   
        }
    }

    _decode_csi2_datatype = strtol(parameters.at("csi2_datatype").get<string>().c_str(), NULL, 16);
    _rotate_flag = parameters.value("rotate_flag", -1);

    /* show the applied camera parameters */
    logger::info("(sxpf_grabber) {} camera device will be attached", _parameter_container.size());
    for(const auto& camera:_parameter_container){
        logger::debug("(sxpf_grabber) attaching {} : endpoint({}), channel({})", camera.name, camera.card_endpoint, camera.channel);
    }
}

sxpf_grabber::~sxpf_grabber(){

}

int sxpf_grabber::get_num_cards(){
    /* find frame grabber card */
    return sxpf_get_num_cards();   
}

bool sxpf_grabber::open(){

    /* Found Grabber Card */
    int n_fg = sxpf_get_num_cards();
    if(n_fg<1){
        logger::error("(sxpf_grabber) Cannot found frame grabber available");
        return false;
    }
    
    /* open grabber card & get the handle (only single card )*/
    _grabber_handle = sxpf_open(0); //0=card index
    sxpf_get_device_fd(_grabber_handle, &_devfd);
    // sxpf_get_timestamp(_grabber_handle, &_last_time); // HW time at start
    sxpf_get_single_card_info(_grabber_handle, &_grabber_info);

    logger::debug("(sxpf_grabber) {} {} Frame Grabber found", n_fg, _grabber_info.model);
    
    for(const auto& camera: _parameter_container){
        switch(camera.channel){
            case 0: _stream_channel_mask |= SXPF_STREAM_VIDEO0; break;
            case 1: _stream_channel_mask |= SXPF_STREAM_VIDEO1; break;
            case 2: _stream_channel_mask |= SXPF_STREAM_VIDEO2; break;
            case 3: _stream_channel_mask |= SXPF_STREAM_VIDEO3; break;
            case 4: _stream_channel_mask |= SXPF_STREAM_VIDEO4; break;
            case 5: _stream_channel_mask |= SXPF_STREAM_VIDEO5; break;
            case 6: _stream_channel_mask |= SXPF_STREAM_VIDEO6; break;
            case 7: _stream_channel_mask |= SXPF_STREAM_VIDEO7; break;
        }
    }

    if(sxpf_start_record(_grabber_handle, _stream_channel_mask)){ // only channel 4
        logger::error("(sxpf_grabber) failed to start grab");
        sxpf_close(_grabber_handle);
        return false;
    }
        
    return true;
}

void sxpf_grabber::close(){

    /* record stop */
    sxpf_stop(_grabber_handle, _stream_channel_mask);

    /* close sxpf channels */
    sxpf_close(_grabber_handle);

    logger::debug("(sxpf_grabber) stop & close frame grabber device");
}

int sxpf_grabber::wait_event(){
    return sxpf_wait_events(1, &this->_devfd, 5 /* ms */);
}

// Core CSI-2 processing for datatype 0x1e (YUV422 8-bit) with left_shift=8
Mat sxpf_grabber::_process_yuv422_frame(sxpf_image_header_t* img_hdr, uint32_t left_shift) {

    if (!img_hdr || img_hdr->frame_size == 0) {
        return cv::Mat();
    }
    
    uint8_t* img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;
    uint32_t packet_offset = 0;
    uint32_t bits_per_pixel, pixel_group_size;
    uint32_t decoded_pix = 0;
    uint32_t x_size = 0, y_size = 0;
    uint16_t* decoded_buffer = nullptr;
    uint32_t total_decoded_pixels = 0;
    
    /* Process all CSI-2 packets */
    for (uint32_t pkt_count = 0; pkt_count < img_hdr->rows; pkt_count++) {
        uint8_t vc_dt;
        uint32_t word_count;
        uint32_t packet_size;
        uint32_t align = 3; // 4-byte alignment for bpp <= 32
        
        /* Parse CSI-2 packet header */
        uint8_t* pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count);
        if (!pixels) {
            return cv::Mat();
        }
        
        /* Calculate packet size */
        if ((vc_dt & 0x3f) <= 0x0f) {
            packet_size = (8 + align) & ~align; // Short packet
        } else {
            packet_size = (8 + word_count + 2 + align) & ~align; // Long packet
        }
        
        /* Process only packets with matching datatype (0x1e) */
        if (vc_dt == 0x1e) {
            uint8_t dt = vc_dt & 0x3f;
            
            /* Get CSI-2 datatype info */
            if (csi2_decode_datatype(dt, &bits_per_pixel, &pixel_group_size)) {
                logger::debug("(sxpf_grabber) Unsupported image type)");
                return cv::Mat();
            }
            
            /* YUV datatype adjustment (0x18-0x1f range) */
            if (dt >= 0x18 && dt <= 0x1f) {
                bits_per_pixel /= 2; // For YUV, bits_per_pixel needs adjustment
            }
            
            /* Calculate decoded pixel count for this packet */
            uint32_t pixels_this_packet = word_count * 8 / bits_per_pixel;
            
            /* Allocate buffer on first packet */
            if (!decoded_buffer) {
                x_size = pixels_this_packet;
                decoded_buffer = (uint16_t*)malloc(sizeof(uint16_t) * x_size * img_hdr->rows);
                if (!decoded_buffer) {
                    return cv::Mat();
                }
            }
            
            /* Decode CSI-2 raw data */
            decoded_pix = csi2_decode_raw16(
                decoded_buffer + total_decoded_pixels,
                pixels_this_packet,
                pixels,
                bits_per_pixel
            );
            
            total_decoded_pixels += decoded_pix;
            y_size++;
        }
        
        packet_offset += packet_size;
    }
    
    cv::Mat result;
    
    if (decoded_buffer && total_decoded_pixels > 0) {
        // Apply left shift (equivalent to -l8 parameter)
        if (left_shift > 0 && left_shift <= 16) {
            for (uint32_t i = 0; i < total_decoded_pixels; i++) {
                decoded_buffer[i] = decoded_buffer[i] << left_shift;
            }
        }
        
        // Convert to OpenCV format
        // For YUV422: decoded pixels = x_size * y_size, where x_size = 3840 for 1920 width
        // Each pair of decoded values represents one UYVY pixel pair
        int actual_width = x_size / 2; // 3840/2 = 1920 for YUV422
        
        if (actual_width > 0 && y_size > 0) {
            /* Convert 16-bit to 8-bit */
            cv::Mat decoded_16(y_size, x_size, CV_16UC1, decoded_buffer);
            cv::Mat decoded_8;
            decoded_16.convertTo(decoded_8, CV_8UC1, 1.0/256.0);
            
            /* Reshape for YUV422 (2 channels per pixel pair) */
            cv::Mat yuv_image = decoded_8.reshape(2, y_size);
            
            try {
                /* Convert YUV422 to BGR */
                cv::cvtColor(yuv_image, result, cv::COLOR_YUV2BGR_UYVY);
            } catch (cv::Exception& e) {
                logger::debug("(sxpf_grabber) YUV conversion failed: {}, using grayscale", e.what());
                result = decoded_8.clone();
            }
        }
        
        free(decoded_buffer);
    }
    
    return result;
}

Mat sxpf_grabber::capture(){

    cv::Mat captured_image;
    
    /* wait & read grab event */
    int len = sxpf_wait_events(1, &this->_devfd, 100 /* ms */);
    if(len <= 0){
        logger::debug("(sxpf_grabber) No events received");
        return captured_image; // return empty Mat
    }
    
    len = sxpf_read_event(_grabber_handle, _events, NELEMENTS(_events));

    /* process events to grab frame */
    for(int n = 0; n < len; n++){
        switch (_events[n].type){
            case SXPF_EVENT_FRAME_RECEIVED: {

                int frame_slot = _events[n].data / (1 << 24);
                sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)(sxpf_get_frame_ptr(_grabber_handle, frame_slot)); //read frame from slot

                if(!img_hdr){
                    sxpf_release_frame(_grabber_handle, frame_slot, 0);
                    continue;
                }
                else {
                    captured_image = _process_yuv422_frame(img_hdr, _left_shift);
                    sxpf_release_frame(_grabber_handle, frame_slot, 0);
                }

                // logger::debug("(sxpf_grabber) Event: SXPF_EVENT_FRAME_RECEIVED on slot {}", frame_slot);

            }
                break;
            
            case SXPF_EVENT_I2C_MSG_RECEIVED:
                logger::debug("(sxpf_grabber) Event: SXPF_EVENT_I2C_MSG_RECEIVED");
                break;
                
            case SXPF_EVENT_CAPTURE_ERROR:
                logger::error("(sxpf_grabber) Event: SXPF_EVENT_CAPTURE_ERROR {}", _events[n].data);
                break;
                
            case SXPF_EVENT_IO_STATE:
                logger::debug("(sxpf_grabber) Event: SXPF_EVENT_IO_STATE");
                break;
                
            default:
                logger::debug("(sxpf_grabber) Unknown event type: {}", _events[n].type);
                break;
        }
    }
    
    return captured_image; // return empty Mat if no valid frame was captured

}



