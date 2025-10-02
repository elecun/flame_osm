
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
            }
            catch(json::exception& e){
                logger::info("{sxpf_grabber} Imcompleted camera parameters");
            }   
        }
    }

    _decode_csi2_datatype = strtol(parameters.at("csi2_datatype").get<string>().c_str(), NULL, 16);

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
    sxpf_get_timestamp(_grabber_handle, &_last_time); // HW time at start
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

    logger::debug("(sxpf_grabber) frame grabber starts recording..");

        
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

cv::Mat sxpf_grabber::capture(){
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
                logger::debug("(sxpf_grabber) Event: SXPF_EVENT_FRAME_RECEIVED");
                
                int frame_slot = _events[n].data / (1 << 24);
                sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)(sxpf_get_frame_ptr(_grabber_handle, frame_slot)); //read frame from slot
                
                if(!img_hdr){
                    logger::error("(sxpf_grabber) Failed to get frame pointer");
                    sxpf_release_frame(_grabber_handle, frame_slot, 0);
                    continue;
                }
                else {

                    if(_decode_csi2_datatype>=0){
                        unsigned int packet_offset = 0;
                        unsigned int filtered_offset = 0;
                        int x_size = 0;
                        int y_size = 0;
                        unsigned char bpp = 16; //16
                        static unsigned int align = (bpp == 64) ? 7 : 3;
                        unsigned int frame_size = x_size * y_size * bpp / 8;
                        unsigned char* pixels;
                        unsigned char vc_dt;
                        unsigned int word_count;
                        std::set<uint8_t>   frame_datatypes;
                        unsigned int packet_size;
                        unsigned int bits_per_pixel;
                        unsigned int pixel_group_size;
                        unsigned short* tmp2 = nullptr;
                        unsigned short* pdst = nullptr;
                        unsigned int decoded_pix;
                        const int components_per_pixel = 2; // default contant value

                        unsigned char* img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;

                        /* packet parsing for each line(1080) */
                        for (uint32_t pkt_count = 0; pkt_count < img_hdr->rows; pkt_count++){

                            pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count);
                            logger::debug("(sxpf_grabber) packet offset : {}, vc_dt : {}, word count : {}", packet_offset, vc_dt, word_count);

                            if(!pixels){
                                logger::error("(sxpf_grabber) Invalied frame data");
                                return captured_image;
                            }

                            frame_datatypes.insert(vc_dt); //0x1e(30)

                            if((vc_dt & 0x3f) <= 0x0f){ //0x1e&0x3f = 0x1e
                                packet_size = (8 + align) & ~align;
                            }
                            else{
                                packet_size = (8 + word_count + 2 + align) & ~align; // enter (packet size = 3852)
                            }

                            if(vc_dt == _decode_csi2_datatype){
                                uint8_t dt = vc_dt & 0x3f;
                                csi2_decode_datatype(dt,&bits_per_pixel, &pixel_group_size);
                                logger::debug("(sxpf_grabber) bits_per_pixel : {}, pixel_group_size : {}", bits_per_pixel, pixel_group_size);

                                if(dt >= 0x18 && dt <= 0x1f){
                                    _bits_per_component = 0;
                                    bits_per_pixel /= 2;
                                }

                                if(tmp2 == nullptr){
                                    x_size = word_count * 8 / bits_per_pixel; //result = 3840
                                    tmp2 = (unsigned short*)malloc(sizeof(unsigned short)*x_size*img_hdr->rows); // memory alloc. 3840*1080*2 bytes
                                    pdst = tmp2;
                                }

                                if ((pdst - tmp2 + x_size) > (x_size * img_hdr->rows)){
                                    logger::warn("(sxpf_grabber) decoded data exceeded allocated memory");
                                    break;
                                }
                                else
                                {
                                    if(dt != 0x24)
                                    {
                                        decoded_pix = csi2_decode_raw16(pdst, word_count * 8 / bits_per_pixel, pixels, bits_per_pixel);
                                        pdst += decoded_pix;
                                    }

                                    y_size += 1;
                                }
                            }

                            packet_offset += packet_size;
                        }

                        img_ptr = (unsigned char*)tmp2;
                        frame_size = x_size * y_size * sizeof(unsigned short);
                        x_size /= components_per_pixel;

                    }
                    
                    
                    // logger::debug("frame size : {}", frame_size);
                    // unsigned short payload_offset = img_hdr->payload_offset;

                    // logger::debug("(sxpf_grabber) col : {}, row : {}, bpp : {}, payload offset : {}", x_size, y_size, bpp, payload_offset);

                    // unsigned char* img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;


                }

                sxpf_release_frame(_grabber_handle, frame_slot, 0);
                logger::debug("(sxpf_grabber) release frame : slot {}", frame_slot);
                
                /* get image info */
                // uint32_t x_size = img_hdr->columns;
                // uint32_t y_size = img_hdr->rows;
                // uint8_t bpp = img_hdr->bpp;
                // uint32_t frame_size = x_size * y_size * bpp / 8;
                // unsigned short payload_offset = img_hdr->payload_offset;
                
                // logger::debug("(sxpf_grabber) Image: {}x{}, bpp:{}, frame_size:{}, payload_offset:{}", 
                //              x_size, y_size, bpp, frame_size, payload_offset);
                
                // uint8_t* img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;
                
                // /* check if this is a new frame */
                // int new_frame_info = _events[n].data;
                // bool is_new_frame = (new_frame_info & 0x00ffffff) > 0;
                
                // if(is_new_frame && img_ptr){
                //     try {
                //         /* Create cv::Mat from raw image data */
                //         if(_csi2_datatype == 0x1e){ // YUV422 format
                //             /* Create Mat for YUV422 data */
                //             cv::Mat yuv_image(y_size, x_size, CV_8UC2, img_ptr);
                            
                //             /* Convert YUV422 to BGR */
                //             cv::cvtColor(yuv_image, captured_image, cv::COLOR_YUV2BGR_YUYV);
                            
                //             logger::debug("(sxpf_grabber) Successfully converted YUV422 to BGR: {}x{}", 
                //                          captured_image.cols, captured_image.rows);
                //         }
                //         else {
                //             /* For other formats, create a grayscale or raw image */
                //             if(bpp == 8){
                //                 captured_image = cv::Mat(y_size, x_size, CV_8UC1, img_ptr).clone();
                //             }
                //             else if(bpp == 16){
                //                 captured_image = cv::Mat(y_size, x_size, CV_16UC1, img_ptr).clone();
                //             }
                //             else {
                //                 logger::warn("(sxpf_grabber) Unsupported bpp: {}, creating raw 8-bit image", bpp);
                //                 captured_image = cv::Mat(y_size, x_size, CV_8UC1, img_ptr).clone();
                //             }
                //         }
                //     }
                //     catch(const cv::Exception& e){
                //         logger::error("(sxpf_grabber) OpenCV Exception during conversion: {}", e.what());
                //         captured_image = cv::Mat(); // return empty Mat on error
                //     }
                // }
                
                // /* Release frame buffer - this is critical for hardware to continue */
                // sxpf_release_frame(_grabber_handle, frame_slot, 0);
                
                // /* Return immediately after processing first valid frame */
                // if(!captured_image.empty()){
                //     return captured_image;
                // }
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

void sxpf_grabber::grab(){

    /* grabbed image (RGB-color)*/
    // Mat converted_image(_resolution_h, _resolution_w, CV_8UC3); //RGB Format
    // Mat grabbed_raw_image(_resolution_h, _resolution_w, CV_8UC2); //YUV422 Format

    /* wait & read grab event */
    int len = sxpf_wait_events(1, &this->_devfd, 5 /* ms */);
    if(len>0){
        len = sxpf_read_event(_grabber_handle, _events, NELEMENTS(_events));
    }

    /* grab after read event */
    for(int n=0;n<len;n++){
        switch (_events[n].type){
            case SXPF_EVENT_FRAME_RECEIVED: {
                // logger::debug("(sxpf grabber) Event : SXPF_EVENT_FRAME_RECEIVED");
                int frame_slot = _events[n].data / (1 << 24);
                sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)(sxpf_get_frame_ptr(_grabber_handle, frame_slot));

                /* show info */
                uint32_t x_size = img_hdr->columns;
                uint32_t y_size = img_hdr->rows;
                uint8_t bpp = img_hdr->bpp;
                uint32_t frame_size = x_size * y_size * bpp / 8;
                unsigned short payload_offset = img_hdr->payload_offset;
                logger::debug("x:{},y:{},bpp:{},frame:{},payload_offset:{}", x_size, y_size, bpp, frame_size, payload_offset);

                uint8_t* img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;

                // /* check new frame */
                int new_frame_info = _events[n].data;
                bool is_new_frame = (new_frame_info & 0x00ffffff)>0;

                if(_csi2_datatype>=0){
                    uint32_t align = (bpp==64)?7:3;
                    uint32_t packet_offset = 0;
                    uint8_t vc_dt;
                    uint32_t word_count;
                    uint8_t* pixels = nullptr;
                    for(uint32_t packet_count=0; packet_count<img_hdr->rows; packet_count++){
                        pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count); // parse CSI2 packet header (word_count = 3840, number of data bytes)
                        
                    }
                }
                // logger::info("count : {}", count); //1080

                

                



                //!!!! this must be done for each received buffer to ensure the hardware can continue to send images
                sxpf_release_frame(_grabber_handle, frame_slot, 0);

                // // new frame info
                // int new_frame_info = _events[n].data;
                // bool isNewFrame = (new_frame_info & 0x00ffffff)>0;
                // if(isNewFrame){
                //     logger::info("---new frame info : {}", isNewFrame);

                //     uint32_t x_size = img_hdr->columns;
                //     uint32_t y_size = img_hdr->rows;
                //     uint8_t bpp = img_hdr->bpp;
                //     uint32_t frame_size = x_size * y_size * bpp / 8; // without header?

                //     uint8_t* img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;


                //     std::set<uint8_t> frame_datatypes;
                //     if(_csi2_datatype>=0){
                //         uint32_t packet_offset = 0;
                //         uint32_t filtered_offset = 0;
                //         uint32_t align = (bpp==64)?7:3;

                //         uint8_t* pixels = nullptr;
                //         uint8_t vc_dt;
                //         uint32_t word_count;
                //         uint32_t packet_size;
                //         uint32_t decoded_pix;
                //         for(uint32_t packet_count=0; packet_count<img_hdr->rows; packet_count){
                //             pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count);
                //             if(!pixels){
                //                 logger::error("(sxpf_grabber) Invalid Frame Data");
                //                 return;
                //             }
                //             frame_datatypes.insert(vc_dt);

                //             if((vc_dt&0x3f)<=0x0f)
                //                 packet_size = (8+align) & ~align;
                //             else
                //                 packet_size = (8+word_count+2+align) & ~align;

                //             uint8_t dt = vc_dt & 0x3f;
                //             uint32_t bits_per_pixel = 0;
                //             uint32_t pixel_group_size = 0;
                //             int bits_per_component = 0;
                //             csi2_decode_datatype(dt, &bits_per_pixel, &pixel_group_size);

                //             if(dt>=0x18 && dt <= 0x1f){
                //                 bits_per_component = 0;
                //                 bits_per_pixel /=2;                                
                //             }

                //             if(tmp2==nullptr){
                //                 x_size = word_count*8/bits_per_pixel;
                //                 tmp2 = (uint16_t*)malloc(sizeof(uint16_t*)*x_size*img_hdr->rows);
                //                 pdst = tmp2;
                //             }

                //             if(dt != 0x24){
                //                 decoded_pix = csi2_decode_raw16(pdst, word_count*8/bits_per_pixel, pixels, bits_per_pixel);
                //                 pdst += decoded_pix;
                //             }

                //             y_size+=1;
                //             packet_offset += packet_size;

                            

                //             logger::info("---packet size : {}", packet_size);
                    //     }
                    // }
                    
                    // logger::info("--- payload offset : {}", po);
                
            }
            break;
            case SXPF_EVENT_I2C_MSG_RECEIVED:{ logger::debug("(sxpf grabber) Event : SXPF_EVENT_I2C_MSG_RECEIVED"); } break;
            case SXPF_EVENT_CAPTURE_ERROR:{ logger::debug("(sxpf grabber) Event : SXPF_EVENT_CAPTURE_ERROR {}", _events[n].data);} break;
            case SXPF_EVENT_IO_STATE: { logger::debug("(sxpf grabber) Event : SXPF_EVENT_IO_STATE"); } break;
        }
    }

    /* color conversion */
    // if(!grabbed_raw_image.empty()){
    //     try{
    //         // cv::cvtColor(grabbed_raw_image, converted_image, cv::COLOR_YUV2BGR_YUY2); //YUV422 to BRG
    //     }
    //     catch(const cv::Exception& e){
    //         logger::error("(sxpf_grabber) CV Exception : {}", e.what());
    //     }
    // }

    // return converted_image;

}

