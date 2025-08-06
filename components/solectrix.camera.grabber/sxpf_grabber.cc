
#include "sxpf_grabber.hpp"
#include <flame/log.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>

#define NELEMENTS(ar)   (sizeof(ar) / sizeof((ar)[0]))

sxpf_grabber::sxpf_grabber(vector<int> channels)
:_channels(channels){
}

sxpf_grabber::~sxpf_grabber(){

}

bool sxpf_grabber::open(){

    /* find frame grabber card */
    int n_fg = sxpf_get_num_cards();
    if(n_fg<1){
        logger::error("(sxpf_grabber) Cannot found frame grabber available", n_fg);
        return false;
    }
    logger::debug("(sxpf_grabber) {} Frame Grabber found", n_fg);

    /* card handle (only single card )*/
    _card_handle = sxpf_open(0);
    sxpf_get_device_fd(_card_handle, &this->_devfd);
    sxpf_get_timestamp(_card_handle, &this->_last_time); // HW time at start
    sxpf_get_single_card_info(_card_handle, &this->_card_info);

    if(sxpf_start_record(_card_handle, SXPF_STREAM_ALL)){
        logger::error("(sxpf_grabber) failed to start grab");
        sxpf_close(_card_handle);
        return false;
    }

    /* frame grabber parameter */
    int csi2_datatype = strtol("0x1e", NULL, 16);
    logger::debug("CSI2 Data Type : {}", csi2_datatype);
    int left_shift = 8;
    logger::debug("Left Shift : {}", left_shift);
        
    return true;
}

void sxpf_grabber::close(){

    sxpf_stop(_card_handle, SXPF_STREAM_ALL);

    /* close sxpf channels */
    sxpf_close(_card_handle);
}

void sxpf_grabber::grab(){

    sxpf_event_t events[20];

    /* wait & read grab event */
    int len = sxpf_wait_events(1, &this->_devfd, 5 /* ms */);
    if(len>0){
        len = sxpf_read_event(_card_handle, events, NELEMENTS(events));
    }

    /* grab after read event */
    for(int n=0;n<len;n++){
        switch (events[n].type){
            case SXPF_EVENT_FRAME_RECEIVED: {
                logger::debug("(sxpf grabber) Event : SXPF_EVENT_FRAME_RECEIVED");
                int frame_slot = events[n].data / (1 << 24);
                sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)(sxpf_get_frame_ptr(_card_handle, frame_slot));

                //!!!! this must be done for each received buffer to ensure the hardware can continue to send images
                sxpf_release_frame(_card_handle, frame_slot, 0);
            }
            break;
            case SXPF_EVENT_I2C_MSG_RECEIVED:{ logger::debug("(sxpf grabber) Event : SXPF_EVENT_I2C_MSG_RECEIVED"); } break;
            case SXPF_EVENT_CAPTURE_ERROR:{ logger::debug("(sxpf grabber) Event : SXPF_EVENT_CAPTURE_ERROR {}", events->data);} break;
            case SXPF_EVENT_IO_STATE: { logger::debug("(sxpf grabber) Event : SXPF_EVENT_IO_STATE"); } break;
        }
    }

    // sxpf_image_header_t* img_hdr;
    // uint8_t* img_ptr;
    // uint32_t align;
    // uint32_t max_rows;
    // sxpf_meta_header_t* i2c_hdr;
    // uint8_t* i2c_ptr;
    // // int frame_slot;
    // long long int rx_time = 0;
    // // long long int           ts_start, ts_start_second;
    // // long long int           latency;
    // uint64_t ots = 0;
    // uint64_t ts = 0;
    // double frame_rate = 0.0;
    // uint8_t vc_dt;
    // uint8_t dt;

    // len = sxpf_wait_events(1, &this->_devfd, 5 /* ms */);
    // if(len>0){
    //     sxpf_get_timestamp(_card_handle, &rx_time);   // current HW time
    //     len = sxpf_read_event(_card_handle, events, NELEMENTS(events));
    // }

    // /* loop over all received events */
    // for(int n = 0;n<len;n++){
    //     switch (events[n].type){
    //         case SXPF_EVENT_FRAME_RECEIVED:
    //         {
    //             int frame_slot = events[n].data / (1 << 24);
    //             sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)(sxpf_get_frame_ptr(_card_handle, frame_slot));

    //             sxpf_release_frame(_card_handle, frame_slot, 0);
                
                /* image header */
                // unsigned int frame_counter = img_hdr->frame_counter;
                // // unsigned int frame_size = img_hdr->frame_size;
                // uint32_t frame_size = events[n].data & 0xffffff; // different from img_hdr->frame_size
                // unsigned short col = img_hdr->columns;
                // unsigned short row = img_hdr->rows;
                // unsigned short offset = img_hdr->payload_offset; //actual header size
                // unsigned char bpp = img_hdr->bpp;
                // uint32_t align = (img_hdr->bpp == 64) ? 7 : 3;
                // uint32_t filtered_offset = 0;
                // logger::info("Frame {}(framesize:{}, offset:{}, res:{}x{})", frame_counter, frame_size, offset, col, row);
                


                //!!!! this must be done for each received buffer to ensure the hardware can continue to send images
        //         sxpf_release_frame(_card_handle, frame_slot, 0);

        //     }
        //     break;
        //     case SXPF_EVENT_I2C_MSG_RECEIVED:
        //     {
        //         logger::info("SXPF_EVENT_I2C_MSG_RECEIVED");
        //     }
        //     break;
        //     case SXPF_EVENT_CAPTURE_ERROR:
        //     {
        //         logger::info("SXPF_EVENT_CAPTURE_ERROR {}", events->data);
        //     }
        //     break;
        //     case SXPF_EVENT_IO_STATE:
        //     {
        //         logger::info("SXPF_EVENT_IO_STATE");
        //     }
        //     break;
        // 
}

