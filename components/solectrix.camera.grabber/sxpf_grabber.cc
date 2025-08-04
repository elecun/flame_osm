
#include "sxpf_grabber.hpp"
#include <flame/log.hpp>
#include <chrono>

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

    if(sxpf_start_record(_card_handle, SXPF_STREAM_VIDEO4)){
        logger::error("(sxpf_grabber) failed to start grab");
        sxpf_close(_card_handle);
        return false;
    }
        
    return true;
}

void sxpf_grabber::close(){

    /* close sxpf channels */
    sxpf_close(_card_handle);
}

void sxpf_grabber::grab(){

    sxpf_event_t            events[20];
    int                     len;

    sxpf_image_header_t* img_hdr;
    uint8_t* img_ptr;
    sxpf_meta_header_t* i2c_hdr;
    uint8_t* i2c_ptr;
    int frame_slot;
    long long int rx_time = 0;
    // long long int           ts_start, ts_start_second;
    // long long int           latency;

    len = sxpf_wait_events(1, &this->_devfd, 1 /* ms */);
    if(len>0){
        sxpf_get_timestamp(_card_handle, &rx_time);   // current HW time
        len = sxpf_read_event(_card_handle, events, NELEMENTS(events));
    }

    /* loop over all received events */
    for(int n = 0;n<len;n++){
        switch (events[n].type){
            case SXPF_EVENT_FRAME_RECEIVED:
            {
                frame_slot = events[n].data / (1 << 24);
                img_hdr = static_cast<sxpf_image_header_t*>(sxpf_get_frame_ptr(_card_handle, frame_slot));
                img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;

                unsigned int channel = img_hdr->cam_id;
                unsigned short col = img_hdr->columns;
                unsigned short row = img_hdr->rows;
                logger::info("SXPF_EVENT_FRAME_REVEIVED : {}({}x{})", channel, col, row);
                

                //!!!! this must be done for each received buffer to ensure the hardware can continue to send images
                sxpf_release_frame(_card_handle, frame_slot, 0);
            }
            break;
            case SXPF_EVENT_I2C_MSG_RECEIVED:
            {
                logger::info("SXPF_EVENT_I2C_MSG_RECEIVED");
            }
            break;
            case SXPF_EVENT_CAPTURE_ERROR:
            {
                logger::info("SXPF_EVENT_CAPTURE_ERROR {}", events->data);
            }
            break;
            case SXPF_EVENT_IO_STATE:
            {
                logger::info("SXPF_EVENT_IO_STATE");
            }
            break;
        }
    }
}

// double sxpf_grabber::_get_elapsed_time(){
//     using clock = chrono::steady_clock;
//     using seconds = chrono::duration<double>;

//     static clock::time_point t0 = clock::now();
//     clock::time_point now = clock::now();

//     seconds elapsed = now - t0;
//     return elapsed.count(); // second unit
// }