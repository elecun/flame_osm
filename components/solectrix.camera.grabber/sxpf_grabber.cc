
#include "sxpf_grabber.hpp"
#include <flame/log.hpp>
#include <chrono>

#define NELEMENTS(ar)   (sizeof(ar) / sizeof((ar)[0]))

sxpf_grabber::sxpf_grabber(vector<int> channels){
    
    /* reserved channels */
    for(const auto& ch:channels){
        _channels.insert(make_pair(ch, input_channel_t{}));
    }
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

    /* open device (endpoint = card )*/
    for(auto& channel:_channels){
        channel.second.fg = sxpf_open(0);
        if(channel.second.fg){
            channel.second.endpoint_id = 0;

            if(sxpf_start_record(channel.second.fg, SXPF_STREAM_VIDEO0 << (channel.first & 0x0f))){
                logger::error("(sxpf_grabber) Failed to start stream channel #{}", channel.first);
                return false;
            }

            sxpf_get_device_fd(channel.secondsxpf_grabber.fg, &channel.second.devfd);

        }
        else{
            logger::error("(sxpf_grabber) Couldn't initialize Frame Grabber");
            return false;
        }
    }

    return true;

}

void sxpf_grabber::close(){

    /* close sxpf channels */
    for(auto& channel:_channels){
        if(channel.second.fg)
            sxpf_close(channel.second.fg);
    }
}

void sxpf_grabber::grab(){

    for(auto& channel:_channels){

        /* wait for new frame */
        if(channel.second.devfd>0){
            sxpf_event_t    events[20];
            sxpf_event_t    *evt = events;
            ssize_t         len;

            len = sxpf_wait_events(1, &channel.second.devfd, 1 /* ms */);
            if(len>0){
                len = sxpf_read_event(channel.second.fg, events, NELEMENTS(events));

                switch(evt->type){
                    case SXPF_EVENT_FRAME_RECEIVED:
                        logger::info("SXPF_EVENT_FRAME_REVEIVED");
                        channel.second.last_rxtime = _get_elapsed_time();
                        int frame_slot = evt->data / (1 << 24);
                        sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)sxpf_get_frame_ptr(channel.second.fg, frame_slot);
                        if(img_hdr){
                           sxpf_release_frame(channel.second.fg,
                                                   sdl_ctrl.new_frame_info /
                                                    (1 << 24), 0);
                        }
                        else{
                            logger::warn("(sxpf_grabber) Failed getting frame buffer pointer");
                        }
                    break;
                    case SXPF_EVENT_I2C_MSG_RECEIVED:
                    break;
                    case SXPF_EVENT_CAPTURE_ERROR:
                    break;
                    case SXPF_EVENT_IO_STATE:
                    break;
                }
            }

            
        }
    }
}

double sxpf_grabber::_get_elapsed_time(){
    using clock = chrono::steady_clock;
    using seconds = chrono::duration<double>;

    static clock::time_point t0 = clock::now();
    clock::time_point now = clock::now();

    seconds elapsed = now - t0;
    return elapsed.count(); // second unit
}