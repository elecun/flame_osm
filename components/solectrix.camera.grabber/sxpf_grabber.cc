
#include "sxpf_grabber.hpp"
#include <flame/log.hpp>

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

    /* open device*/
    for(auto& channel:_channels){
        channel.second.fg = sxpf_open(0);
        if(channel.second.fg){
            channel.second.endpoint_id = 0;

            if(sxpf_start_record(channel.second.fg, SXPF_STREAM_VIDEO0 << (channel.first & 0x0f))){
                logger::error("(sxpf_grabber) Failed to start stream channel #{}", channel.first);
                return false;
            }

            sxpf_get_device_fd(channel.second.fg, &channel.second.devfd);

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

            len = sxpf_wait_events(1, &channel.second.devfd, 50 /* ms */);
            if(len>0)
                len = sxpf_read_event(channel.second.fg, events, NELEMENTS(events));

            while(len>0){
                switch (evt->type)
                {
                    case SXPF_EVENT_FRAME_RECEIVED:
                        logger::info("SXPF_EVENT_FRAME_REVEIVED");
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