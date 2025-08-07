/**
 * @file sxpf_grabber.hpp
 * @author Byunghun hwnag <bh.hwang@iae.re.kr>
 * @brief Solectrix Frame Grabber Device
 * @version 0.1
 * @date 2025-07-31
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_SOLECTRIX_CAMERA_GRABBER_DEVICE_HPP_INCLUDED
#define FLAME_SOLECTRIX_CAMERA_GRABBER_DEVICE_HPP_INCLUDED

#include <sxpf/sxpf.h>
#include <sxpf/csi-2.h>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* channel */
typedef struct input_channel_s
{
    sxpf_hdl        fg = 0;
    HWAITSXPF       devfd = 0;
    int             endpoint_id = 0;
    double          last_rxtime = 0;
    uint32_t        frame_info = 0;
} input_channel_t;


class sxpf_grabber {
    public:
        sxpf_grabber(vector<int> channels, int width, int height);
        ~sxpf_grabber();

        /* support functions */
        int get_num_cards();    /* get number of grabber cards */
        bool open();            /* grabber card open */
        void close();           /* grabber card close */
        void grab();             /* grab and return opencv image */

    private:
        vector<int> _channels;          /* to use multi channel */
        int _resolution_h { 1080 };     /* output image resolution (height) */
        int _resolution_w { 1920 };     /* output image resolution (width) */

        HWAITSXPF _devfd;                   /* device file description */
        sxpf_hdl _grabber_handle;           /* grabber card handle */
        sxpf_card_info_t _grabber_info;     /* grabber card  info */
        sxpf_card_props_t props;            /* grabber card properties */
        sxpf_event_t _events[20];           /* frame grabber events */

        
        
        int _csi2_datatype = 30;        /* CSI2 Data Type (YUV422=0x1e) */
        int _left_shift = 8;            /* Left Shift*/
        


        long long _last_time {0};

    

        uint16_t* tmp2 = nullptr;
        uint16_t* pdst = nullptr;
        

};

#endif