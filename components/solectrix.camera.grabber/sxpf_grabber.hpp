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

#include "include/sxpf.h"
#include <map>
#include <vector>

using namespace std;

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
        sxpf_grabber(vector<int> channels);
        ~sxpf_grabber();

        bool open();
        void close();
        void grab();

    // private:
    //     double _get_elapsed_time();

    private:
        vector<int> _channels;
        sxpf_card_props_t props;

        sxpf_hdl _card_handle;          /*card handle */
        sxpf_card_info_t _card_info;    /* card info */
        HWAITSXPF _devfd;               /* device file desc. */
        long long _last_time {0};
        

};

#endif