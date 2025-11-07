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
#include <dep/json.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;

/* channel */
typedef struct input_channel_s
{
    sxpf_hdl        fg = 0;
    HWAITSXPF       devfd = 0;
    int             endpoint_id = 0;
    double          last_rxtime = 0;
    uint32_t        frame_info = 0;
} input_channel_t;

/* camera setup */
typedef struct camera_param {
    string name;
    int card_endpoint = 0;
    int channel = 0;
    double rotate_cw = 0.0;
    int rotate_flag = -1;
    camera_param(const json& param){
        name = param.at("name").get<string>();
        card_endpoint = param.at("card").get<int>();
        channel = param.at("channel").get<int>();
        rotate_cw = param.at("rotate").get<double>();
    }
} camera_param_t;


class sxpf_grabber {
    public:
        // sxpf_grabber(vector<int> channels, int width, int height);
        sxpf_grabber(json parameters);
        ~sxpf_grabber();

        /* support functions */
        int get_num_cards();    /* get number of grabber cards */
        bool open();            /* grabber card open */
        void close();           /* grabber card close */
        int wait_event();       /* wait for event */

        Mat capture();          /* capture frame and return cv::Mat*/
        Mat _process_yuv422_frame(sxpf_image_header_t* img_hdr, uint32_t left_shift);

        const vector<camera_param_t>& get_parameter_container() const { return _parameter_container; } /* access parameter container */
        int get_rotate_flag() const { return _rotate_flag; }
        

    private:
        unsigned int _stream_channel_mask = 0;    /* stream channel SXPF_STREAM_VIDEOX*/
        int _decode_csi2_datatype = 0x1e;
        int _bits_per_component = 16;       //default is 16
        int _left_shift = 8;                //default is 8
        int _rotate_flag = -1;              //default is -1 (no rotation)


        vector<camera_param_t> _parameter_container;
        vector<int> _channels;          /* to use multi channel */
        int _resolution_h { 1080 };     /* output image resolution (height) */
        int _resolution_w { 1920 };     /* output image resolution (width) */

        HWAITSXPF _devfd;                   /* device file description */
        sxpf_hdl _grabber_handle;           /* grabber card handle */
        sxpf_card_info_t _grabber_info;     /* grabber card  info */
        sxpf_event_t _events[20];           /* frame grabber events */
       

};

#endif