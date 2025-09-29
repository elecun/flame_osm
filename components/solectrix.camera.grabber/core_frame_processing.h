/**
 * Header for core frame processing functions
 * Extracted essential functionality for --card 0 --channel 4 -d0x1e -l8 parameters
 */

#ifndef CORE_FRAME_PROCESSING_H
#define CORE_FRAME_PROCESSING_H

#include "include/sxpf.h"
#include "include/sxpftypes.h"
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Process YUV422 frame data (datatype 0x1e) with specified left shift
 * @param img_hdr SXPF image header
 * @param left_shift Bit shift amount (typically 8)
 * @return OpenCV Mat containing RGB image
 */
cv::Mat process_yuv422_frame(sxpf_image_header_t* img_hdr, uint32_t left_shift);

/**
 * Wait for frame and process it
 * @param fg SXPF handle
 * @param devfd Device file descriptor
 * @param output Output OpenCV Mat
 * @return true if frame was processed successfully
 */
bool wait_and_process_frame(sxpf_hdl fg, HWAITSXPF devfd, cv::Mat& output);

/**
 * Initialize SXPF device (equivalent to --card X --channel Y)
 * @param endpoint_id Card number (0)
 * @param channel_id Channel number (4)  
 * @param fg Output SXPF handle
 * @param devfd Output device file descriptor
 * @return true if initialization successful
 */
bool initialize_device(int endpoint_id, int channel_id, sxpf_hdl* fg, HWAITSXPF* devfd);

/**
 * Cleanup SXPF device
 * @param fg SXPF handle to cleanup
 */
void cleanup_device(sxpf_hdl fg);

#ifdef __cplusplus
}
#endif

#endif // CORE_FRAME_PROCESSING_H