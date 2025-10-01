/**
 * Core frame processing functions extracted from sxpf-app.cpp and update_texture.cpp
 * Focused on --card 0 --channel 4 -d0x1e -l8 parameters
 */

#include "include/sxpf.h""
#include "include/csi-2.h"
#include "include/sxpftypes.h"
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cstring>

// Core CSI-2 processing for datatype 0x1e (YUV422 8-bit) with left_shift=8
cv::Mat process_yuv422_frame(sxpf_image_header_t* img_hdr, uint32_t left_shift) {
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
    
    // Process all CSI-2 packets
    for (uint32_t pkt_count = 0; pkt_count < img_hdr->rows; pkt_count++) {
        uint8_t vc_dt;
        uint32_t word_count;
        uint32_t packet_size;
        uint32_t align = 3; // 4-byte alignment for bpp <= 32
        
        // Parse CSI-2 packet header
        uint8_t* pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count);
        if (!pixels) {
            printf("Invalid frame data\n");
            return cv::Mat();
        }
        
        // Calculate packet size
        if ((vc_dt & 0x3f) <= 0x0f) {
            packet_size = (8 + align) & ~align; // Short packet
        } else {
            packet_size = (8 + word_count + 2 + align) & ~align; // Long packet
        }
        
        // Process only packets with matching datatype (0x1e)
        if (vc_dt == 0x1e) {
            uint8_t dt = vc_dt & 0x3f;
            
            // Get CSI-2 datatype info
            if (csi2_decode_datatype(dt, &bits_per_pixel, &pixel_group_size)) {
                printf("Unsupported image type\n");
                return cv::Mat();
            }
            
            // YUV datatype adjustment (0x18-0x1f range)
            if (dt >= 0x18 && dt <= 0x1f) {
                bits_per_pixel /= 2; // For YUV, bits_per_pixel needs adjustment
            }
            
            // Calculate decoded pixel count for this packet
            uint32_t pixels_this_packet = word_count * 8 / bits_per_pixel;
            
            // Allocate buffer on first packet
            if (!decoded_buffer) {
                x_size = pixels_this_packet;
                decoded_buffer = (uint16_t*)malloc(sizeof(uint16_t) * x_size * img_hdr->rows);
                if (!decoded_buffer) {
                    return cv::Mat();
                }
            }
            
            // Decode CSI-2 raw data
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
            // Convert 16-bit to 8-bit
            cv::Mat decoded_16(y_size, x_size, CV_16UC1, decoded_buffer);
            cv::Mat decoded_8;
            decoded_16.convertTo(decoded_8, CV_8UC1, 1.0/256.0);
            
            // Reshape for YUV422 (2 channels per pixel pair)
            cv::Mat yuv_image = decoded_8.reshape(2, y_size);
            
            try {
                // Convert YUV422 to RGB
                cv::cvtColor(yuv_image, result, cv::COLOR_YUV2RGB_UYVY);
                
                printf("Successfully converted YUV422 to RGB: %dx%d\n", result.cols, result.rows);
            } catch (cv::Exception& e) {
                printf("YUV conversion failed: %s, using grayscale\n", e.what());
                result = decoded_8.clone();
            }
        }
        
        free(decoded_buffer);
    }
    
    return result;
}

// Simplified frame grabbing loop (from sxpf-app.cpp main loop)
bool wait_and_process_frame(sxpf_hdl fg, HWAITSXPF devfd, cv::Mat& output) {
    sxpf_event_t events[20];
    ssize_t len;
    
    // Wait for events with 5ms timeout
    len = sxpf_wait_events(1, &devfd, 5);
    
    if (len > 0) {
        len = sxpf_read_event(fg, events, 20);
    }
    
    // Process events
    for (int i = 0; i < len; i++) {
        sxpf_event_t* evt = &events[i];
        
        switch (evt->type) {
            case SXPF_EVENT_FRAME_RECEIVED:
            {
                int frame_slot = evt->data / (1 << 24);
                sxpf_image_header_t* img_hdr = (sxpf_image_header_t*)sxpf_get_frame_ptr(fg, frame_slot);
                
                if (img_hdr && img_hdr->frame_size > 0) {
                    // Process frame with left_shift=8
                    output = process_yuv422_frame(img_hdr, 8);
                    
                    // Release frame buffer
                    sxpf_release_frame(fg, frame_slot, 0);
                    
                    return true; // Frame processed successfully
                }
                
                sxpf_release_frame(fg, frame_slot, 0);
                break;
            }
            
            case SXPF_EVENT_CAPTURE_ERROR:
            {
                uint32_t error_code = evt->data;
                if (error_code & 0x1000) {
                    printf("Timestamp encoder error (non-critical): 0x%08x\n", error_code);
                } else {
                    printf("Capture error: 0x%08x\n", error_code);
                }
                break;
            }
            
            case SXPF_EVENT_IO_STATE:
                if (evt->data != SXPF_IO_NORMAL) {
                    printf("PCIe error: %d\n", evt->data);
                    return false;
                }
                break;
                
            case SXPF_EVENT_I2C_MSG_RECEIVED:
                sxpf_release_frame(fg, evt->data / (1 << 24), 0);
                break;
        }
    }
    
    return false; // No frame processed
}

// Device initialization (equivalent to --card 0 --channel 4)
bool initialize_device(int endpoint_id, int channel_id, sxpf_hdl* fg, HWAITSXPF* devfd) {
    *fg = sxpf_open(endpoint_id);
    if (!*fg) {
        printf("Failed to open SXPF endpoint %d\n", endpoint_id);
        return false;
    }
    
    if (sxpf_start_record(*fg, SXPF_STREAM_VIDEO0 << (channel_id & 0xf))) {
        printf("Failed to start stream on channel %d\n", channel_id);
        sxpf_close(*fg);
        return false;
    }
    
    sxpf_get_device_fd(*fg, devfd);

    
    printf("Device initialized: endpoint=%d, channel=%d\n", endpoint_id, channel_id);
    return true;
}

void cleanup_device(sxpf_hdl fg) {
    if (fg) {
        sxpf_stop(fg, SXPF_STREAM_ALL);
        sxpf_close(fg);
    }
}