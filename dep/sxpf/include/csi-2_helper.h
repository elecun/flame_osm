#ifndef CSI_2_HELPER_H
#define CSI_2_HELPER_H

#include "csi-2.h"

#include <vector>
#include <map>

struct csi2RawDecode
{
    uint32_t width;
    uint32_t height;
    uint32_t word_count;
    std::vector<uint8_t> data;
};

std::map<uint8_t, struct csi2RawDecode> csi2_get_decoded_images(sxpf_image_header_t *img_hdr, bool left_align)
{
    uint8_t *img_ptr;

    uint32_t            packet_offset;
    uint8_t             dt = 0;
    uint8_t             vc_dt;
    uint8_t             vc_dt_old = 255;
    uint32_t            word_count;
    uint8_t             *pixels;
    uint32_t            align;
    uint32_t            bits_per_pixel;
    uint32_t            pixel_group_size;

    std::map<uint8_t, struct csi2RawDecode> rawDecode;

    struct csi2RawDecode* decode;

    img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;

    packet_offset = 0;
    align = (img_hdr->bpp == 64) ? 7 : 3;

    while (1)
    {
        pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count);

        if (!pixels)
        {
            //RCLCPP_INFO(get_logger(), "invalid frame data");
            break;
        }

        if ( vc_dt != vc_dt_old )
        {
            vc_dt_old = vc_dt;

            dt = vc_dt & 0x3f;
            if (csi2_decode_datatype(dt, &bits_per_pixel, &pixel_group_size))
            {
                //RCLCPP_INFO(get_logger(), "unsupported image type");
                break;
            }

            if (rawDecode.find(vc_dt) == rawDecode.end())
            {
                rawDecode[vc_dt] = { word_count * 8 / bits_per_pixel, // width
                                     0,                               // height
                                     word_count,                      // word_count
                                     std::vector<uint8_t>()           // data
                                   };
                // TODO: parameter for reserve?
                rawDecode[vc_dt].data.reserve(10 * 1024 * 1024);
            }

            decode = &rawDecode[vc_dt];

            if (decode->word_count != word_count)
            {
                //RCLCPP_INFO(get_logger(), "word count changed within vc/dt");
                break;
            }
        }

        // Advance packet_offset to start of next packet:
        // - 8 bytes at the start make up the packet header, incl. preamble
        // - word_count payload data bytes follow
        // - Payload data is followed by a 2-byte checksum.
        // The following packet then starts at the next address that is an
        // integer multiple of 4 (if bpp <= 32) or 8 (if bpp == 64).
        packet_offset += (8 + word_count + 2 + align) & ~align;

        //if (dt == 0x1e || dt == 0x1f) {
        //  bits_per_pixel /= 2; // bits_per_pixel value for yuv is not correct here
        //}

        if(dt >= 0x28 && dt <= 0x2e)
        {
            decode->data.resize(decode->data.size() + decode->width * 2);

            if (left_align)
            {
                csi2_decode_raw16_msb(
                      (uint16_t*)(decode->data.data()) + (decode->width * decode->height),
                      decode->width, pixels, bits_per_pixel);
            } else {
                csi2_decode_raw16(
                      (uint16_t*)(decode->data.data()) + (decode->width * decode->height),
                      decode->width, pixels, bits_per_pixel);
            }
            decode->height++;
        }
        else if(dt >= 0x1e && dt <= 0x1f)
        {
            decode->data.resize(decode->data.size() + decode->width * 3);

            csi2_decode_uyvy8_24(
                  (uint8_t*)(decode->data.data()) + (decode->width * 3 * decode->height),
                  decode->width, pixels, bits_per_pixel);

            decode->height++;
        }
    }

    return rawDecode;
}

#endif
