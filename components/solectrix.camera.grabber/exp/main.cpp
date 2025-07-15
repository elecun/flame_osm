#include "sxpf.h"
#include "csi-2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>

#define IMG_DST_SIZE 100 * 1024 * 1024
uint8_t dst[IMG_DST_SIZE];

int main(int argc, char* argv[])
{
    FILE *file;
    char *buffer;
    int fileLen;
    sxpf_image_header_t *img_hdr;
    uint8_t             *img_ptr;
    uint32_t            packet_offset;
    uint8_t             vc_dt;
    uint8_t             dt = 0;
    uint32_t            word_count;
    uint8_t             *pixels;
    uint32_t            align;
    uint32_t            bits_per_pixel;
    uint32_t            pixel_group_size;
    uint8_t             *pdst = dst;
    uint32_t            decoded_pix = 0;
    int                 lines;
    int                 ret;
    long                filter_vc_dt;
    uint8_t             filter_dt;
    cv::Mat             rgb;
    long                color_conversion_code;
    long                shift_parameter = 0;
    char                *png_file = NULL;
    long                show_image = 1;

    if (argc < 4)
    {
        printf("The %s sample applications decodes RAW frame data into PNG files using OpenCV\n"
               "Usage: %s <raw_file> <datatype> <color_conversion_code> <shift_parameter> <png_file> <show_image>\n"
               "       color_conversion_codes:   \n"
               "             RGB2BGR      =   4  \n"
               "             BayerBG2BGR  =  46  \n"
               "             BayerGB2BGR  =  47  \n"
               "             BayerRG2BGR  =  48  \n"
               "             BayerGR2BGR  =  49  \n"
               "             YUV2BGR_UYVY = 108  \n"
               "             YUV2BGR_YVYU = 118  \n"
               "       shift_parameter:          \n"
               "       X = Shifiting value       \n"
               "             Default      :  X=0 \n"
               "             brighter     = +X   \n"
               "             darker       = -X   \n"
               , basename(argv[0]),  basename(argv[0]));
        return 0;
    }

    file = fopen(argv[1], "rb");
    if (!file)
    {
        fprintf(stderr, "Unable to open file\n");
        return -1;
    }

    filter_vc_dt = strtol(argv[2], NULL, 16);
    filter_dt = filter_vc_dt & 0x3f;

    color_conversion_code = strtol(argv[3], NULL, 10);

    if (argc >= 5)
    {
        shift_parameter = strtol(argv[4], NULL, 10);
    }

    if (argc >= 6)
    {
        png_file = argv[5];
    }

    if (argc >= 7)
    {
        show_image = strtol(argv[6], NULL, 10);
    }

    fseek(file, 0, SEEK_END);
    fileLen=ftell(file);
    fseek(file, 0, SEEK_SET);

    buffer=(char *)malloc(fileLen+1);
    if (!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(file);
        return -1;
    }

    ret = (int)fread(buffer, 1, fileLen, file);
    if(ret != fileLen)
    {
        printf("Read failed %d %d\n", ret, fileLen);
        exit(-1);
    }
    fclose(file);

    img_hdr = (sxpf_image_header_t*)buffer;
    img_ptr = (uint8_t*)img_hdr + img_hdr->payload_offset;
    packet_offset = 0;
    align = (img_hdr->bpp == 64) ? 7 : 3;
    lines = 0;

    for (uint32_t pkt_count = 0; pkt_count < img_hdr->rows; pkt_count++)
    {
        pixels = csi2_parse_dphy_ph(img_ptr + packet_offset, &vc_dt, &word_count);

        if (pixels)
        {
            //vc = vc_dt >> 6;
            dt = vc_dt & 0x3f;

            // Advance packet_offset to start of next packet:
            // - 8 bytes at the start make up the packet header, incl. preamble
            // - word_count payload data bytes follow
            // - Payload data is followed by a 2-byte checksum.
            // The following packet then starts at the next address that is an
            // integer multiple of 4 (if bpp <= 32) or 8 (if bpp == 64).
            packet_offset += (8 + word_count + (dt >= 0x10 ? 2 : 0) + align) & ~align;
        }
        else
        {
            // no CSI-2 packet header found - assume raw CSI-2 payload data
            pixels = img_ptr + packet_offset;
            vc_dt = (uint8_t)filter_vc_dt;
            word_count = img_hdr->columns * img_hdr->bpp / 8;
            packet_offset += word_count;
            dt = vc_dt & 0x3f;
        }

        if (vc_dt == filter_vc_dt)
        {
            if (csi2_decode_datatype(dt, &bits_per_pixel, &pixel_group_size))
            {
                printf("unsupported image type\n");
                return -1;
            }

            if (dt == 0x1e || dt == 0x1f) {
                bits_per_pixel /= 2; // bits_per_pixel value for yuv is not correct here
            }

            if (dt == 0x24)
            {
                decoded_pix =
                    csi2_decode_rgb8_24(pdst,
                                        word_count * 8 / bits_per_pixel, pixels,
                                        bits_per_pixel);
                pdst += decoded_pix * 3;
            }
            else
            {
                decoded_pix =
                    csi2_decode_raw32((uint32_t*)pdst, word_count * 8 / bits_per_pixel,
                                      pixels, bits_per_pixel);
                pdst += decoded_pix * 4;
            }

            lines++;
        }
    }

    if (decoded_pix > 0)
    {
        // YUV422 8bit - YUV422 10bit
        if (filter_dt >= 0x1e && filter_dt <= 0x1f)
        {
            cv::Mat img32;
            img32.create(lines, decoded_pix / 2, CV_32SC2);
            memcpy(img32.data, dst, sizeof(uint32_t) * lines * decoded_pix);
            cv::Mat img8;
            img32.convertTo(img8, CV_8UC2,
                            1.0 /
                                pow(2, (bits_per_pixel - 8) - shift_parameter));

            rgb = cv::Mat(img8.rows, img8.cols, CV_8UC3);
            cvtColor(img8, rgb, color_conversion_code);
        }
        // RGB888
        else if (filter_dt == 0x24)
        {
            cv::Mat img;
            img.create(lines, decoded_pix, CV_8UC3);
            memcpy(img.data, dst, sizeof(uint8_t) * 3 * lines * decoded_pix);
            rgb = cv::Mat(img.rows, img.cols, CV_8UC3);
            cvtColor(img, rgb, color_conversion_code);
        }
        // RAW6 - RAW20
        else if (filter_dt >= 0x28 && filter_dt <= 0x2f)
        {
            if (color_conversion_code >= 1000) // RAW16 coded data from e.g. camAD3 DUAL MAX96705/96706 tested with 8-bit YUV422 data
            {
                cv::Mat img32;
                img32.create(lines, decoded_pix, CV_32SC1);
                memcpy(img32.data, dst, sizeof(uint32_t) * lines * decoded_pix);
                img32 = img32 & 0xff;
                cv::Mat img8;
                img32.convertTo(img8, CV_8UC1); //, 1.0 / pow(2, (bits_per_pixel - 8) - shift_parameter));

                cv::Mat img8c2 = cv::Mat(img8.rows, img8.cols / 2, CV_8UC2);
                memcpy(img8c2.data, img8.data, sizeof(uint8_t) * img8.rows * img8.cols);

                rgb = cv::Mat(img8c2.rows, img8c2.cols, CV_8UC3);
                cvtColor(img8c2, rgb, color_conversion_code - 1000);
            }
            else
            {
                cv::Mat img32;
                img32.create(lines, decoded_pix, CV_32SC1);
                memcpy(img32.data, dst, sizeof(uint32_t) * lines * decoded_pix);
                cv::Mat img8;
                img32.convertTo(img8, CV_8UC1, 1.0 / pow(2, (bits_per_pixel - 8) - shift_parameter));
                rgb = cv::Mat(img8.rows, img8.cols, CV_8UC3);
                cvtColor(img8, rgb, color_conversion_code);
            }
        }
        else
        {
            printf("not implemented\n");
            goto end;
        }

        if (png_file != NULL)
        {
            cv::imwrite(png_file, rgb);
        }
        else
        {
            cv::imwrite("out.png", rgb);
        }

        if (show_image != 0)
        {
            cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
            cv::imshow("Display window", rgb);
            cv::waitKey(0);
        }
    }
    else
    {
        printf("datatype not found\n");
    }

end:
    free(buffer);

    return 0;
}
