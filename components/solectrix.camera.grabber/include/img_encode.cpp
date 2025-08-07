/** @file img_encode.cpp
 *
 * Implementation of CSI-2 image encoding.
 */
#include "csi-2.h"
#include "checksum/crc.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#define BV(n)    (1UL << (n))

#define MAX_NUM_VC_DTS 100

/** Portable parity function.
 *
 * @return  The parity of the function's argument \c v.
 */
static uint8_t parity(uint32_t v)
{
    // https://graphics.stanford.edu/~seander/bithacks.html#ParityParallel
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v &= 0xf;
    return (0x6996 >> v) & 1;
}


/** Fill in the 6bit ECC field of a CSI-2 packet header.
 *
 * @param ph    Pointer to the start of a 4-byte packet header.
 */
void csi2_update_ph_ECC(uint8_t *ph)
{
    static const uint32_t m0 =
        BV(0)  | BV(1)  | BV(2)  | BV(4)  | BV(5)  | BV(7)  | BV(10) | BV(11) |
        BV(13) | BV(16) | BV(20) | BV(21) | BV(22) | BV(23) | BV(30);
    static const uint32_t m1 =
        BV(0)  | BV(1)  | BV(3)  | BV(4)  | BV(6)  | BV(8)  | BV(10) | BV(12) |
        BV(14) | BV(17) | BV(20) | BV(21) | BV(22) | BV(23) | BV(31);
    static const uint32_t m2 =
        BV(0)  | BV(2)  | BV(3)  | BV(5)  | BV(6)  | BV(9)  | BV(11) | BV(12) |
        BV(15) | BV(18) | BV(20) | BV(21) | BV(22) | BV(30) | BV(31);
    static const uint32_t m3 =
        BV(1)  | BV(2)  | BV(3)  | BV(7)  | BV(8)  | BV(9)  | BV(13) | BV(14) |
        BV(15) | BV(19) | BV(20) | BV(21) | BV(23) | BV(30) | BV(31);
    static const uint32_t m4 =
        BV(4)  | BV(5)  | BV(6)  | BV(7)  | BV(8)  | BV(9)  | BV(16) | BV(17) |
        BV(18) | BV(19) | BV(20) | BV(22) | BV(23) | BV(30) | BV(31);
    static const uint32_t m5 =
        BV(10) | BV(11) | BV(12) | BV(13) | BV(14) | BV(15) | BV(16) | BV(17) |
        BV(18) | BV(19) | BV(21) | BV(22) | BV(23) | BV(30) | BV(31);

    uint32_t    h = ph[0] + ph[1] * 256u + ph[2] * 65536u + ph[3] * 16777216u;

    uint8_t     ecc =
        (parity(h & m0) << 0) |
        (parity(h & m1) << 1) |
        (parity(h & m2) << 2) |
        (parity(h & m3) << 3) |
        (parity(h & m4) << 4) |
        (parity(h & m5) << 5);

    ph[3] = ecc | (ph[3] & 0xc0);   // update ECC in packet header
}


/** Compute the payload checksum of a long packet.
 *
 * @param data      Pointer to the long packet's payload data.
 * @param wordcount Number of payload bytes to compute the checksum for
 *
 * @return The payload CRC.
 */
uint16_t csi2_payload_checksum(uint8_t const *data, uint32_t wordcount)
{
    return crc_update(0xffff, data, wordcount);
}


/** Encode a complete CSI-2 image frame (without the 64byte frame header).
 *
 * @note    Input data is expected to be aligned to the lowest significant bit.
 *
 * @param out           Pointer to first output byte (directly after header)
 *                      or \c NULL for just requesting needed buffer size
 * @param max_out_bytes Max.\ number of bytes that can be written to out
 * @param out_align     Number of bits each long packet needs to be aligned to.
 *                      Allowed are only 32 or 64, depending on card version.
 * @param vc_dt_config  Pointer to array of \c csi2_encode_params_t entries
 * @param num_vc_dts    Number of elements in the \c vc_dt_config array (max 8)
 * @param do_checksums  0 = write dummy checksums, 1 = compute CRC and ECC
 * @param force_equal_line_size Flag: ensure that all CSI-2 packets have the
 *                      same word count.
 *
 * @return  positive: Number of payload bytes written or needed number of bytes
 *                    if \c out==NULL
 * @return  negative: Error
 * @return -1:  invalid parameter
 * @return -2:  invalid virtual channel
 * @return -3:  data type not supported
 * @return -4:  input pixel depth not supported
 * @return -5:  internal error
 * @return -6:  columns not an integer multiple of pixel group size
 * @return -7:  too many columns (byte count must fit in 16 bit)
 * @return -8:  doesn't fit into size of output buffer
 * @return -9:  out of memory allocating intermediate data structure
 * @return -10: invalid VC/DT config (increment/first line too high)
 * @return -11: invalid VC/DT config (overlapping lines of separate VC/DTs)
 * @return -12: internal error (low level encoders returned wrong pixel count)
 * @return -13: force_equal_line_size was set to true, but lines of different
 *              data types have different length
 */
int csi2_multi_encode(uint8_t *out, uint32_t max_out_bytes, uint32_t out_align,
                      csi2_encode_params_t *vc_dt_config, unsigned num_vc_dts,
                      int do_checksums, int force_equal_line_size)
{
    if (!(out_align == 16 || out_align == 32 || out_align == 64) ||
        vc_dt_config == NULL || num_vc_dts < 1 || num_vc_dts > MAX_NUM_VC_DTS ||
        do_checksums / 2 != 0)
    {
        return -1;  // invalid parameter
    }

    uint32_t    output_align_mask = (out_align == 64) ? 7 :
                                    (out_align == 32) ? 3 : 1;
    uint32_t    total_lines = 0;
    uint32_t    total_bytes = 0;

    struct vc_state_s
    {
        uint32_t    line_cnt; // line count per VC/DT config; 256 = upper limit
        uint16_t    word_count;
        uint32_t    data_bits;
        uint8_t    *src;
        uint32_t    num_padding;

    } vc_state[MAX_NUM_VC_DTS];

    memset(vc_state, 0, num_vc_dts * sizeof(struct vc_state_s)); // reset states

    // validate VC/DT configuration:
    // - are all requested data types supported
    // - find total number of lines/long packets to output
    // - sum up all sizes of long packets to write incl. padding
    for (unsigned i = 0; i < num_vc_dts; i++)
    {
        csi2_encode_params_t   *cfg = vc_dt_config + i;

        if (cfg->vc >= 4)
            return -2;  // invalid virtual channel

        if (!((/*cfg->dt >= 0x00 &&*/ cfg->dt <= 0x03) ||   // Synchronization Short Packet
              (cfg->dt >= 0x08 && cfg->dt <= 0x0F) ||   // Generic Short Packet
              (cfg->dt >= 0x10 && cfg->dt <= 0x12) ||   // NULL, Blanking, EBD
              (cfg->dt >= 0x13 && cfg->dt <= 0x17) ||   // Generic Long Packet
              (cfg->dt >= 0x18 && cfg->dt <= 0x1F) ||   // YUV Image Data Types
              (cfg->dt == 0x24)                    ||   // RGB888
              (cfg->dt >= 0x28 && cfg->dt <= 0x2f) ||   // RAW6..RAW20
              (cfg->dt >= 0x30 && cfg->dt <= 0x37) ))   // User Defined Byte-based Data
        {
            return -3;  // data type not supported
        }

        uint32_t    bits_per_pixel, pixel_group_size, packet_size;

        if (!(cfg->bpp == 8 || cfg->bpp == 16 || cfg->bpp == 24 ||
              cfg->bpp == 32))
            return -4;  // input pixel depth not supported

        if (cfg->dt >= 0x10) // long packages
        {
            if (csi2_decode_datatype(cfg->dt, &bits_per_pixel, &pixel_group_size))
                return -5;  // shouldn't happen after data type check above
            /*if (cfg->dt == 0x1e || cfg->dt == 0x1f)
            {
                bits_per_pixel /=
                    2; // bits_per_pixel value for yuv is not correct here
            }*/

            if (cfg->columns % pixel_group_size != 0)
                return -6;  // columns not an integer multiple of pixel group size

            // compute the size of a single long packet of this VC/DT
            packet_size = cfg->columns * cfg->bpp / 8;   // data bytes

            if (packet_size > 0xffff || cfg->columns > 65535)
                return -7;  // too many columns (byte count must fit in 16 bit)

            vc_state[i].word_count = packet_size;   // remember word count for later
            vc_state[i].data_bits = cfg->bpp;

            packet_size += 10;   // add preamble (4), header (4), and crc (2)

            // align long packet size to FPGA's requirement
            packet_size = (packet_size + output_align_mask) & ~output_align_mask;

            vc_state[i].num_padding = packet_size - vc_state[i].word_count - 10;

            // update total counts
            total_lines += cfg->rows;
            total_bytes += packet_size * cfg->rows;
        }
        else
        {
            packet_size = 8;  // add preamble (4), header (4)

            // update total counts
            total_lines += 1;
            total_bytes += packet_size;
        }

        vc_state[i].src = cfg->src;
    }

    if (out == NULL)    // out==NULL requests needed size computation
        return total_bytes;

    if (total_bytes > max_out_bytes)
        return -8;      // doesn't fit into size of output buffer

    uint8_t    *line_cfg = (uint8_t*)malloc(total_lines);

    if (!line_cfg)
        return -9;

    memset(line_cfg, 0xff, total_lines);    // mark all lines "unused"

    int res = 0;

    // validate more VC/DT configuration aspects
    // - no overlapping lines
    // - determine which VC/DT is written to each output line/long-packet
    // - if force_equal_line_size != 0, enforce all vc's word counts to be equal
    for (unsigned i = 0; res == 0 && i < num_vc_dts; i++)
    {
        csi2_encode_params_t   *cfg = vc_dt_config + i;

        if (force_equal_line_size && i > 0)
        {
            // force equal word count values
            if (vc_state[i].word_count != vc_state[i - 1].word_count)
            {
                fprintf(stderr, "Error in CSI-2 endcoder config: "
                        "Lines of VC/DT %d/0x%02x differ in word count\n",
                        cfg->vc, cfg->dt);
                res = -13;

                goto exit_from_func;
            }
        }

        // iterate over all lines of this VC/DT and assign them an index in the
        // list of output lines
        for (uint32_t l = 0; l < cfg->rows; l++)
        {
            uint32_t    output_line = cfg->first_line + l * cfg->line_incr;

            if (output_line >= total_lines)
            {
                fprintf(stderr, "Error in CSI-2 endcoder config: "
                        "Lines of VC/DT %d/0x%02x overrun total line count\n",
                        cfg->vc, cfg->dt);
                res = -10;

                goto exit_from_func;    // error
            }

            if (line_cfg[output_line] != 0xff)
            {
                csi2_encode_params_t    *other_cfg =
                    vc_dt_config + line_cfg[output_line];

                fprintf(stderr, "Error in CSI-2 endcoder config: "
                        "Lines of VC/DT %d/0x%02x overlap with %d/0x%02x\n",
                        cfg->vc, cfg->dt, other_cfg->vc, other_cfg->dt);
                res = -11;

                goto exit_from_func;    // error
            }

            // remember by which VC/DT this output line is going to be filled
            line_cfg[output_line] = (uint8_t)i;
        }
    }

    // If we get here, there are no gaps in the output lines, as they were all
    // found to fall into the valid range and to not overlap with another VC/DT.

    // Now write all long packets/lines to the output buffer in their
    // transmission order
    for (uint32_t l = 0; l < total_lines; l++)
    {
        static uint8_t  preamble[4] = { 0xb8, 0xb8, 0xb8, 0xb8 };

        unsigned idx = line_cfg[l];

        //assert(idx < num_vc_dts);

        vc_state_s  *vc = vc_state + idx;

        // VC/DT config that provides the data for this output line
        csi2_encode_params_t   *cfg = vc_dt_config + idx;

        uint8_t vc_dt = cfg->vc * 64 + cfg->dt;

        // fill in preamble
        memcpy(out, preamble, 4);
        out += 4;

        // fill in packet header
        out[0] = vc_dt;                     // VC/DT

        if (cfg->dt >= 0x10)
        {
            out[1] = vc->word_count % 256;      // word count LSB
            out[2] = vc->word_count / 256;      // word count MSB
        }
        else
        {
            out[1] = vc->src[0];      // data field LSB
            out[2] = vc->src[1];      // data field MSB
        }

        out[3] = 0;                         // clear VCX from CSI-2 ver. 2.0
        if (do_checksums)
            csi2_update_ph_ECC(out);        // insert 6bit header ECC
        out += 4;

        if (cfg->dt >= 0x10)
        {
            uint32_t    bytes_out, pixels_out;

            switch (cfg->bpp)
            {
            case 8:
            case 24:    // special case: RGB888
                pixels_out = csi2_encode_raw8(out, cfg->columns, vc->src,
                                              vc->data_bits);
                break;

            case 16:
                pixels_out = csi2_encode_raw16(out, cfg->columns,
                                               (uint16_t*)vc->src,
                                               vc->data_bits);
                break;

            case 32:
            default:
                pixels_out = csi2_encode_raw32(out, cfg->columns,
                                               (uint32_t*)vc->src,
                                               vc->data_bits);
                break;
            }

            bytes_out = pixels_out * vc->data_bits / 8;

            if (bytes_out != vc->word_count)
            {
                res = -12;
                goto exit_from_func;
            }

            // payload CRC
            uint16_t crc = do_checksums ? csi2_payload_checksum(out, bytes_out) : 0;

            out += bytes_out;
            *out++ = crc % 256;
            *out++ = crc / 256;

            // add padding for alignment required by FPGA
            if (vc->num_padding > 0)
            {
                memset(out, 0, vc->num_padding);
                out += vc->num_padding;
            }

            // increment line counter for this VC/DT
            vc->src += cfg->line_pitch;
            vc->line_cnt++;
            res += bytes_out + 10 + vc->num_padding; // 10 = preamble + header + CRC
        }
        else
        {
            res += 8;
        }
    }

    exit_from_func:
    free(line_cfg);

    return res;
}


/** Convenience function to make calling \ref csi2_multi_encode() easier, if the
 *  source data has a simple layout consisting of only one virtual channel and
 *  data type. The virtual channel in the CSI-2 output will be set to 0.
 *
 * @note    Input data is expected to be aligned to the lowest significant bit.
 *
 * @param out           Pointer to first output byte (directly after header)
 * @param max_out_bytes Max.\ number of bytes that can be written to out
 * @param out_align     Number of bits each long packet needs to be aligned to.
 *                      Allowed are only 32 or 64, depending on card version.
 * @param src           Pointer to first input pixel to copy
 * @param columns       Number of pixels per input row to copy
 * @param rows          Number of input rows to copy to  output
 * @param bpp           Bits per input pixel (8 or 16)
 * @param line_pitch    Number of bytes per input line with padding
 * @param data_type     Data type to encode to (0x28..0x2f)
 * @param do_checksums  0 = write dummy checksums, 1 = compute CRC and ECC
 *
 * @return  positive: Number of payload bytes written
 * @return  negative: Error (see \ref csi2_multi_encode() for error codes)
 */
int csi2_single_encode(uint8_t *out, uint32_t max_out_bytes, uint32_t out_align,
                       uint8_t *src, uint32_t columns, uint32_t rows,
                       uint32_t bpp, uint32_t line_pitch,
                       uint8_t data_type, int do_checksums)
{
    csi2_encode_params_t    parm;

    parm.src = src;
    parm.line_pitch = line_pitch;
    parm.bpp = bpp;
    parm.columns = columns;
    parm.rows = rows;

    parm.vc = 0;
    parm.dt = data_type;
    parm.first_line = 0;
    parm.line_incr = 1;

    return csi2_multi_encode(out, max_out_bytes, out_align, &parm, 1,
                             do_checksums, 0);
}


/** Encode 8bit raw pixel values to CSI-2 encoded output bytes. Input pixel
 *  values must be aligned at the least-significant bit. This function can only
 *  be used to encode complete pixel groups. The size of a pixel group depends
 *  on the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output pointer for CSI-2-encoded data.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       8bit raw input data pointer. Must point to the first pixel
 *                  of a pixel group.
 * @param data_bits Number of data bits per pixel in the encoded CSI-2 stream.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_encode_raw8(uint8_t *__restrict dest, uint32_t n_pixels,
                          uint8_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    default:
        break;

    case 6:
        for (; i + 3 < n_pixels; i += 4)
        {
            dest[0] = ((src[0] & 0x3f) >> 0) | ((src[1] & 0x03) << 6);
            dest[1] = ((src[1] & 0x3c) >> 2) | ((src[2] & 0x0f) << 4);
            dest[2] = ((src[2] & 0x30) >> 4) | ((src[3] & 0x3f) << 2);
            dest += 3;      // per 3 output bytes we consume 4 input pixels
            src += 4;
        }
        break;

    case 7:
        for (; i + 7 < n_pixels; i += 8)
        {
            dest[0] = ((src[0] & 0x7f) >> 0) | ((src[1] & 0x01) << 7);
            dest[1] = ((src[1] & 0x7e) >> 1) | ((src[2] & 0x03) << 6);
            dest[2] = ((src[2] & 0x7c) >> 2) | ((src[3] & 0x07) << 5);
            dest[3] = ((src[3] & 0x78) >> 3) | ((src[4] & 0x0f) << 4);
            dest[4] = ((src[4] & 0x70) >> 4) | ((src[5] & 0x1f) << 3);
            dest[5] = ((src[5] & 0x60) >> 5) | ((src[6] & 0x3f) << 2);
            dest[6] = ((src[6] & 0x40) >> 6) | ((src[7] & 0x7f) << 1);
            dest += 7;      // per 7 output bytes we consume 8 input pixels
            src += 8;
        }
        break;

    case 8:
        memcpy(dest, src, i = n_pixels);
        break;

    case 10:
        for (; i + 3 < n_pixels; i += 4)
        {
            // write top 8 of 10 pixel bits with 8bit input value
            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = src[2];
            dest[3] = src[3];
            dest[4] = 0;    // lowest 2 of 10 bits are set to 0
            dest += 5;      // per 5 output bytes we consume 4 input pixels
            src += 4;
        }
        break;

    case 12:
        for (; i + 1 < n_pixels; i += 2)
        {
            // write top 8 of 12 pixel bits with 8bit input value
            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = 0;    // lowest 4 of 12 bits are set to 0
            dest += 3;      // per 3 output bytes we consume 2 input pixels
            src += 2;
        }
        break;

    case 14:
        for (; i + 3 < n_pixels; i += 4)
        {
            // write top 8 of 14 pixel bits with 8bit input value
            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = src[2];
            dest[3] = src[3];
            dest[4] = 0;    // lowest 6 of 14 bits are set to 0
            dest[5] = 0;
            dest[6] = 0;
            dest += 7;      // per 7 output bytes we consume 4 input pixels
            src += 4;
        }

    case 16:
        for (; i < n_pixels; i++)
        {
            // write top 8 of 16 pixel bits to 8bit output value
            dest[0] = src[0];
            dest[1] = 0;    // lowest 8 of 16 bits are set to 0
            dest += 2;      // per 2 output bytes we consume 1 input pixel
            src += 1;
        }
        break;

    case 20:
        for (; i + 1 < n_pixels; i += 2)
        {
            // write top 8 of 20 output pixel bits from 8bit input value
            dest[0] = src[0];   // upper 8 bits of first ooutput pixel
            dest[1] = 0;        // middle 8 of 20 bits are set to 0
            dest[2] = src[1];   // upper 8 bits of second ooutput pixel
            dest[3] = 0;        // middle 8 of 20 bits are set to 0
            dest[4] = 0;    // lowest 4 of 20 bits are set to 0 for both pixels
            dest += 3;      // per 5 output bytes we consume 2 input pixels
            src += 1;
        }
        break;

    case 24:
        // NOTE: this is a special case for RGB888 data, as there is no CSI-2
        //       data type "RAW24"
        memcpy(dest, src, 3 * (i = n_pixels));
        break;
    }

    return i;
}


/** Encode 16bit raw pixel values to CSI-2 encoded output bytes. Input pixel
 *  values must be aligned at the least-significant bit. This function can only
 *  be used to encode complete pixel groups. The size of a pixel group depends
 *  on the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output pointer for CSI-2-encoded data.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       16bit raw input data pointer. Must point to the first pixel
 *                  of a pixel group.
 * @param data_bits Number of data bits per pixel in the encoded CSI-2 stream.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_encode_raw16(uint8_t *__restrict dest, uint32_t n_pixels,
                           uint16_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    default:
        break;

    case 6:
        for (; i + 3 < n_pixels; i += 4)
        {
            dest[0] = ((src[0] & 0x3f) >> 0) | ((src[1] & 0x03) << 6);
            dest[1] = ((src[1] & 0x3c) >> 2) | ((src[2] & 0x0f) << 4);
            dest[2] = ((src[2] & 0x30) >> 4) | ((src[3] & 0x3f) << 2);
            dest += 3;      // per 3 output bytes we consume 4 input pixels
            src += 4;
        }
        break;

    case 7:
        for (; i + 7 < n_pixels; i += 8)
        {
            dest[0] = ((src[0] & 0x7f) >> 0) | ((src[1] & 0x01) << 7);
            dest[1] = ((src[1] & 0x7e) >> 1) | ((src[2] & 0x03) << 6);
            dest[2] = ((src[2] & 0x7c) >> 2) | ((src[3] & 0x07) << 5);
            dest[3] = ((src[3] & 0x78) >> 3) | ((src[4] & 0x0f) << 4);
            dest[4] = ((src[4] & 0x70) >> 4) | ((src[5] & 0x1f) << 3);
            dest[5] = ((src[5] & 0x60) >> 5) | ((src[6] & 0x3f) << 2);
            dest[6] = ((src[6] & 0x40) >> 6) | ((src[7] & 0x7f) << 1);
            dest += 7;      // per 7 output bytes we consume 8 input pixels
            src += 8;
        }
        break;

    case 8:
        for (; i < n_pixels; i++)
        {
            // write lower 8 of 16 pixel bits to each 8bit output value
            *dest++ = *src++ & 0xff;  // per output byte we consume 1 input pixel
        }
        break;

    case 10:
        for (; i + 3 < n_pixels; i += 4)
        {
            // write top 8 of the 4x 10 bits of 4 input pixels
            dest[0] = (src[0] >> 2) & 0xff;
            dest[1] = (src[1] >> 2) & 0xff;
            dest[2] = (src[2] >> 2) & 0xff;
            dest[3] = (src[3] >> 2) & 0xff;
            dest[4] =       // write lowest 2 bits of each input pixel
                ((src[0] & 0x03) << 0) | ((src[1] & 0x03) << 2) |
                ((src[2] & 0x03) << 4) | ((src[3] & 0x03) << 6);
            dest += 5;      // per 5 output bytes we consume 4 input pixels
            src += 4;
        }
        break;

    case 12:
        for (; i + 1 < n_pixels; i += 2)
        {
            // write top 8 of 12 pixel bits with 8bit input value
            dest[0] = (src[0] >> 4) & 0xff;
            dest[1] = (src[1] >> 4) & 0xff;
            dest[2] = ((src[0] & 0x0f) << 0) | ((src[1] & 0x0f) << 4);
            dest += 3;      // per 3 output bytes we consume 2 input pixels
            src += 2;
        }
        break;

    case 14:
        for (; i + 3 < n_pixels; i += 4)
        {
            // write top 8 of 14 pixel bits with 8bit input value
            dest[0] = (src[0] >> 6) & 0xff;
            dest[1] = (src[1] >> 6) & 0xff;
            dest[2] = (src[2] >> 6) & 0xff;
            dest[3] = (src[3] >> 6) & 0xff;
            // collect lower 6 bits of the 4 pixels in 3 more bytes
            dest[4] = ((src[0] >> 0) & 0x3f) | ((src[1] & 0x03) << 6);
            dest[5] = ((src[1] >> 2) & 0x0f) | ((src[2] & 0x0f) << 4);
            dest[6] = ((src[2] >> 4) & 0x03) | ((src[3] & 0x3f) << 2);
            dest += 7;      // per 7 output bytes we consume 4 input pixels
            src += 4;
        }

    case 16:
        for (; i < n_pixels; i++)
        {
            dest[0] = src[0] / 256; // upper 8 bits of the pixel (MSB)
            dest[1] = src[0] % 256; // lower 8 bits of the pixel (LSB)
            dest += 2;      // per 2 output bytes we consume 1 input pixel
            src += 1;
        }
        break;

    case 20:
        for (; i + 1 < n_pixels; i += 2)
        {
            // write top 16 of 20 pixel bits from 16bit input values
            dest[0] = src[0] / 256; // upper 8 bits of the first pixel (MSB)
            dest[1] = src[0] % 256; // lower 8 bits of the first pixel (LSB)
            dest[2] = src[1] / 256; // upper 8 bits of the second pixel (MSB)
            dest[3] = src[1] % 256; // lower 8 bits of the second pixel (LSB)
            dest[4] = 0;    // lowest 4 of 20 bits are set to 0 for both pixels
            dest += 5;      // per 5 output bytes we consume 2 input pixels
            src += 2;
        }
        break;
    }

    return i;
}

uint32_t csi2_encode_raw32(uint8_t *__restrict dest, uint32_t n_pixels,
                           uint32_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    default:
        break;

    case 6:
        for (; i + 3 < n_pixels; i += 4)
        {
            dest[0] = ((src[0] & 0x3f) >> 0) | ((src[1] & 0x03) << 6);
            dest[1] = ((src[1] & 0x3c) >> 2) | ((src[2] & 0x0f) << 4);
            dest[2] = ((src[2] & 0x30) >> 4) | ((src[3] & 0x3f) << 2);
            dest += 3;      // per 3 output bytes we consume 4 input pixels
            src += 4;
        }
        break;

    case 7:
        for (; i + 7 < n_pixels; i += 8)
        {
            dest[0] = ((src[0] & 0x7f) >> 0) | ((src[1] & 0x01) << 7);
            dest[1] = ((src[1] & 0x7e) >> 1) | ((src[2] & 0x03) << 6);
            dest[2] = ((src[2] & 0x7c) >> 2) | ((src[3] & 0x07) << 5);
            dest[3] = ((src[3] & 0x78) >> 3) | ((src[4] & 0x0f) << 4);
            dest[4] = ((src[4] & 0x70) >> 4) | ((src[5] & 0x1f) << 3);
            dest[5] = ((src[5] & 0x60) >> 5) | ((src[6] & 0x3f) << 2);
            dest[6] = ((src[6] & 0x40) >> 6) | ((src[7] & 0x7f) << 1);
            dest += 7;      // per 7 output bytes we consume 8 input pixels
            src += 8;
        }
        break;

    case 8:
        for (; i < n_pixels; i++)
        {
            // write lower 8 of 16 pixel bits to each 8bit output value
            *dest++ = *src++ & 0xff;  // per output byte we consume 1 input pixel
        }
        break;

    case 10:
        for (; i + 3 < n_pixels; i += 4)
        {
            // write top 8 of the 4x 10 bits of 4 input pixels
            dest[0] = (src[0] >> 2) & 0xff;
            dest[1] = (src[1] >> 2) & 0xff;
            dest[2] = (src[2] >> 2) & 0xff;
            dest[3] = (src[3] >> 2) & 0xff;
            dest[4] =       // write lowest 2 bits of each input pixel
                ((src[0] & 0x03) << 0) | ((src[1] & 0x03) << 2) |
                ((src[2] & 0x03) << 4) | ((src[3] & 0x03) << 6);
            dest += 5;      // per 5 output bytes we consume 4 input pixels
            src  += 4;
        }
        break;

    case 12:
        for (; i + 1 < n_pixels; i += 2)
        {
            // write top 8 of 12 pixel bits with 8bit input value
            dest[0] = (src[0] >> 4) & 0xff;
            dest[1] = (src[1] >> 4) & 0xff;
            dest[2] = ((src[0] & 0x0f) << 0) | ((src[1] & 0x0f) << 4);
            dest += 3;      // per 3 output bytes we consume 2 input pixels
            src += 2;
        }
        break;

    case 14:
        for (; i + 3 < n_pixels; i += 4)
        {
            // write top 8 of 14 pixel bits with 8bit input value
            dest[0] = (src[0] >> 6) & 0xff;
            dest[1] = (src[1] >> 6) & 0xff;
            dest[2] = (src[2] >> 6) & 0xff;
            dest[3] = (src[3] >> 6) & 0xff;
            // collect lower 6 bits of the 4 pixels in 3 more bytes
            dest[4] = ((src[0] >> 0) & 0x3f) | ((src[1] & 0x03) << 6);
            dest[5] = ((src[1] >> 2) & 0x0f) | ((src[2] & 0x0f) << 4);
            dest[6] = ((src[2] >> 4) & 0x03) | ((src[3] & 0x3f) << 2);
            dest += 7;      // per 7 output bytes we consume 4 input pixels
            src += 4;
        }
        break;

    case 16:
        for (; i < n_pixels; i++)
        {
            dest[0] = src[0] / 256; // upper 8 bits of the pixel (MSB)
            dest[1] = src[0] % 256; // lower 8 bits of the pixel (LSB)
            dest += 2;      // per 2 output bytes we consume 1 input pixel
            src += 1;
        }
        break;

    case 20:
        for (; i + 1  < n_pixels; i += 2)
        {
            // write top 10 of the 2x 20 bits of 2 input pixels
            dest[0] = ((src[0] >> 12) & 0xff);
            dest[1] = ((src[0] >> 2)  & 0xff);
            dest[2] = ((src[1] >> 12) & 0xff);
            dest[3] = ((src[1] >> 2)  & 0xff);
            // write lowest 2 bits of each input pixel

            // collect lower 4 bits of the 2 pixels in 1 more byte
            dest[4] = (  ((src[0] & 0x0C00) >> 10) | ((src[0] & 0x0003) << 2)
                       | ((src[1] & 0x0C00) >> 6)  | ((src[1] & 0x0003) << 6));
            dest += 5;      // per 5 output bytes we consume 4 input pixels
            src  += 2;

        }
        break;

    case 24:
        // NOTE: this is a special case for RGB888 data, as there is no CSI-2
        //       data type "RAW24"
        for (; i < n_pixels; i++)
        {
            // write lower 24 of 32 pixel bits as 3 8bit output values
            dest[0] = (*src >> 0) & 0xff;
            dest[1] = (*src >> 8) & 0xff;
            dest[2] = (*src >> 16) & 0xff;
            dest += 3;  // per 3 output bytes we consume 1 input pixel
            src += 1;
        }
        break;
    }

    return i;
}

