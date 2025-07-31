/** @file csi-2.h
 *
 * Declarations for CSI-2 image decoding and encoding.
 */
#pragma once
#ifndef CSI2_H_
#define CSI2_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Parameters needed to encode a part of a CSI-2 image relating to a single
 *  VirtualChannel/DataType combination.
 */
typedef struct
{
    /* input description */
    uint8_t    *src;        /**< Pointer to first input pixel to copy */
    uint32_t    line_pitch; /**< Number of bytes per input line with padding */
    uint32_t    bpp;        /**< Bits per input pixel (8 or 16) */
    uint32_t    columns;    /**< Number of pixels per input row to copy */
    uint32_t    rows;       /**< Number of input rows to copy to  output */

    /* output setup */
    uint32_t    vc;         /**< Virtual Channel to output */
    uint32_t    dt;         /**< Data Type to encode to (only RAW supported) */
    uint32_t    first_line; /**< Index of first long packet with this VC/DT */
    uint32_t    line_incr;  /**< Number of lines of a different VC/DT that are
                             *   inserted bewteen two lines of this VC/DT plus 1
                             */
} csi2_encode_params_t;


/* Color conversion metrix declarations for use in csi2_decode_uyvyXXX_ccm */
extern const float CCM_LIMITED_RANGE_BT_601[12];
extern const float CCM_LIMITED_RANGE_BT_709[12];
extern const float CCM_FULL_RANGE_BT_601[12];
extern const float CCM_FULL_RANGE_BT_709[12];


void csi2_update_ph_ECC(uint8_t *ph);
uint16_t csi2_payload_checksum(uint8_t const *data, uint32_t wordcount);

uint8_t* csi2_parse_dphy_ph(uint8_t *pdata, uint8_t *vc_dt,
                            uint32_t *word_count);
int csi2_decode_datatype(unsigned data_type, uint32_t *bits_per_pixel,
                         uint32_t *pixel_group_size);

int is_raw(uint32_t data_type);
int is_rgb(uint32_t data_type);
int is_yuv(uint32_t data_type);

uint32_t csi2_decode_raw8(uint8_t *__restrict dest, uint32_t n_pixels,
                          uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_raw8_msb(uint8_t *__restrict dest, uint32_t n_pixels,
                              uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_raw16(uint16_t *__restrict dest, uint32_t n_pixels,
                           uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_raw16_msb(uint16_t *__restrict dest, uint32_t n_pixels,
                               uint8_t *__restrict src, uint32_t data_bits);

uint32_t csi2_decode_raw32(uint32_t *__restrict dest, uint32_t n_pixels,
                           uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_raw32_msb(uint32_t *__restrict dest, uint32_t n_pixels,
                               uint8_t *__restrict src, uint32_t data_bits);

uint32_t csi2_decode_uyvy8_24_generic(uint8_t* __restrict dest, uint32_t n_pixels,
                                      uint8_t* __restrict src, uint32_t data_bits,
                                      int output_format);
uint32_t csi2_decode_uyvy8_24_generic_ccm(uint8_t* __restrict dest, uint32_t n_pixels,
                                          uint8_t* __restrict src, uint32_t data_bits,
                                          int output_format, const float ccm[12]);
uint32_t csi2_decode_uyvy8_24(uint8_t *__restrict dest, uint32_t n_pixels,
                              uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_uyvy8_24_rgb(uint8_t* __restrict dest, uint32_t n_pixels,
                                  uint8_t* __restrict src, uint32_t data_bits);
uint32_t csi2_decode_uyvy8_24_bgr(uint8_t* __restrict dest, uint32_t n_pixels,
                                  uint8_t* __restrict src, uint32_t data_bits);
uint32_t csi2_decode_uyvy8_32_generic(uint8_t* __restrict dest, uint32_t n_pixels,
                                      uint8_t* __restrict src, uint32_t data_bits,
                                      int output_format);
uint32_t csi2_decode_uyvy8_32_generic_ccm(uint8_t* __restrict dest, uint32_t n_pixels,
                                          uint8_t* __restrict src, uint32_t data_bits,
                                          int output_format, const float ccm[12]);
uint32_t csi2_decode_uyvy8_32(uint8_t *__restrict dest, uint32_t n_pixels,
                              uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_uyvy8_32_rgba(uint8_t* __restrict dest, uint32_t n_pixels,
                                   uint8_t* __restrict src, uint32_t data_bits);
uint32_t csi2_decode_uyvy8_32_bgra(uint8_t* __restrict dest, uint32_t n_pixels,
                                   uint8_t* __restrict src, uint32_t data_bits);

uint32_t csi2_decode_rgb8_24_generic(uint8_t* __restrict dest, uint32_t n_pixels,
                                     uint8_t* __restrict src, uint32_t data_bits,
                                     int output_format);
uint32_t csi2_decode_rgb8_24(uint8_t *__restrict dest, uint32_t n_pixels,
                             uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_rgb8_24_rgb(uint8_t* __restrict dest, uint32_t n_pixels,
                                 uint8_t* __restrict src, uint32_t data_bits);
uint32_t csi2_decode_rgb8_24_bgr(uint8_t* __restrict dest, uint32_t n_pixels,
                                 uint8_t* __restrict src, uint32_t data_bits);

uint32_t csi2_decode_rgb8_32_generic(uint8_t* __restrict dest, uint32_t n_pixels,
                                     uint8_t* __restrict src, uint32_t data_bits,
                                     int output_format);
uint32_t csi2_decode_rgb8_32(uint8_t *__restrict dest, uint32_t n_pixels,
                             uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_decode_rgb8_32_rgba(uint8_t* __restrict dest, uint32_t n_pixels,
                                  uint8_t* __restrict src, uint32_t data_bits);
uint32_t csi2_decode_rgb8_32_bgra(uint8_t* __restrict dest, uint32_t n_pixels,
                                  uint8_t* __restrict src, uint32_t data_bits);

int csi2_multi_encode(uint8_t *out, uint32_t max_out_bytes, uint32_t out_align,
                      csi2_encode_params_t *vc_dt_config, unsigned num_vc_dts,
                      int do_checksums, int force_equal_line_size);
int csi2_single_encode(uint8_t *out, uint32_t max_out_bytes, uint32_t out_align,
                       uint8_t *src, uint32_t columns, uint32_t rows,
                       uint32_t bpp, uint32_t line_pitch,
                       uint8_t data_type, int do_checksums);

uint32_t csi2_encode_raw8(uint8_t *__restrict dest, uint32_t n_pixels,
                          uint8_t *__restrict src, uint32_t data_bits);
uint32_t csi2_encode_raw16(uint8_t *__restrict dest, uint32_t n_pixels,
                           uint16_t *__restrict src, uint32_t data_bits);
uint32_t csi2_encode_raw32(uint8_t *__restrict dest, uint32_t n_pixels,
                           uint32_t *__restrict src, uint32_t data_bits);

#ifdef __cplusplus
}
#endif
#endif /* CSI2_H_ */
