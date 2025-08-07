/** @file img_decode.cpp
 *
 * Implementation of CSI-2 image decoding.
 */
#include <sxpf/csi-2.h>
#include <string.h>

/** Limited Range BT.601 color conversion matrix from YCbCr to full range RGB */
const float CCM_LIMITED_RANGE_BT_601[12] =
{
    1.168951f,       0.0f,  1.602285f,  -16.0f,
    1.168951f, -0.393297f, -0.816155f, -128.0f,
    1.168951f,  2.025144f,       0.0f, -128.0f
};

/** Limited Range BT.709 color conversion matrix from YCbCr to full range RGB */
const float CCM_LIMITED_RANGE_BT_709[12] =
{
    1.168949f,       0.0f, 1.799770f,  -16.0f,
    1.168949f, -0.214085f,   -0.535f, -128.0f,
    1.168949f,  2.120685f,      0.0f, -128.0f
};

/** Full Range BT.601 color conversion matrix from YCbCr to full range RGB */
const float CCM_FULL_RANGE_BT_601[12] =
{
    1.0f,       0.0f,     1.402f,    0.0f,
    1.0f, -0.344136f, -0.714136f, -128.0f,
    1.0f,     1.772f,       0.0f, -128.0f
};

/** Full Range BT.709 color conversion matrix from YCbCr to full range RGB */
const float CCM_FULL_RANGE_BT_709[12] =
{
    1.0f,       0.0f,    1.5748f,    0.0f,
    1.0f, -0.187324f, -0.468124f, -128.0f,
    1.0f,    1.8556f,       0.0f, -128.0f
};


/** Parse and verify a captured D-PHY CSI-2 data (long) packet's header
 *
 * @param pdata        Pointer to start of captured packet.
 * @param vc_dt        Pointer to variable where to store the packet's Data
 *                     Identifier (virtual channel & data type).
 * @param data_field   Pointer to variable where to store the number of data
 *                     bytes in the packet (long packet) or the data field
 *                     of the header (short packet)
 *
 * @return \c nullptr on error
 * @return Pointer to pixel data (long packet) or next CSI packet (short packet)
 *         on success
 */
uint8_t* csi2_parse_dphy_ph(uint8_t *pdata, uint8_t *vc_dt,
                            uint32_t *data_field)
{
    if (!pdata || !vc_dt || !data_field || (uintptr_t)pdata & 1)
        return nullptr;

    // verify D-PHY preamble
    if (*(uint32_t*)pdata != 0xb8b8b8b8)
        return nullptr;

    // TODO verify header ECC

    // extract virtual channel and data type
    *vc_dt = pdata[4];

    // extract word count (long packet) or data field (short packet)
    *data_field = pdata[5] + 256u * pdata[6];

    // return pointer to payload data (long packet) or
    // pointer to next CSI packet (short packet)
	return pdata + 8;
}


/** Return image parameters encoded by the passed data type.
 *
 * @param data_type         Data type of CSI-2 packet.
 * @param bits_per_pixel    Number of bits occupied per pixel in CSI-2 data
 *                          packet.
 * @param pixel_group_size  Number of pixels in a group that occupy an integer
 *                          multiple of 8 bits.
 * @return 0 on success
 * @return 1 if the data type is not supported or an output pointer was \c NULL
 */
int csi2_decode_datatype(unsigned data_type, uint32_t *bits_per_pixel,
                         uint32_t *pixel_group_size)
{
    if (!bits_per_pixel || !pixel_group_size)
        return 1;

    auto decode = [bits_per_pixel,pixel_group_size](uint32_t bpp, uint32_t pg)
    {
        *bits_per_pixel = bpp;
        *pixel_group_size = pg;
    };

    switch (data_type)
    {
    default:
        break;

    // generic Long Packet Data Types (e.g., embedded data lines)
    case 0x10:  // fall through
    case 0x11:  // fall through
    case 0x12:  // fall through
    case 0x13:  // fall through
    case 0x14:  // fall through
    case 0x15:  // fall through
    case 0x16:  // fall through
    case 0x17:  decode(8, 1); return 0;     // treat it like RAW8

    // YUV-422 image data types
    // note: bits-per-pixel values encode one luminance and one chrominance
    // value together
    case 0x1e:  decode(16, 2); return 0;    // YUV422 8-bit (UYVY)
    case 0x1f:  decode(20, 2); return 0;    // YUV422 10-bit (UYVY)

    // RGB image data types
    // note: bits-per-pixel values cover all color channels for one pixel
    case 0x24:  decode(24, 1); return 0;    // RGB888 (BGR byte sequence)

    // RAW image data types
    case 0x28:  decode( 6, 4); return 0;    // RAW6
    case 0x29:  decode( 7, 8); return 0;    // RAW7
    case 0x2a:  decode( 8, 1); return 0;    // RAW8
    case 0x2b:  decode(10, 4); return 0;    // RAW10
    case 0x2c:  decode(12, 2); return 0;    // RAW12
    case 0x2d:  decode(14, 4); return 0;    // RAW14
    case 0x2e:  decode(16, 1); return 0;    // RAW16
    case 0x2f:  decode(20, 2); return 0;    // RAW20


    // user-defined byte-based data; treated like RAW8
    case 0x30:  // fall through
    case 0x31:  // fall through
    case 0x32:  // fall through
    case 0x33:  // fall through
    case 0x34:  // fall through
    case 0x35:  // fall through
    case 0x36:  // fall through
    case 0x37:  decode(8, 1); return 0;
    }

    return 1;   // not supported
}


/** Check, whether a data type code represents RAW data.
 *
 * @param data_type Data type code to check (0..63).
 *
 * @return  \c true for a supported RAW data type, \c false otherwise
 */
int is_raw(uint32_t data_type)
{
    return data_type >= 0x28 && data_type <= 0x2f;
}


/** Check, whether a data type code represents RGB data.
 *
 * @param data_type Data type code to check (0..63).
 *
 * @return  \c true for a supported RGB data type, \c false otherwise
 */
int is_rgb(uint32_t data_type)
{
    return data_type == 0x24;
}


/** Check, whether a data type code represents YUV data.
 *
 * @param data_type Data type code to check (0..63).
 *
 * @return  \c true for a supported YUV data type, \c false otherwise
 */
int is_yuv(uint32_t data_type)
{
    return data_type >= 0x1e && data_type <= 0x1f;
}


/** Decode CSI-2 encoded pixels to 8bit output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_raw8(uint8_t *__restrict dest, uint32_t n_pixels,
                          uint8_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    case 6:
        while (i + 3 < n_pixels)
        {
            // align 6 pixel bits at LSB of 8bit output value
            dest[i++] =   src[0] & 0x3f;
            dest[i++] = ((src[1] & 0x0f) << 2) | ((src[0] & 0xc0) >> 6);
            dest[i++] = ((src[2] & 0x03) << 4) | ((src[1] & 0xf0) >> 4);
            dest[i++] =                           (src[2] & 0xfc) >> 2;
            src += 3;   // per 4 output pixels we consume 3 input bytes
        }
        break;

    case 7:
        while (i + 7 < n_pixels)
        {
            // align 7 pixel bits at LSB of 8bit output value
            dest[i++] =   src[0] & 0x7f;
            dest[i++] = ((src[1] & 0x3f) << 1) | ((src[0] & 0x80) >> 7);
            dest[i++] = ((src[2] & 0x1f) << 2) | ((src[1] & 0xc0) >> 6);
            dest[i++] = ((src[3] & 0x0f) << 3) | ((src[2] & 0xe0) >> 5);
            dest[i++] = ((src[4] & 0x07) << 4) | ((src[3] & 0xf0) >> 4);
            dest[i++] = ((src[5] & 0x03) << 5) | ((src[4] & 0xf8) >> 3);
            dest[i++] = ((src[6] & 0x01) << 6) | ((src[5] & 0xfc) >> 2);
            dest[i++] =                           (src[6] & 0xfe) >> 1;
            src += 7;   // per 8 output pixels we consume 7 input bytes
        }
        break;

    case 8:
        memcpy(dest, src, i = n_pixels);
        break;

    case 10:
        while (i + 3 < n_pixels)
        {
            // write top 8 of 10 pixel bits to 8bit output value
            dest[i++] = src[0];
            dest[i++] = src[1];
            dest[i++] = src[2];
            dest[i++] = src[3];
            src += 5;   // per 4 output pixels we consume 5 input bytes
        }
        break;

    case 12:
        while (i + 1 < n_pixels)
        {
            // write top 8 of 12 pixel bits to 8bit output value
            dest[i++] = src[0];
            dest[i++] = src[1];
            src += 3;   // per 2 output pixels we consume 3 input bytes
        }
        break;

    case 14:
        while (i + 3 < n_pixels)
        {
            // write top 8 of 14 pixel bits to 8bit output value
            dest[i++] = src[0];
            dest[i++] = src[1];
            dest[i++] = src[2];
            dest[i++] = src[3];
            src += 7;   // per 4 output pixels we consume 7 input bytes
        }
        break;


    case 16:
        while (i < n_pixels)
        {
            // write top 8 of 16 pixel bits to 8bit output value
            dest[i++] = src[0];
            src += 2;   // per output pixel we consume 2 input bytes
        }
        break;
    }

    return i;
}


/** Decode CSI-2 encoded pixels to 8bit output pixels. Output pixel values
 *  will be aligned at the most-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_raw8_msb(uint8_t *__restrict dest, uint32_t n_pixels,
                              uint8_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    case 6:
        while (i + 3 < n_pixels)
        {
            // align 6 pixel bits at MSB of 8bit output value
            dest[i++] =  (src[0] & 0x3f) << 2;
            dest[i++] = ((src[1] & 0x0f) << 4) | ((src[0] & 0xc0) >> 4);
            dest[i++] = ((src[2] & 0x03) << 6) | ((src[1] & 0xf0) >> 2);
            dest[i++] =                            src[2] & 0xfc;
            src += 3;   // per 4 output pixels we consume 3 input bytes
        }
        break;

    case 7:
        while (i + 7 < n_pixels)
        {
            // align 7 pixel bits at MSB of 8bit output value
            dest[i++] =  (src[0] & 0x7f) << 1;
            dest[i++] = ((src[1] & 0x3f) << 2) | ((src[0] & 0x80) >> 6);
            dest[i++] = ((src[2] & 0x1f) << 3) | ((src[1] & 0xc0) >> 5);
            dest[i++] = ((src[3] & 0x0f) << 4) | ((src[2] & 0xe0) >> 4);
            dest[i++] = ((src[4] & 0x07) << 5) | ((src[3] & 0xf0) >> 3);
            dest[i++] = ((src[5] & 0x03) << 6) | ((src[4] & 0xf8) >> 2);
            dest[i++] = ((src[6] & 0x01) << 7) | ((src[5] & 0xfc) >> 1);
            dest[i++] =                            src[6] & 0xfe;
            src += 7;   // per 8 output pixels we consume 7 input bytes
        }
        break;

    default:
        // all other data types deliver at least 8 bit, so they decode the same
        // as with the LSB-aligned decoder
        i = csi2_decode_raw8(dest, n_pixels, src, data_bits);
        break;
    }

    return i;
}


/** Decode CSI-2 encoded pixels to 16bit output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_raw16(uint16_t *__restrict dest, uint32_t n_pixels,
                           uint8_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    case 6:
        while (i + 3 < n_pixels)
        {
            // align 6 pixel bits at LSB of 16bit output value
            dest[i++] =   src[0] & 0x3f;
            dest[i++] = ((src[1] & 0x0f) << 2) | ((src[0] & 0xc0) >> 6);
            dest[i++] = ((src[2] & 0x03) << 4) | ((src[1] & 0xf0) >> 4);
            dest[i++] =                           (src[2] & 0xfc) >> 2;
            src += 3;   // per 4 output pixels we consume 3 input bytes
        }
        break;

    case 7:
        while (i + 7 < n_pixels)
        {
            // align 7 pixel bits at LSB of 16bit output value
            dest[i++] =   src[0] & 0x7f;
            dest[i++] = ((src[1] & 0x3f) << 1) | ((src[0] & 0x80) >> 7);
            dest[i++] = ((src[2] & 0x1f) << 2) | ((src[1] & 0xc0) >> 6);
            dest[i++] = ((src[3] & 0x0f) << 3) | ((src[2] & 0xe0) >> 5);
            dest[i++] = ((src[4] & 0x07) << 4) | ((src[3] & 0xf0) >> 4);
            dest[i++] = ((src[5] & 0x03) << 5) | ((src[4] & 0xf8) >> 3);
            dest[i++] = ((src[6] & 0x01) << 6) | ((src[5] & 0xfc) >> 2);
            dest[i++] =                           (src[6] & 0xfe) >> 1;
            src += 7;   // per 8 output pixels we consume 7 input bytes
        }
        break;

    case 8:
        while (i < n_pixels)
        {
            // align 8 pixel bits at LSB of 16bit output value
            dest[i++] = src[0];
            src += 1;   // per 1 output pixel we consume 1 input byte
        }
        break;

    case 10:
        while (i + 3 < n_pixels)
        {
            // align 10 pixel bits at LSB of 16bit output value
            dest[i++] = (src[0] << 2) |  (src[4] & 0x03);
            dest[i++] = (src[1] << 2) | ((src[4] & 0x0c) >> 2);
            dest[i++] = (src[2] << 2) | ((src[4] & 0x30) >> 4);
            dest[i++] = (src[3] << 2) | ((src[4] & 0xc0) >> 6);
            src += 5;   // per 4 output pixels we consume 5 input bytes
        }
        break;

    case 12:
        while (i + 1 < n_pixels)
        {
            // align 12 pixel bits at LSB of 16bit output value
            dest[i++] = (src[0] << 4) |  (src[2] & 0x0f);
            dest[i++] = (src[1] << 4) | ((src[2] & 0xf0) >> 4);
            src += 3;   // per 2 output pixels we consume 3 input bytes
        }
        break;

    case 14:
        while (i + 3 < n_pixels)
        {
            // align 14 pixel bits at LSB of 16bit output value
            dest[i++] = (src[0] << 6) |  (src[4] & 0x3f);
            dest[i++] = (src[1] << 6) | ((src[5] & 0x0f) << 2) | ((src[4] & 0xc0) >> 6);
            dest[i++] = (src[2] << 6) | ((src[6] & 0x03) << 4) | ((src[5] & 0xf0) >> 4);
            dest[i++] = (src[3] << 6) | ((src[6] & 0xfc) >> 2);
            src += 7;   // per 4 output pixels we consume 7 input bytes
        }
        break;

    case 16:
        while (i < n_pixels)
        {
            // reorder big-endian input pixel bits to write 16bit output value
            dest[i++] = (src[0] << 8) | src[1];
            src += 2;   // per output pixel we consume 2 input bytes
        }
        break;

    case 20:
        while ((i + 1) < n_pixels)
        {
            // align 20 pixel bits at LSB of 32bit output value
            dest[i++] = ((src[0] << 8) | (src[1] >> 2) | ((src[4] & 0x03) << 6) | ((src[4] & 0x0c) >> 6));
            dest[i++] = ((src[2] << 8) | (src[3] >> 2) | ((src[4] & 0x30) << 2) | ((src[4] & 0xc0) >> 10));
            src += 5;   // per 3 output pixels we consume 5 input bytes
        }
        break;
    }
    return i;
}


/** Decode CSI-2 encoded pixels to 16bit output pixels. Output pixel values
 *  will be aligned at the most-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_raw16_msb(uint16_t *__restrict dest, uint32_t n_pixels,
                               uint8_t *__restrict src, uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    case 6:
        while (i + 3 < n_pixels)
        {
            // align 6 pixel bits at MSB of 16bit output value
            dest[i++] =  (src[0] & 0x3f) << 10;
            dest[i++] = ((src[1] & 0x0f) << 12) | ((src[0] & 0xc0) << 4);
            dest[i++] = ((src[2] & 0x03) << 14) | ((src[1] & 0xf0) << 6);
            dest[i++] =                            (src[2] & 0xfc) << 8;
            src += 3;   // per 4 output pixels we consume 3 input bytes
        }
        break;

    case 7:
        while (i + 7 < n_pixels)
        {
            // align 7 pixel bits at MSB of 16bit output value
            dest[i++] =  (src[0] & 0x7f) << 9;
            dest[i++] = ((src[1] & 0x3f) << 10) | ((src[0] & 0x80) >> 7);
            dest[i++] = ((src[2] & 0x1f) << 11) | ((src[1] & 0xc0) >> 6);
            dest[i++] = ((src[3] & 0x0f) << 12) | ((src[2] & 0xe0) >> 5);
            dest[i++] = ((src[4] & 0x07) << 13) | ((src[3] & 0xf0) >> 4);
            dest[i++] = ((src[5] & 0x03) << 14) | ((src[4] & 0xf8) >> 3);
            dest[i++] = ((src[6] & 0x01) << 15) | ((src[5] & 0xfc) >> 2);
            dest[i++] =                            (src[6] & 0xfe) >> 1;
            src += 7;   // per 8 output pixels we consume 7 input bytes
        }
        break;

    case 8:
        while (i < n_pixels)
        {
            // align 8 pixel bits at MSB of 16bit output value
            dest[i++] = src[0] << 8;
            src += 1;   // per 1 output pixel we consume 1 input byte
        }
        break;

    case 10:
        while (i + 3 < n_pixels)
        {
            // align 10 pixel bits at MSB of 16bit output value
            dest[i++] = (src[0] << 8) | ((src[4] & 0x03) << 6);
            dest[i++] = (src[1] << 8) | ((src[4] & 0x0c) << 4);
            dest[i++] = (src[2] << 8) | ((src[4] & 0x30) << 2);
            dest[i++] = (src[3] << 8) |  (src[4] & 0xc0);
            src += 5;   // per 4 output pixels we consume 5 input bytes
        }
        break;

    case 12:
        while (i + 1 < n_pixels)
        {
            // align 12 pixel bits at MSB of 16bit output value
            dest[i++] = (src[0] << 8) | ((src[2] & 0x0f) << 4);
            dest[i++] = (src[1] << 8) |  (src[2] & 0xf0);
            src += 3;   // per 2 output pixels we consume 3 input bytes
        }
        break;

    case 14:
        while (i + 3 < n_pixels)
        {
            // align 14 pixel bits at MSB of 16bit output value
            dest[i++] = (src[0] << 8) | ((src[4] & 0x3f) << 2);
            dest[i++] = (src[1] << 8) | ((src[5] & 0x0f) << 4) | ((src[4] & 0xc0) >> 4);
            dest[i++] = (src[2] << 8) | ((src[6] & 0x03) << 6) | ((src[5] & 0xf0) >> 2);
            dest[i++] = (src[3] << 8) |  (src[6] & 0xfc);
            src += 7;   // per 4 output pixels we consume 7 input bytes
        }
        break;

    case 16:
        while (i < n_pixels)
        {
            // reorder big-endian input pixel bits to write 16bit output value
            dest[i++] = (src[0] << 8) | src[1];
            src += 2;   // per output pixel we consume 2 input bytes
        }
        break;
    }

    return i;
}

/** Decode CSI-2 encoded pixels to 32bits output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_raw32(uint32_t *__restrict dest, uint32_t n_pixels,
                           uint8_t *__restrict src,   uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    case 6:
        while (i + 3 < n_pixels)
        {
            // align 6 pixel bits at LSB of 32bit output value
            dest[i++] =   src[0] & 0x3f;
            dest[i++] = ((src[1] & 0x0f) << 2) | ((src[0] & 0xc0) >> 6);
            dest[i++] = ((src[2] & 0x03) << 4) | ((src[1] & 0xf0) >> 4);
            dest[i++] =                           (src[2] & 0xfc) >> 2;
            src += 3;   // per 4 output pixels we consume 3 input bytes
        }
        break;

    case 7:
        while (i + 7 < n_pixels)
        {
            // align 7 pixel bits at LSB of 326bit output value
            dest[i++] =   src[0] & 0x7f;
            dest[i++] = ((src[1] & 0x3f) << 1) | ((src[0] & 0x80) >> 7);
            dest[i++] = ((src[2] & 0x1f) << 2) | ((src[1] & 0xc0) >> 6);
            dest[i++] = ((src[3] & 0x0f) << 3) | ((src[2] & 0xe0) >> 5);
            dest[i++] = ((src[4] & 0x07) << 4) | ((src[3] & 0xf0) >> 4);
            dest[i++] = ((src[5] & 0x03) << 5) | ((src[4] & 0xf8) >> 3);
            dest[i++] = ((src[6] & 0x01) << 6) | ((src[5] & 0xfc) >> 2);
            dest[i++] =                           (src[6] & 0xfe) >> 1;
            src += 7;   // per 8 output pixels we consume 7 input bytes
        }
        break;

    case 8:
        while (i < n_pixels)
        {
            // align 8 pixel bits at LSB of 32bit output value
            dest[i++] = src[0];
            src += 1;   // per 1 output pixel we consume 1 input byte
        }
        break;

    case 10:
        while (i + 3 < n_pixels)
        {
            // align 10 pixel bits at LSB of 32bit output value
            dest[i++] = (src[0] << 2) |  (src[4] & 0x03);
            dest[i++] = (src[1] << 2) | ((src[4] & 0x0c) >> 2);
            dest[i++] = (src[2] << 2) | ((src[4] & 0x30) >> 4);
            dest[i++] = (src[3] << 2) | ((src[4] & 0xc0) >> 6);
            src += 5;   // per 4 output pixels we consume 5 input bytes
        }
        break;

    case 12:
        while (i + 1 < n_pixels)
        {
            // align 12 pixel bits at LSB of 32bit output value
            dest[i++] = (src[0] << 4) |  (src[2] & 0x0f);
            dest[i++] = (src[1] << 4) | ((src[2] & 0xf0) >> 4);
            src += 3;   // per 2 output pixels we consume 3 input bytes
        }
        break;

    case 14:
        while (i + 3 < n_pixels)
        {
            // align 14 pixel bits at LSB of 32bit output value
            dest[i++] = (src[0] << 6) |  (src[4] & 0x3f);
            dest[i++] = (src[1] << 6) | ((src[5] & 0x0f) << 2) | ((src[4] & 0xc0) >> 6);
            dest[i++] = (src[2] << 6) | ((src[6] & 0x03) << 4) | ((src[5] & 0xf0) >> 4);
            dest[i++] = (src[3] << 6) | ((src[6] & 0xfc) >> 2);
            src += 7;   // per 4 output pixels we consume 7 input bytes
        }
        break;

    case 16:
        while (i < n_pixels)
        {
            // reorder big-endian input pixel bits to write 32bit output value
            dest[i++] = (src[0] << 8) | src[1];
            src += 2;   // per output pixel we consume 2 input bytes
        }
        break;

    case 20:
        while ((i + 1) < n_pixels)
        {
            // align 20 pixel bits at LSB of 32bit output value
            dest[i++] = ((src[0] << 12) | (src[1] << 2) | ((src[4] & 0x03) << 10) | ((src[4] & 0x0c) >> 2));
            dest[i++] = ((src[2] << 12) | (src[3] << 2) | ((src[4] & 0x30) << 6)  | ((src[4] & 0xc0) >> 6));
            src += 5;   // per 3 output pixels we consume 5 input bytes
        }
        break;

    case 24:
        while (i < n_pixels)
        {
            // align 24 pixel bits at LSB of 32bit output value
            dest[i++] = (src[0] << 16) | (src[1] << 8) | src[2];
            src += 3;   // per output pixel we consume 3 input bytes
        }
        break;

    }

    return i;
}

/** Decode CSI-2 encoded pixels to 32bits output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_raw32_msb(uint32_t *__restrict dest, uint32_t n_pixels,
                               uint8_t *__restrict src,   uint32_t data_bits)
{
    uint32_t    i = 0;

    switch (data_bits)
    {
    case 6:
        while (i + 3 < n_pixels)
        {
            // align 6 pixel bits at MSB of 32bit output value
            dest[i++] = ((src[0] & 0x3f) << 26);
            dest[i++] = ((src[1] & 0x0f) << 28) | ((src[0] & 0xc0) << 6);
            dest[i++] = ((src[2] & 0x03) << 30) | ((src[1] & 0xf0) << 4);
            dest[i++] =                           (src[2] & 0xfc)  << 2;
            src += 3;   // per 4 output pixels we consume 3 input bytes
        }
        break;

    case 7:
        while (i + 7 < n_pixels)
        {
            // align 7 pixel bits at MSB of 326bit output value
            dest[i++] = ((src[0] & 0x7f) << 25);
            dest[i++] = ((src[1] & 0x3f) << 26) | ((src[0] & 0x80) >> 7);
            dest[i++] = ((src[2] & 0x1f) << 27) | ((src[1] & 0xc0) >> 6);
            dest[i++] = ((src[3] & 0x0f) << 28) | ((src[2] & 0xe0) >> 5);
            dest[i++] = ((src[4] & 0x07) << 29) | ((src[3] & 0xf0) >> 4);
            dest[i++] = ((src[5] & 0x03) << 30) | ((src[4] & 0xf8) >> 3);
            dest[i++] = ((src[6] & 0x01) << 31) | ((src[5] & 0xfc) >> 2);
            dest[i++] =                           ((src[6] & 0xfe) >> 1);
            src += 7;   // per 8 output pixels we consume 7 input bytes
        }
        break;

    case 8:
        while (i < n_pixels)
        {
            // align 8 pixel bits at MSB of 32bit output value
            dest[i++] = (src[0] << 24);
            src += 1;   // per 1 output pixel we consume 1 input byte
        }
        break;

    case 10:
        while (i + 3 < n_pixels)
        {
            // align 10 pixel bits at MSB of 32bit output value
            dest[i++] = (src[0] << 24) | ((src[4] & 0x03) << 24);
            dest[i++] = (src[1] << 24) | ((src[4] & 0x0c) >> 20);
            dest[i++] = (src[2] << 24) | ((src[4] & 0x30) >> 18);
            dest[i++] = (src[3] << 24) |  (src[4] & 0xc0);
            src += 5;   // per 4 output pixels we consume 5 input bytes
        }
        break;

    case 12:
        while (i + 1 < n_pixels)
        {
            // align 12 pixel bits at MSB of 32bit output value
            dest[i++] = (src[0] << 24) | ((src[2] & 0x0f) << 20);
            dest[i++] = (src[1] << 24) |  ( src[2] & 0xf0);
            src += 3;   // per 2 output pixels we consume 3 input bytes
        }
        break;

    case 14:
        while (i + 3 < n_pixels)
        {
            // align 14 pixel bits at MSB of 32bit output value
            dest[i++] = (src[0] << 24) | ((src[4] & 0x3f) << 18);
            dest[i++] = (src[1] << 24) | ((src[5] & 0x0f) << 20) | ((src[4] & 0xc0) >> 24);
            dest[i++] = (src[2] << 24) | ((src[6] & 0x03) << 24) | ((src[5] & 0xf0) >> 22);
            dest[i++] = (src[3] << 24) | (src[6] & 0xfc);
            src += 7;   // per 4 output pixels we consume 7 input bytes
        }
        break;

    case 16:
        while (i < n_pixels)
        {
            // reorder big-endian input pixel bits to write 32bit output value
            dest[i++] = (src[0] << 24) | (src[1] << 16U);
            src += 2;   // per output pixel we consume 2 input bytes
        }
        break;

    case 20:
        while ((i + 1) < n_pixels)
        {
            // align 20 pixel bits at MSB of 32bit output value
            dest[i++] = ((src[0] << 24) | (src[1] << 14) | ((src[4] & 0x03) << 22) | ((src[4] & 0x0c) >> 14));
            dest[i++] = ((src[2] << 24) | (src[3] << 14) | ((src[4] & 0x30) << 18) | ((src[4] & 0xc0) >> 18));
            src += 5;   // per 3 output pixels we consume 5 input bytes
        }
        break;

    case 24:
        while (i < n_pixels)
        {
            // align 24 pixel bits at MSB of 32bit output value
            dest[i++] = (src[0] << 24) | (src[1] << 16) | (src[2] << 8);
            src += 3;   // per output pixel we consume 3 input bytes
        }
        break;
    }


    return i;
}


/** Decode CSI-2 encoded pixels to 24bit RGB output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_24(uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_uyvy8_24_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 24bit RGB output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_24_rgb(uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_uyvy8_24_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 24bit BGR output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_24_bgr(uint8_t* __restrict dest, uint32_t n_pixels,
                                  uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_uyvy8_24_generic(dest, n_pixels, src, data_bits, 1);
}

/** Decode CSI-2 encoded pixels to 24bit RGB or BGR output pixels.
 *
 * Output pixel values will be aligned at the least-significant bit. This
 * function can only be used to decode complete pixel groups. The size of a
 * pixel group is implicit in the number of bits occupied per pixel (@see
 * csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @param output_format BGR or RGB.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_24_generic(uint8_t* __restrict dest, uint32_t n_pixels,
                                      uint8_t* __restrict src, uint32_t data_bits,
                                      int output_format)
{
    return csi2_decode_uyvy8_24_generic_ccm(dest, n_pixels, src, data_bits,
                                            output_format,
                                            CCM_LIMITED_RANGE_BT_601);
}


/** Decode CSI-2 encoded pixels to 24bit RGB or BGR output pixels.
 *
 * Output pixel values will be aligned at the least-significant bit. This
 * function can only be used to decode complete pixel groups. The size of a
 * pixel group is implicit in the number of bits occupied per pixel (@see
 * csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @param output_format BGR or RGB.
 *
 * @param ccm       Color conversion matrix. The following conversion matrices
 *                  are pre-defined:
 *                  - CCM_LIMITED_RANGE_BT_601: Limited Range BT.601
 *                  - CCM_LIMITED_RANGE_BT_709: Limited Range BT.709
 *                  - CCM_FULL_RANGE_BT_601: Full Range BT.601
 *                  - CCM_FULL_RANGE_BT_709: Full Range BT.709
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_24_generic_ccm(
    uint8_t *__restrict dest, uint32_t n_pixels,
    uint8_t *__restrict src, uint32_t data_bits,
    int output_format, const float ccm[12])
{
    uint32_t    pg_bytes; // number of source bytes to consume per pixel group
    int         dst_offset_0;
    int         dst_offset_1;
    int         dst_offset_2;
    int         dst_offset_3;
    int         dst_offset_4;
    int         dst_offset_5;

    if (data_bits == 16)
    {
        // YUV422 with 8bit per color component (i.e., 16 bit per pixel)
        pg_bytes = 4;
    }
    else if (data_bits == 20)
    {
        // YUV422 with 10bit per color component (20 bit per pixel)
        pg_bytes = 5;
    }
    else
        return 0;   // unsupported conversion

    switch (output_format)
    {
    case 1: // BGR
        dst_offset_0 = 2;
        dst_offset_1 = 1;
        dst_offset_2 = 0;
        dst_offset_3 = 5;
        dst_offset_4 = 4;
        dst_offset_5 = 3;
        break;
    case 0:
    default: // RGB
        dst_offset_0 = 0;
        dst_offset_1 = 1;
        dst_offset_2 = 2;
        dst_offset_3 = 3;
        dst_offset_4 = 4;
        dst_offset_5 = 5;
    }

    for (uint32_t i = 0; i + 1 < n_pixels; i += 2)
    {
        auto u  = src[0] + ccm[7] ;    // Cb: -128
        auto y0 = src[1] + ccm[3] ;    //     -16
        auto v  = src[2] + ccm[11] ;   // Cr: -128
        auto y1 = src[3] + ccm[3] ;    //     -16
        // YUV422-10: ignore LSBs in src[4] since we are decoding to RGB24
        src += pg_bytes;    // no. of bytes to consume per 2 output pixels

        // decode two RGB values without chrominance interpolation
        auto r0 = ccm[0] * y0 + ccm[1] * u + ccm[2]  * v;
        auto g0 = ccm[4] * y0 + ccm[5] * u + ccm[6]  * v;
        auto b0 = ccm[8] * y0 + ccm[9] * u + ccm[10] * v;

        auto r1 = ccm[0] * y1 + ccm[1] * u + ccm[2]  * v;
        auto g1 = ccm[4] * y1 + ccm[5] * u + ccm[6]  * v;
        auto b1 = ccm[8] * y1 + ccm[9] * u + ccm[10] * v;

        // write two RBG pixels
        dest[3 * i + dst_offset_0] = r0 > 255. ? 255 : (r0 < 0. ? 0 : (uint8_t)r0);
        dest[3 * i + dst_offset_1] = g0 > 255. ? 255 : (g0 < 0. ? 0 : (uint8_t)g0);
        dest[3 * i + dst_offset_2] = b0 > 255. ? 255 : (b0 < 0. ? 0 : (uint8_t)b0);

        dest[3 * i + dst_offset_3] = r1 > 255. ? 255 : (r1 < 0. ? 0 : (uint8_t)r1);
        dest[3 * i + dst_offset_4] = g1 > 255. ? 255 : (g1 < 0. ? 0 : (uint8_t)g1);
        dest[3 * i + dst_offset_5] = b1 > 255. ? 255 : (b1 < 0. ? 0 : (uint8_t)b1);
    }

    return n_pixels;
}


/** Decode CSI-2 encoded pixels to 32bit RGBA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_32(
    uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_uyvy8_32_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 32bit RGBA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_32_rgba(
    uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_uyvy8_32_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 32bit BGRA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_32_bgra(
    uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_uyvy8_32_generic(dest, n_pixels, src, data_bits, 1);
}


/** Decode CSI-2 encoded pixels to 32bit BGR output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @param output_format BGR or RGB.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_32_generic(uint8_t* __restrict dest, uint32_t n_pixels,
                                      uint8_t* __restrict src, uint32_t data_bits,
                                      int output_format)
{
    return csi2_decode_uyvy8_32_generic_ccm(dest, n_pixels, src, data_bits,
                                            output_format,
                                            CCM_LIMITED_RANGE_BT_601);
}


/** Decode CSI-2 encoded pixels to 32bit RGBA or BGRA output pixels.
 *
 * Output pixel values will be aligned at the least-significant bit. This
 * function can only be used to decode complete pixel groups. The size of a
 * pixel group is implicit in the number of bits occupied per pixel (@see
 * csi2_decode_datatype).  The alpha component of the decoded pixels will be
 * set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @param output_format BGR or RGB.
 *
 * @param ccm       Color conversion matrix. The following conversion matrices
 *                  are pre-defined:
 *                  - CCM_LIMITED_RANGE_BT_601: Limited Range BT.601
 *                  - CCM_LIMITED_RANGE_BT_709: Limited Range BT.709
 *                  - CCM_FULL_RANGE_BT_601: Full Range BT.601
 *                  - CCM_FULL_RANGE_BT_709: Full Range BT.709
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_uyvy8_32_generic_ccm(
    uint8_t *__restrict dest, uint32_t n_pixels,
    uint8_t *__restrict src, uint32_t data_bits,
    int output_format, const float ccm[12])
{
    uint32_t    pg_bytes; // number of source bytes to consume per pixel group
    int         dst_offset_0;
    int         dst_offset_1;
    int         dst_offset_2;
    int         dst_offset_3;
    int         dst_offset_4;
    int         dst_offset_5;
    int         dst_offset_6;
    int         dst_offset_7;

    if (data_bits == 16)
    {
        // YUV422 with 8bit per color component (i.e., 16 bit per pixel)
        pg_bytes = 4;
    }
    else if (data_bits == 20)
    {
        // YUV422 with 10bit per color component (20 bit per pixel)
        pg_bytes = 5;
    }
    else
        return 0;   // unsupported conversion

    switch (output_format)
    {
    case 1: // BGRA
        dst_offset_0 = 2;
        dst_offset_1 = 1;
        dst_offset_2 = 0;
        dst_offset_3 = 3;
        dst_offset_4 = 6;
        dst_offset_5 = 5;
        dst_offset_6 = 4;
        dst_offset_7 = 7;
        break;
    case 0:
    default: // RGBA
        dst_offset_0 = 0;
        dst_offset_1 = 1;
        dst_offset_2 = 2;
        dst_offset_3 = 3;
        dst_offset_4 = 4;
        dst_offset_5 = 5;
        dst_offset_6 = 6;
        dst_offset_7 = 7;
    }

    for (uint32_t i = 0; i + 1 < n_pixels; i += 2)
    {
        auto u  = src[0] + ccm[7] ;    // Cb: -128
        auto y0 = src[1] + ccm[3] ;    //     -16
        auto v  = src[2] + ccm[11] ;   // Cr: -128
        auto y1 = src[3] + ccm[3] ;    //     -16

        // YUV422-10: ignore LSBs in src[4] since we are decoding to RGB24
        src += pg_bytes;    // no. of bytes to consume per 2 output pixels

        // decode two RGB values without chrominance interpolation
        auto r0 = ccm[0] * y0 + ccm[1] * u + ccm[2]  * v;
        auto g0 = ccm[4] * y0 + ccm[5] * u + ccm[6]  * v;
        auto b0 = ccm[8] * y0 + ccm[9] * u + ccm[10] * v;

        auto r1 = ccm[0] * y1 + ccm[1] * u + ccm[2]  * v;
        auto g1 = ccm[4] * y1 + ccm[5] * u + ccm[6]  * v;
        auto b1 = ccm[8] * y1 + ccm[9] * u + ccm[10] * v;

        // write two RBGA pixels
        dest[4 * i + dst_offset_0] = r0 > 255. ? 255 : (r0 < 0. ? 0 : (uint8_t)r0);
        dest[4 * i + dst_offset_1] = g0 > 255. ? 255 : (g0 < 0. ? 0 : (uint8_t)g0);
        dest[4 * i + dst_offset_2] = b0 > 255. ? 255 : (b0 < 0. ? 0 : (uint8_t)b0);
        dest[4 * i + dst_offset_3] = 255;

        dest[4 * i + dst_offset_4] = r1 > 255. ? 255 : (r1 < 0. ? 0 : (uint8_t)r1);
        dest[4 * i + dst_offset_5] = g1 > 255. ? 255 : (g1 < 0. ? 0 : (uint8_t)g1);
        dest[4 * i + dst_offset_6] = b1 > 255. ? 255 : (b1 < 0. ? 0 : (uint8_t)b1);
        dest[4 * i + dst_offset_7] = 255;
    }

    return n_pixels;
}


/** Decode CSI-2 encoded pixels to 24bit RGB output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_24(uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_rgb8_24_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 24bit RGB output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_24_rgb(uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_rgb8_24_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 24bit BGR output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_24_bgr(uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_rgb8_24_generic(dest, n_pixels, src, data_bits, 1);
}


/** Decode CSI-2 encoded pixels to 24bit RGB or BGR output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *
 * @param dest          Output data pointer.
 * @param n_pixels      Number of pixels to decode. Only a number of pixels that is
 *                      the largest integer multiple of the pixel group size and is
 *                      less than or equal to n_pixels will be decoded.
 * @param src           CSI-2 encoded input data pointer. Must point to the first
 *                      byte of a pixel group.
 * @param data_bits     Number of data bits per pixel.
 * @param output_format BGR or RGB.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_24_generic(uint8_t *__restrict dest, uint32_t n_pixels,
                             uint8_t *__restrict src, uint32_t data_bits,
                             int output_format)
{
    uint32_t    pg_bytes; // number of source bytes to consume per pixel group
    int         dst_offset_0;
    int         dst_offset_1;
    int         dst_offset_2;

    if (data_bits == 24)
    {
        // RGB with 8bit per color component (i.e., 24 bit per pixel)
        pg_bytes = 3;
    }
    else
        return 0;   // unsupported conversion

    switch (output_format)
    {
    case 1: // BGR
        dst_offset_0 = 2;
        dst_offset_1 = 1;
        dst_offset_2 = 0;
        break;
    case 0:
    default: // RGB
        dst_offset_0 = 0;
        dst_offset_1 = 1;
        dst_offset_2 = 2;
    }

    for (uint32_t i = 0; i < n_pixels; i++)
    {
        // write an RGBA/BGRA pixel (note: source is in BGR byte order)
        dest[3 * i + dst_offset_0] = src[2];   // R
        dest[3 * i + dst_offset_1] = src[1];   // G
        dest[3 * i + dst_offset_2] = src[0];   // B

        src += pg_bytes;
    }

    return n_pixels;
}


/** Decode CSI-2 encoded pixels to 32bit RGBA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_32(
    uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_rgb8_32_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 32bit RGBA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_32_rgba(
    uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_rgb8_32_generic(dest, n_pixels, src, data_bits, 0);
}


/** Decode CSI-2 encoded pixels to 32bit BGRA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest      Output data pointer.
 * @param n_pixels  Number of pixels to decode. Only a number of pixels that is
 *                  the largest integer multiple of the pixel group size and is
 *                  less than or equal to n_pixels will be decoded.
 * @param src       CSI-2 encoded input data pointer. Must point to the first
 *                  byte of a pixel group.
 * @param data_bits Number of data bits per pixel.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_32_bgra(
    uint8_t* __restrict dest, uint32_t n_pixels,
    uint8_t* __restrict src, uint32_t data_bits)
{
    return csi2_decode_rgb8_32_generic(dest, n_pixels, src, data_bits, 1);
}


/** Decode CSI-2 encoded pixels to 32bit RGBA or BGRA output pixels. Output pixel values
 *  will be aligned at the least-significant bit. This function can only be used
 *  to decode complete pixel groups. The size of a pixel group is implicit in
 *  the number of bits occupied per pixel (@see csi2_decode_datatype).
 *  The alpha component of the decoded pixels will be set to 255.
 *
 * @param dest          Output data pointer.
 * @param n_pixels      Number of pixels to decode. Only a number of pixels that is
 *                      the largest integer multiple of the pixel group size and is
 *                      less than or equal to n_pixels will be decoded.
 * @param src           CSI-2 encoded input data pointer. Must point to the first
 *                      byte of a pixel group.
 * @param data_bits     Number of data bits per pixel.
 * @param output_format BGR or RGB.
 *
 * @return  Number of decoded pixels
 */
uint32_t csi2_decode_rgb8_32_generic(
    uint8_t *__restrict dest, uint32_t n_pixels,
    uint8_t *__restrict src, uint32_t data_bits,
    int output_format)
{
    uint32_t    pg_bytes; // number of source bytes to consume per pixel group
    int         dst_offset_0;
    int         dst_offset_1;
    int         dst_offset_2;
    int         dst_offset_3;

    switch (output_format)
    {
    case 1: // BGR
        dst_offset_0 = 2;
        dst_offset_1 = 1;
        dst_offset_2 = 0;
        dst_offset_3 = 3;
        break;
    case 0:
    default: // RGB
        dst_offset_0 = 0;
        dst_offset_1 = 1;
        dst_offset_2 = 2;
        dst_offset_2 = 3;
    }

    if (data_bits == 24)
    {
        // RGB with 8bit per color component (i.e., 24 bit per pixel)
        pg_bytes = 3;
    }
    else
        return 0;   // unsupported conversion

    for (uint32_t i = 0; i < n_pixels; i++)
    {
        // write an RGBA/BGRA pixel (note: source is in BGR byte order)
        dest[4 * i + dst_offset_0] = src[2];   // R
        dest[4 * i + dst_offset_1] = src[1];   // G
        dest[4 * i + dst_offset_2] = src[0];   // B
        dest[4 * i + dst_offset_3] = 255;

        src += pg_bytes;
    }

    return n_pixels;
}
