#define _XOPEN_SOURCE 600   /* struct timespec, fd_set */
#include <SDL.h>
#include "sxpf.h"
#include "helpers.h"
#include "imx290.h"
#include "csi-2.h"
#include "gl_fbo.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <libgen.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <set>
#include <thread>
#include <vector>
#include <algorithm>

#include "update_texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NELEMENTS(ar)   (sizeof(ar) / sizeof((ar)[0]))

// adapted from:
//  http://stackoverflow.com/questions/8583308/timespec-equivalent-for-windows
#ifdef WIN32
#include <windows.h>
double getElapsedTime(void)
{
    static LARGE_INTEGER freq, start;
    LARGE_INTEGER count;

    QueryPerformanceCounter(&count);

    if (!freq.QuadPart)
    {
        QueryPerformanceFrequency(&freq);
        start = count;
    }
    return (double)(count.QuadPart - start.QuadPart) / freq.QuadPart;
}

size_t getExePath(char *pathname, size_t bufsize)
{
    char    *exename;
    size_t   path_len;

    if (_get_pgmptr(&exename))
        return 0;

    // find last path separator
    char *last_sep = strrchr(exename, '\\');

    if (last_sep == NULL)
        return 0;

    path_len = last_sep - exename;

    if (path_len == 0)
        return 0;

    if (path_len >= bufsize)
        return 0;

    memcpy(pathname, exename, path_len + 1);

    return path_len + 1;
}
#else
#include <sys/time.h>           /* for gettimeofday() */
#include <unistd.h>             /* for readlink() */
double getElapsedTime(void)
{
    static struct timeval t0;
    struct timeval tv;
    gettimeofday(&tv, 0);
    if (!t0.tv_sec)
        t0 = tv;
    return (double)tv.tv_sec - t0.tv_sec
        + ((double)tv.tv_usec - t0.tv_usec) / 1000000.;
}

size_t getExePath(char *pathname, size_t bufsize)
{
    char    exename[1024];

    ssize_t len = readlink("/proc/self/exe", exename, sizeof(exename) - 1);

    if (len < 0 || len == sizeof(exename) - 1)
        return 0;   // error getting executable's name or path too long

    exename[len] = 0; // add EOS, as readlink doesn't zero-terminate the result

    // find last path separator
    char *last_sep = strrchr(exename, '/');

    if (last_sep == NULL)
        return 0;

    size_t path_len = last_sep - exename;

    if (path_len == 0)
        return 0;

    if (path_len >= bufsize)
        return 0;

    memcpy(pathname, exename, path_len + 1);

    return path_len + 1;
}
#endif

typedef struct frame_buf_timing_s
{
    int   frame_buffer = -1;
    __s64 receive_time = -1;
    __s64 release_time = -1;
} frame_buf_timing_t;

/** Input channel state */
typedef struct input_channel_s
{
    sxpf_hdl        fg = 0;
    HWAITSXPF       devfd = 0;
    int             endpoint_id = 0;
    double          last_rxtime = 0;
    uint32_t        frame_info = 0;
} input_channel_t;


typedef struct opts_s
{
    int         endpoint_id;
    int         channel_id = 0;

    /** select field from interleavced input: 0=even, 1=odd, 2=all lines */
    uint32_t    field_sel = 0;          // select LEF field by default

    /** abort after x images */
    int32_t     abort_after = -1;       // run continuously by default

    /** abort after x seconds if no image is received */
    int32_t     abort_after_sec = -1;   // run continuoudly by default

    /** timeout after x seconds */
    int32_t     timeout_after_sec = -1; // run continuoudly by default

    /** frame buffer debugging option */
    int         buffer_debug = 0;

    /** reference image for sxpfapp debugging */
    const char  *dbg_image = "";

    /** reference image for loopback testing */
    const char  *ref_image = "";

    /** filename of saved error images */
    const char  *error_image_name = "err_%010d.raw";

    /** filename of saved images */
    const char  *image_name = "img_%010d.raw";

    /** write video to disk */
    uint32_t    write_video = 0;

    /** write x images to disk */
    int32_t     write_images = 0;

    /** write x error images to disk */
    int32_t     write_error_images = 0;

    /** use newline instead of \r */
    int         newline_mode = 0;
} opts_t;


static opts_t   g_opts;

enum Opt
{
    // ensure long-options-only parameters have a code unequal to any character
    _ = 256,
    optCard,
    optChannel,
    optDrop,
    optNewlineMode,
    optShowTimestamp,
    optBorderless,
    optBackground,
    optSaturation,
    optGamma,
    optVFlip,
    optHFlip,
    optZoom,
    optOutputFormat,
    optDropColumn,
    optYuvInRaw16,
    optBufferDebug,
};

static struct option long_options[] =
{
    { "help",            no_argument,       0,  'h' },
    { "card",            required_argument, 0,  optCard },
    { "channel",         required_argument, 0,  optChannel },
    { "drop",            required_argument, 0,  optDrop },
    { "newline",         no_argument,       0,  optNewlineMode },
    { "show-ts",         no_argument,       0,  optShowTimestamp },
    { "borderless",      no_argument,       0,  optBorderless },
    { "bg",              required_argument, 0,  optBackground },
    { "saturation",      required_argument, 0,  optSaturation },
    { "gamma",           required_argument, 0,  optGamma },
    { "vflip",           no_argument,       0,  optVFlip },
    { "hflip",           no_argument,       0,  optHFlip },
    { "zoom",            required_argument, 0,  optZoom },
    { "output-format",   required_argument, 0,  optOutputFormat },
    { "drop-column",     required_argument, 0,  optDropColumn },
    { "yuv-in-raw16",    no_argument,       0,  optYuvInRaw16 },
    { "buffer-debug",    no_argument,       0,  optBufferDebug },
    { 0, 0, 0, 0 }
};

input_channel_t channels[4];

// container for data exchange to sdl specific code
sdl_ctrl_t          sdl_ctrl;

// reference for image for loopback testing
uint16_t ref_img[4096*2160];

time_t   timeOld = 0;
time_t   timeStart = 0;

FILE     *rawVideo;

sxpf_card_props_t props;

// declare extern (from getopt.h), to suppress 'invalid option' output
extern int opterr;
// declare extern , to reset option index used by getopt_long
extern int optind;

void usage(char *pname);
void options(int argc, char **argv, opts_t *opts);

static void quit(int rc);

void init_channel(input_channel_t *ch, int endpoint_id, int channel_id);
void close_channel(input_channel_t *ch);

void sigint_handler(int signo)
{
    (void)signo;
    sdl_ctrl.sdl_done = SDL_TRUE;
}

void release_buffer(sxpf_hdl fg, int slot)
{
#if 0 /* activated only for debugging low-level image transmission */
    // clear buffer before the grabber hardware writes new data to it
    sxpf_card_props_t   props;

    if (!sxpf_get_card_properties(fg, &props))
    {
        void   *pbuf = sxpf_get_frame_ptr(fg, slot);

        if (pbuf)
            memset(pbuf, 0, props.buffer_size);
    }
#endif

    sxpf_release_frame(fg, slot, 0);
}


void close_channel(input_channel_t *ch)
{
    if (ch->fg)
    {
        sxpf_close(ch->fg);
    }
}


void quit(int rc)
{
    SDL_Quit();

    close_channel(&channels[0]);

    fprintf(stderr, "\nDone.\n");

    exit(rc);
}


void init_channel(input_channel_t *ch, int endpoint_id, int channel_id)
{
    *ch = input_channel_t{};

    printf("edpoint : %d, channel : %d", endpoint_id, channel_id);

    ch->fg = sxpf_open(endpoint_id);
    if (ch->fg)
    {
        ch->endpoint_id = endpoint_id;

        if (sxpf_start_record(ch->fg, SXPF_STREAM_VIDEO0 << (channel_id & 0xf)))
        {
            fprintf(stderr, "failed to start stream\n");
        }

        sxpf_get_device_fd(ch->fg, &ch->devfd);

#ifndef WIN32
        //fcntl(ch->devfd, F_SETFL, 0); // reset O_NONBLOCK flag
        fcntl(ch->devfd, F_SETFL, O_NONBLOCK); // set O_NONBLOCK flag
#endif
    }
    else if (strlen(g_opts.dbg_image))
    {
        // nop
    }
    else
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Couldn't initialize Frame Grabber.\n");
        exit(1);
    }
}


sxpf_image_header_t* load_dbg_image(const char *dbg_image)
{
    static int dbg_image_idx = 0;
    static int dbg_image_cnt = -1;

    FILE *file;
    char *buffer;
    char *load_image;
    int fileLen;
    int ret;
    char *ptr;
    char  delim[] = ",";
    char  filename[10240];
    int   i;

    strcpy(filename, dbg_image);
    load_image = filename;
    if (strpbrk(filename, delim) != NULL)
    {
        ptr = strtok(filename, delim);
        i   = 0;
        while (ptr != NULL)
        {
            if (i == dbg_image_idx)
            {
                break;
            }

            ptr = strtok(NULL, delim);
            i++;

            if (ptr == NULL)
            {
                dbg_image_cnt = i;
                dbg_image_idx = 0;
                i             = 0;
                ptr           = strtok(filename, delim);
            }
        }

        dbg_image_idx++;
        if (dbg_image_cnt != -1)
        {
            dbg_image_idx %= dbg_image_cnt;
        }
        load_image                  = ptr;
    }

    file = fopen(load_image, "rb");
    if (!file)
    {
        fprintf(stderr, "Unable to open file\n");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    fileLen=ftell(file);
    fseek(file, 0, SEEK_SET);

    buffer=(char *)malloc(fileLen+1);
    if (!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(file);
        return NULL;
    }

    ret = (int)fread(buffer, 1, fileLen, file);
    if (ret != fileLen)
    {
        printf("Read failed %d %d\n", ret, fileLen);
        free(buffer);
        return NULL;
    }
    fclose(file);

    return (sxpf_image_header_t*)buffer;
}


void usage(char *pname)
{
    printf("The sxpfapp sample application allows capturing image frames from "
           "the grabber card.\n"
           "The commandline output shows the received frame size and frame "
           "rate.\n"
           "\n"
           "Usage: %s [options]\n"
           "\n"
           "Options:\n"
           "\t-a num            Abort after number of images\n"
           "\t-A num            Abort after number of seconds if no image is received\n"
           "\t-c image          Compare received images against reference image\n"
           "\t-D image          Input image for debugging sxpfapp image pipeline\n"
           "\t-e name           File name of saved error images (default: err_%s010d.raw)\n"
           "\t-h/--help         Show this help screen\n"
           "\t-n name           File name of saved images (default: img_%s010d.raw)\n"
           "\t-q                Do not show SDL window\n"
           "\t-T num            Abort after number of seconds\n"
           "\t-v                Store video into binary video.raw file\n"
           "\t-w num            Write number of images to disk (-1 = save all)\n"
           "\t-W num            Write number of error images to disk (-1 = save all)\n"
           "\t--card            Select card to capture (0...cards-1)\n"
           "\t--channel         Select channel to capture (0...7)\n"
           "\t--buffer-debug    Enable debug output of frame buffer timing\n"
           "\t--newline         Output \\n instead of \\r when printing frame informations\n",
           basename(pname), "%", "%");

    // print sdl specific usage information
    SdlUsage();

    printf("\n"
           "Deprecated options:\n"
           "\t-C10|-C12|-C14|-C16\n"
           "\t                  Assume CSI-2 format with 10, 12, 14 or 16 bits per "
           "component\n"
           "\t-d dt             CSI-2 datatype/virtual channel to parse RAW CSI-2 data format\n"
           "\t                  Bit 7:6 = virtual channel\n"
           "\t                  Bit 5:0 = datatype\n"
           "\t                  Do not use together with -C parameter\n"
           "\t-f                Start fullscreen\n"
           "\t-F num            Write number of filtered CSI-2 images to disk (-1 = save all)\n"
           "\t                  Filtering is based on -d parameter\n"
           "\t-g WxH[@XxY]      Set window geometry, e.g. -g 1280x720\n"
           "\t                  Optionally add window position, e.g. -g 1280x720@0x0\n"
           "\t-G WxH            Force image geometry and disregard geometry of received data\n"
           "\t-i                Input is 12bpp packed raw DOL2 from SONY IMX290\n"
           "\t-l num            Shift input pixels left by this amount (default: 0)\n"
           "\t-m file           Mix incoming data bits based on the given mapping file\n"
           "\t-o                Input is RGB24 in 32 bit per pixel OpenLDI-scrambled\n"
           "\t-O num            Write number of rendered OpenGL output images to disk (-1 = save all)\n"
           "\t                  use -g parameter to set image size\n"
           "\t-p 0..3           Select 1 out of 4 possible Bayer patterns\n"
           "\t-r                RAW input: de-bayer received images\n"
           "\t-R24|-R32         Select RGB input (24bit RGB or 32bit RGBA)\n"
           "\t-s shader         Override shader file(s): filename without extension\n"
           "\t                  Built-in shaders: %s\n"
           "\t-t                Enable pixel test\n"
           "\t-V                Verify CSI-2 data (check for CRC errors)\n"
           "\t-8                Input is compacted to 8 bit per component\n"
           "\t--drop x          x >=  2: Drop every x frame\n"
           "\t                  x <= -2: Show every x frame\n"
           "\t                  In this mode the displayed frame rate will be a calculated mean value\n"
           "\t--drop-column x   Only use every x column for the CSI-2 decoded image output\n"
           "\t--show-ts         Show timestamp informations\n"
           "\t--borderless      Show SDL window in borderless mode\n"
           "\t--bg 0xrrggbb     Set background color for SDL window (default: 0x4c4cff)\n"
           "\t--gamma x         Set gamma value (0.5-2.0, default: 1.2)\n"
           "\t--saturation x    Set saturation value (0.0-2.0, default: 1.0)\n"
           "\t--vflip           Flip image vertically\n"
           "\t--hflip           Flip image horizontally\n"
           "\t--yuv-in-raw16    YUV data is captured in RAW16 datatype e.g. using a camAD3 DUAL MAX96705/96706\n"
           "\t--zoom x          Set zoom value (1.0-5.0, default: 1.0)\n"
           "\t--output-format x Set format of rendered output images when using -O parameter to\n"
           "\t                  jpg, png or bmp (default: jpg:90)\n"
           "\t                  Optionally set\n"
           "\t                  jpg quality (0...100) using parameter 'jpg:quality'\n"
           "\t                  png compression level (0..10) using parameter 'png:compression'\n",
           list_shaders().c_str());
}

void options(int argc, char **argv, opts_t *opts)
{
    int     c;
    int     v, w, h, x, y;
    char    s[4];
    int     option_index;
    int     curr_index = 0;

    sdlopt_long_option_grade_e sdl_long_opt_grade;

    // suppress 'invalid option' output when calling getopt_long(..)
    opterr = 0;

    while (1)
    {
        option_index = 0;
        curr_index = optind;

        c = getopt_long(argc, argv, "a:A:bc:C:d:D:e:fF:g:G:hil:m:n:oO:p:qrR:s:tT:w:W:vV8", long_options, &option_index);

        if (c == -1)
            break;

        switch (c)
        {
        case optCard:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
                g_opts.endpoint_id = atoi(optarg);
            break;

        case optChannel:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
                g_opts.channel_id = atoi(optarg);
            break;

        case optDrop:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_ctrl.sdl_opt.drop_show_count = atoi(optarg);
            }
            break;

        case optDropColumn:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_ctrl.sdl_opt.drop_column = atoi(optarg);
            }
            break;

        case optNewlineMode:
            g_opts.newline_mode = 1;
            break;

        case optShowTimestamp:
            sdl_ctrl.sdl_opt.show_timestamp = 1;
            break;

        case optBorderless:
            sdl_ctrl.sdl_opt.borderless = 1;
            break;

        case optBackground:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
                sdl_ctrl.sdl_opt.background = strtol(optarg, NULL, 16);
            break;

        case optGamma:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_ctrl.sdl_opt.gl_gamma = strtof(optarg, nullptr);
                if (sdl_ctrl.sdl_opt.gl_gamma < 0.5)
                    sdl_ctrl.sdl_opt.gl_gamma = 0.5;
                if (sdl_ctrl.sdl_opt.gl_gamma > 2)
                    sdl_ctrl.sdl_opt.gl_gamma = 2;
            }
            break;

        case optVFlip:
            sdl_ctrl.sdl_opt.vflip = 1;
            break;

        case optHFlip:
            sdl_ctrl.sdl_opt.hflip = 1;
            break;

        case optZoom:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_ctrl.sdl_opt.gl_zoom = strtof(optarg, nullptr);
                if (sdl_ctrl.sdl_opt.gl_zoom < 1.0)
                    sdl_ctrl.sdl_opt.gl_zoom = 1.0;
                if (sdl_ctrl.sdl_opt.gl_zoom > 6.0)
                    sdl_ctrl.sdl_opt.gl_zoom = 6.0;
            }

            break;

        case optOutputFormat:
            c = sscanf(optarg, "%3s:%d", s, &v);
            if (strstr(s, "jpg") != NULL)
            {
                sdl_ctrl.sdl_opt.write_output_format = jpg;
                if (c > 1)
                {
                    sdl_ctrl.sdl_opt.write_output_jpg_quality = v;
                }
            }
            else if (strstr(s, "png") != NULL)
            {
                sdl_ctrl.sdl_opt.write_output_format = png;
                if (c > 1)
                {
                    sdl_ctrl.sdl_opt.write_output_png_compression = v;
                }
            }
            else if (strstr(s, "bmp") != NULL)
            {
                sdl_ctrl.sdl_opt.write_output_format = bmp;
            }
            break;

        case optSaturation:
            if (long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_ctrl.sdl_opt.gl_saturation = strtof(optarg, nullptr);

                if (sdl_ctrl.sdl_opt.gl_saturation < 0)
                    sdl_ctrl.sdl_opt.gl_saturation = 0;
                if (sdl_ctrl.sdl_opt.gl_saturation > 2)
                    sdl_ctrl.sdl_opt.gl_saturation = 2;
            }
            break;

        case optYuvInRaw16:
            sdl_ctrl.sdl_opt.yuv_in_raw16 = 1;
            break;

        case optBufferDebug:
            g_opts.buffer_debug = 1;
            break;

        case 'a':   // abort after x images
            g_opts.abort_after = atoi(optarg);
            break;

        case 'A':   // abort after x seconds if no image is received
            g_opts.abort_after_sec = atoi(optarg);
            break;

        case 'c':   // set reference image for loopback test
            g_opts.ref_image = optarg;
            break;

        case 'C':   // CSI-2 data format selection
            sdl_ctrl.sdl_opt.bits_per_component = atoi(optarg);
            sdl_ctrl.sdl_opt.is_csi2 = 1;
            break;

        case 'd':   // CSI-2 data type selection
            sdl_ctrl.sdl_opt.decode_csi2_datatype = strtol(optarg, NULL, 16);
            break;

        case 'D':   // set reference image for sxpfapp debugging
            g_opts.dbg_image = optarg;
            sdl_ctrl.sdl_opt.dbg_image = true;
            break;

        case 'e':   // override error filename
            g_opts.error_image_name = optarg;
            break;

        case 'f':
            sdl_ctrl.sdl_opt.start_fullscreen = 1;
            break;

        case 'F':   // write x filtered images to disk
            sdl_ctrl.sdl_opt.write_filtered_images = atoi(optarg);
            break;

        case 'g':
            c = sscanf(optarg, "%dx%d@%dx%d", &w, &h, &x, &y);
            if (c < 2)
            {
                fprintf(stderr, "invalid geometry: %s\n", optarg);
                usage(argv[0]);
                exit(1);
            }
            sdl_ctrl.sdl_opt.win_width = w;
            sdl_ctrl.sdl_opt.win_height = h;
            if (c >= 3)
                sdl_ctrl.sdl_opt.win_x_pos = x;
            if (c >= 4)
                sdl_ctrl.sdl_opt.win_y_pos = y;
            break;

        case 'G':
            c = sscanf(optarg, "%dx%d", &w, &h);
            if (c != 2)
            {
                fprintf(stderr, "invalid geometry: %s\n", optarg);
                usage(argv[0]);
                exit(1);
            }
            sdl_ctrl.sdl_opt.aspect = (float) w / h;
            break;

        case 'h':
            usage(argv[0]);
            exit(0);

        case 'i':   // SONY IMX290 in DOL2 mode
            sdl_ctrl.sdl_opt.is_csi2 = 1;
            sdl_ctrl.sdl_opt.bits_per_component = 12;
            sdl_ctrl.sdl_opt.components_per_pixel = 1;
            sdl_ctrl.sdl_opt.left_shift = 4;
            sdl_ctrl.sdl_opt.shader = "debayer";
            sdl_ctrl.sdl_opt.bayer_pattern = 0;
            break;

        case 'l':   // left shift value
            v = atoi(optarg);
            if (v >= 0 && v <= 16)
                sdl_ctrl.sdl_opt.left_shift = v;
            break;

        case 'm':   // bit mapping file
        {
            FILE *fp = fopen(optarg, "r");
            if(fp == NULL)
            {
                fprintf(stderr, "Unable to open mapping file!\n");
                exit(1);
            }

            char line[1024];
            for (int i=0; i<32; i++)
            {
                sdl_ctrl.sdl_opt.bit_mapping[i] = -1;
                if (fgets(line, sizeof(line), fp) != NULL)
                {
                    if (strlen(line) >= 32)
                    {
                        for (int n=0; n<32; n++)
                        {
                            if (line[n] == '0')
                            {
                            }
                            else if (line[n] == '1')
                            {
                                sdl_ctrl.sdl_opt.bit_mapping[i] = n;
                            }
                            else
                            {
                                fprintf(stderr, "Invalid character format of mapping file!\n");
                                exit(1);
                            }
                        }
                    }
                    else
                    {
                        fprintf(stderr, "Invalid line format of mapping file!\n");
                        exit(1);
                    }
                }
                else
                {
                    fprintf(stderr, "Invalid format of mapping file!\n");
                    exit(1);
                }
            }
            break;
        }
        case 'n':   // override filename
            g_opts.image_name = optarg;
            break;

        case 'o':   // OpenLDI mode: RGB24 in 32 bit
            sdl_ctrl.sdl_opt.is_oldi = 1;
            sdl_ctrl.sdl_opt.bits_per_component = 8;
            sdl_ctrl.sdl_opt.components_per_pixel = 4;
            sdl_ctrl.sdl_opt.left_shift = 0;
            sdl_ctrl.sdl_opt.shader = "oldi";
            break;

        case 'O':   // write rendered output images
            sdl_ctrl.sdl_opt.write_output_images = atoi(optarg);
            break;

        case 'p':   // select bayer pattern
            v = atoi(optarg);
            if (v >= 0 && v < 4)
                sdl_ctrl.sdl_opt.bayer_pattern = atoi(optarg);
            else
                fprintf(stderr, "invalid bayer pattern selection! using %d.\n",
                        sdl_ctrl.sdl_opt.bayer_pattern);
            break;

        case 'q':
            sdl_ctrl.sdl_opt.show_sdl = 0;
            break;

        case 'r':   // raw input
            sdl_ctrl.sdl_opt.components_per_pixel = 1;
            sdl_ctrl.sdl_opt.shader = "debayer";
            break;

        case 'R':   // RGBA input with 32bpp
            v = atoi(optarg);   // bits per pixel
            if (v != 24 && v != 32)
            {
                fprintf(stderr, "invalid RGB format\n");
                exit(1);
            }
            sdl_ctrl.sdl_opt.components_per_pixel = v / 8;
            sdl_ctrl.sdl_opt.bits_per_component = 8;
            sdl_ctrl.sdl_opt.left_shift = 0;
            sdl_ctrl.sdl_opt.shader = "rgb";
            break;

        case 's':   // override shader
            sdl_ctrl.sdl_opt.shader = optarg;
            break;

        case 't':   // enable pixel test
            sdl_ctrl.sdl_opt.pix_test = 1;
            break;

        case 'T':   // timeout after x seconds
            g_opts.timeout_after_sec = atoi(optarg);
            break;

        case 'v':   // write video.raw file
            g_opts.write_video = 1;
            break;

        case 'V':   // verify CSI-2 data
            sdl_ctrl.sdl_opt.verify_csi2 = 1;
            break;

        case 'w':   // write x images to disk
            g_opts.write_images = atoi(optarg);
            break;

        case 'W':   // write x error images to disk
            g_opts.write_error_images = atoi(optarg);
            break;

        case '8':   // 8 bit per component
            sdl_ctrl.sdl_opt.bits_per_component = 8;
            sdl_ctrl.sdl_opt.components_per_pixel = 2;
            sdl_ctrl.sdl_opt.left_shift = 0;
            sdl_ctrl.sdl_opt.shader = "uyvy";
            break;

        case '?':
            if(optopt)
            {
                printf("invalid short option '%c'\n", optopt);
                usage(argv[0]);
                exit(1);
            }
            else
            {
                sdl_long_opt_grade = SdlGetLongOptionGrade(argv[curr_index]);
                if(sdl_long_opt_grade == sdlopt_invalid)
                {
                    printf("invalid long option '%s'!\n", argv[curr_index]);
                    usage(argv[0]);
                    exit(1);
                }
                else if(sdl_long_opt_grade == sdlopt_valid_no_params)
                {
                    // valid "--sdl-<xx>" option without required parameter,
                    // continue
                }
                else if(sdl_long_opt_grade == sdlopt_valid_with_params)
                {
                    // for "--sdl-<xx>" options, skip next argv
                    // (which is the option's argument value)
                    optind++;
                }
            }
            break;

        default:
            fprintf(stderr, "Invalid switch: -%c\n", c);
            usage(argv[0]);
            exit(1);
        }
    }

    int sdl_option_result;

    optind = 1;
    sdl_option_result = SdlLongOptions(argc, argv, &sdl_ctrl.sdl_opt);
    if(sdl_option_result < 0)
    {
        usage(argv[0]);
        exit(1);
    }

    if (opts->endpoint_id < 0)
    {
        if (optind < argc)
        {
            opts->endpoint_id = atoi(argv[optind]) / 4;
            opts->channel_id  = atoi(argv[optind]) % 4;
            optind++;
        }
        else
        {
            opts->endpoint_id = 0;
            opts->channel_id  = 0;
        }
    }
}


static void save_file_sxpfapp(const char *name, uint8_t* buff, size_t bufsize)
{
    FILE    *file = fopen(name, "wb");

    if (file)
    {
#if defined(__aarch64__)
        // align size for memcpy (called by fwrite) from uncached memory to
        // prevent Bus error due to unaligned access
#define ALIGN_SIZE  (128)   // normal cache-line size is 64 for ARM64
        size_t  aligned_size = bufsize & ~(ALIGN_SIZE - 1);

        size_t  written = fwrite(buff, 1, aligned_size, file);

        if (written == aligned_size &&  // main chunk sucessfully written
            aligned_size < bufsize)     // and there is still a remainder
        {
            uint8_t staging[ALIGN_SIZE];

            // copy more than the remainder into the staging area to ensure we
            // perform an aligned read access
            memcpy(staging, buff + written, ALIGN_SIZE);

            // but write only the actually remaining bytes
            written += fwrite(staging, 1, bufsize - written, file);
        }
#else
        size_t  written = fwrite(buff, 1, bufsize, file);
#endif
        fclose(file);

        fprintf(stderr, "\nwrote %d bytes to file '%s'.\n", (int)written, name);
    }
    else
    {
        fprintf(stderr, "\ncould not write to file '%s'.\n", name);
    }
}


int main(int argc, char **argv)
{
    int endpoint_id = 0;
    int channel_id = 0;
    double rxtime;
    input_channel_t *ch = &channels[0]; // for now: use only one channel
    unsigned int ref_img_size;
    int ret = 0;
    int frame_drop_cnt = 0;
    int frame_slot = 0;

    sxpf_image_header_t *img_hdr;
    uint64_t ots = 0;
    uint64_t ts = 0;
    double frame_rate = 0.0;

    __s64 sys_time_start;
    __s64 sys_clock_rate;
    frame_buf_timing_t frame_buf_timing[31]; // max 31 frame buffers
    frame_buf_timing_t frame_buf_timing_sort[31];

    std::vector<std::thread> workers;

    setbuf(stdout, NULL);

    // per default, show the sdl window
    sdl_ctrl.sdl_opt.show_sdl = 1;

    g_opts.endpoint_id = -1;
    options(argc, argv, &g_opts);
    endpoint_id = g_opts.endpoint_id;
    channel_id = g_opts.channel_id;

    time(&timeOld);
    time(&timeStart);

    sys_time_start = sxpf_get_system_time(SXPF_CLOCK_DEFAULT);
    sys_clock_rate = sxpf_get_system_clock_rate(SXPF_CLOCK_DEFAULT);

    if ( strlen(g_opts.ref_image) != 0 )
    {
        FILE *r = fopen(g_opts.ref_image, "rb");
        if (r == NULL)
        {
            fprintf(stderr, "Error opening reference image\n");
        }
        if (fseek(r, 0L, SEEK_END) != 0)
        {
            fprintf(stderr, "Error seeking file\n");
        }
        ref_img_size = ftell(r);
        fseek(r, 0L, SEEK_SET);
        if (fread(ref_img, 1, ref_img_size, r) != ref_img_size)
        {
            fprintf(stderr, "Error reading file\n");
        }
    }

    if (g_opts.write_video != 0)
    {
        rawVideo = fopen("video.raw", "wb");
    }

    // open endpoint/grabber device 0
    init_channel(ch, endpoint_id, channel_id);

    // get card properties
    sxpf_get_card_properties(ch->fg, &props);

    // allow clean shut-down when CTRL-C is pressed
    signal(SIGINT, sigint_handler);

    if (sdl_ctrl.sdl_opt.show_sdl != 0)
    {
        int prepare_sdl = SdlPrepare(&sdl_ctrl);
        if(prepare_sdl != 0)
        {
            printf("prepare_sdl failed %d.\n", prepare_sdl);
            return 1;
        }
    }

    while (!sdl_ctrl.sdl_done)
    {
        sdl_ctrl.new_frame_info = -1;

        // wait for new frame
        if (ch->devfd > 0)
        {
            sxpf_event_t    events[20];
            sxpf_event_t    *evt = events;
            ssize_t         len;

#if 0
            uint32_t        latest_frame;
            if (!sxpf_get_latest_frame_id(ch->fg, &latest_frame))
            {
                // shorten select timeout, if we are lagging behind in
                // processing, probably after a short pause period (of less
                // than 32 frames)
                if ((int)(latest_frame - ch->frame) > 1 && !sdl_ctrl.is_pause)
                    timeout.tv_usec = 100;
            }
#endif

            len = sxpf_wait_events(1, &ch->devfd, 5 /* ms */);

            if (len > 0)
                len = sxpf_read_event(ch->fg, events, NELEMENTS(events));

            while (len > 0)
            {
                switch (evt->type)
                {
                case SXPF_EVENT_FRAME_RECEIVED:
                    rxtime = getElapsedTime();
                    ch->last_rxtime = rxtime;
                    frame_slot = evt->data / (1 << 24);
                    frame_buf_timing[frame_slot].receive_time =
                        sxpf_get_system_time(SXPF_CLOCK_DEFAULT) -
                        sys_time_start;
                    frame_buf_timing[frame_slot].frame_buffer = frame_slot;

                    // if (g_opts.buffer_debug != 0)
                    // {
                    //     memcpy(frame_buf_timing_sort,
                    //            frame_buf_timing,
                    //            sizeof(frame_buf_timing_t) * 31);
                    //     std::sort(frame_buf_timing_sort,
                    //     frame_buf_timing_sort+31,
                    //     []( const frame_buf_timing_t& a,
                    //         const frame_buf_timing_t&b ) {
                    //                   return a.receive_time > b.receive_time;
                    //               });

                    //     printf("\n");
                    //     for (__u32 i=0; i<props.num_buffers; i++)
                    //     {
                    //         printf("#%d:%.02f ",
                    //                frame_buf_timing_sort[i].frame_buffer,
                    //                (float)frame_buf_timing_sort[i].receive_time /
                    //                 sys_clock_rate);
                    //     }
                    //     printf("\n");
                    // }

                    img_hdr =
                        (sxpf_image_header_t*)sxpf_get_frame_ptr(ch->fg,
                                                                 frame_slot);
                    // if (!img_hdr)
                    // {
                    //     printf("Failed getting Frame buffer pointer!\n");
                    // }
                    // else
                    // {
                    //     ts = img_hdr->ts_start_hi * 0x100000000ull +
                    //         img_hdr->ts_start_lo;
                    //     frame_rate = 40e6 / (ts - ots);
                    //     ots = ts;
                    // }

                    if (!img_hdr || sdl_ctrl.is_pause)
                    {
                        // even if we ignore the received image, we need to
                        // release its resources!
                        frame_buf_timing[frame_slot].release_time =
                            sxpf_get_system_time(SXPF_CLOCK_DEFAULT) -
                            sys_time_start;
                        release_buffer(ch->fg, evt->data / (1 << 24));
                    }
                    else
                    {
                        if((abs(sdl_ctrl.sdl_opt.drop_show_count) < 2 ) ||
                           (sdl_ctrl.sdl_opt.drop_show_count >= 0 &&
                             (frame_drop_cnt % abs(sdl_ctrl.sdl_opt.drop_show_count)
                             ) != 0) ||
                             (sdl_ctrl.sdl_opt.drop_show_count < 0 &&
                              (frame_drop_cnt % abs(sdl_ctrl.sdl_opt.drop_show_count))
                             == 0))
                        {
                            if (sdl_ctrl.new_frame_info != -1)
                            {

                                // we have already received a frame
                                // in this loop
                                // --> release it back to the hardware, since
                                // we won't use it
                                sxpf_release_frame(ch->fg,
                                                   sdl_ctrl.new_frame_info /
                                                    (1 << 24), 0);

                                fprintf(stderr, "\nImage processing too slow. Dropping frame.\n"); // 최초 아니면, 아주 가끔 발생하는데, 값이 있어야 한다는 의미겠군.
                            }
                            sdl_ctrl.new_frame_info = evt->data;
                            printf("------------stl ctrl new frame info : %d\n", evt->data); //이벤트가 수신되면 data에 값이 생기는데, 이게 무슨의미인지 모르겠네. 버퍼에 쌓이는 byte array 크기를 의미하나..
                        }
                        else 
                        {
                            frame_buf_timing[frame_slot].release_time =
                                sxpf_get_system_time(SXPF_CLOCK_DEFAULT) -
                                sys_time_start;
                            sxpf_release_frame(ch->fg, frame_slot, 0);
                        }

                        frame_drop_cnt++;
                    }
                    break;

                case SXPF_EVENT_I2C_MSG_RECEIVED:
                    // ignore i2c messages for now
                    release_buffer(ch->fg, evt->data / (1 << 24));
                    break;

                case SXPF_EVENT_SW_TEST:
                    fprintf(stderr, "\nTest Interrupt received!\n");
                    break;

                case SXPF_EVENT_TRIGGER:
                    //fprintf(stderr, "\nTrigger Interrupt received (timestamp"
                    //        "= 0x%016llx)!\n", evt->extra.timestamp);
                    break;

                case SXPF_EVENT_CAPTURE_ERROR:
                    fprintf(stderr, "\ncapture error: 0x%08x\n", evt->data);
                    break;

                case SXPF_EVENT_IO_STATE:
                    if (evt->data != SXPF_IO_NORMAL)
                    {
                        fprintf(stderr, "\nPCIe error (%d) - aborting\n",
                                        evt->data);
                        exit(evt->data & 255);
                    }
                    break;
                }

                len--;
                ++evt;
            }

            if (len != 0 && errno != EAGAIN)
            {
                perror("error reading sxpf");
                break;
            }
        }

        if (sdl_ctrl.sdl_opt.show_sdl != 0)
        {
            // printf("---show\n"); //여기에 진입은 하는데, 이미지를 얻는것과는 별개인듯. rendering window와 관련있음. 이미지 캡쳐와는 별개
            SdlPollEvents(&sdl_ctrl);
        }

        if (sdl_ctrl.new_frame_info != -1 || strlen(g_opts.dbg_image))
        {
            // printf("new frame\n"); // 새로운 프레임이 들어오면 여기에 들어오네
            int isNewFrame;

            sdl_ctrl.frame_info = ch->frame_info;

            if (strlen(g_opts.dbg_image))
            {
                sdl_ctrl.img_hdr = load_dbg_image(g_opts.dbg_image);
            }
            else
            {
                // 기본은 여기에 진입
                sdl_ctrl.img_hdr =
                    (sxpf_image_header_t*)sxpf_get_frame_ptr(ch->fg, frame_slot);
            }

            // printf("-----sdl ctrl save index : %d\n", sdl_ctrl.save_idx); // 대부분 0으로 뜨는데, 왜 필요한지 잘...
            sdl_ctrl.save_idx_old = sdl_ctrl.save_idx;
            

            isNewFrame = (sdl_ctrl.new_frame_info & 0x00ffffff) > 0; // 새로운 프레임인지 체크
            
            if(isNewFrame)
            {
                printf("------is new frame : %d\n", isNewFrame);

                time(&timeOld);

                if (g_opts.abort_after > 0)
                {
                    g_opts.abort_after--;
                }
                if (g_opts.abort_after == 0)
                {
                    sdl_ctrl.sdl_done = SDL_TRUE;
                }

                if (sdl_ctrl.doSaveImage ||
                    (g_opts.write_images > 0) || (g_opts.write_images == -1))
                {
                    printf("----save?"); // 여기에 진입하지 않음. 옵션인가봄
                    char buf[1024];
                    sprintf(buf, g_opts.image_name, sdl_ctrl.save_idx);
                    sdl_ctrl.increment_save_idx = 1;
                    save_file_sxpfapp(buf,
                        (uint8_t*)sdl_ctrl.img_hdr,
                        sdl_ctrl.img_hdr->frame_size);
                }

                if (g_opts.write_images > 0)
                {
                    g_opts.write_images--;
                }

                if (strlen(g_opts.ref_image) != 0)
                {
                    uint32_t memcmp_ret = memcmp(sdl_ctrl.img_hdr + 1,
                        ref_img + 32,
                        sdl_ctrl.img_hdr->frame_size - 64);

                    if (memcmp_ret != 0)
                    {
                        sdl_ctrl.error_frame = 1;
                        printf(" Reference image memcmp = %d (total %d)\n",
                            memcmp_ret,
                            sdl_ctrl.total_err_cnt);
                    }
                }


                if (g_opts.write_video != 0)
                {
                    fwrite(sdl_ctrl.img_hdr,
                           sdl_ctrl.img_hdr->frame_size,
                           1,
                           rawVideo);
                }

                if (sdl_ctrl.error_frame)
                {
                    if ((g_opts.write_error_images > 0) |
                        (g_opts.write_error_images == -1))
                    {
                        char buf[1024];
                        sprintf(buf,
                            g_opts.error_image_name,
                            sdl_ctrl.total_err_cnt++);
                        save_file_sxpfapp(buf,
                            (uint8_t*)sdl_ctrl.img_hdr,
                            sdl_ctrl.img_hdr->frame_size);
                    }

                    if (g_opts.write_error_images > 0)
                    {
                        g_opts.write_error_images--;
                    }
                }
            }


            printf("---- x:%ld, y:%ld, fps:%d\n", sdl_ctrl.frame_x_size, sdl_ctrl.frame_y_size, frame_rate); //여기에서 해상도가 이미 1920x1080으로 나옴, fps=23?
            SdlUpdateTexture(&sdl_ctrl, frame_rate); //결국 여기에서 window에 업데이트
            printf("---- x:%ld, y:%ld, fps:%d\n", sdl_ctrl.frame_x_size, sdl_ctrl.frame_y_size, frame_rate); //여기에서 해상도가 이미 1920x1080으로 나옴

            fprintf(stderr, "%s%s", g_opts.newline_mode == 1 ? "\n" : "\r", sdl_ctrl.output_str);

            ch->frame_info = sdl_ctrl.frame_info;

            frame_buf_timing[frame_slot].release_time =
                sxpf_get_system_time(SXPF_CLOCK_DEFAULT) - sys_time_start;

            if (strlen(g_opts.dbg_image))
            {
                free(sdl_ctrl.img_hdr);
            }
            else
            {
                release_buffer(ch->fg, sdl_ctrl.new_frame_info / (1 << 24));
            }
        }

        // start display only after the first image has been received - this
        // prevents OpenGL errors due to an invalid view frustum
        if (sdl_ctrl.sdl_opt.show_sdl != 0 && sdl_ctrl.frame_x_size > 0)
        {
            SdlRenderTexture(&sdl_ctrl);

            if (sdl_ctrl.new_frame_info != -1 || strlen(g_opts.dbg_image))
            {
                printf("----1\n");
                if (((sdl_ctrl.doSaveOutput != 0) ||
                     (sdl_ctrl.sdl_opt.write_output_images > 0) ||
                     (sdl_ctrl.sdl_opt.write_output_images == -1)) &&
                    (sdl_ctrl.output_image_buffer != 0))
                {
                    // 진입하지 않음.
                    workers.push_back(std::thread([](int idx,
                                                     int w,
                                                     int h,
                                                     void* buffer,
                                                     sdl_ctrl_t* sdl_ctrl_ptr)
                    {
                        char buf[1024];
                        sprintf(buf, g_opts.image_name, idx);

                        stbi_flip_vertically_on_write(true);

                        if (sdl_ctrl_ptr->sdl_opt.write_output_format == jpg)
                        {
                            strcat(buf, ".jpg");
                            stbi_write_jpg(buf, w, h, 4, buffer, sdl_ctrl_ptr->sdl_opt.write_output_jpg_quality);
                        }
                        else if (sdl_ctrl_ptr->sdl_opt.write_output_format == png)
                        {
                            strcat(buf, ".png");
                            stbi_write_png_compression_level = sdl_ctrl_ptr->sdl_opt.write_output_png_compression;
                            stbi_write_png(buf, w, h, 4, buffer, w * 4);
                        }
                        else if (sdl_ctrl_ptr->sdl_opt.write_output_format == bmp)
                        {
                            strcat(buf, ".bmp");
                            stbi_write_bmp(buf, w, h, 4, buffer);
                        }
                        fprintf(stderr, "\nwrote file '%s'.\n", buf);

                        free(buffer);
                    }, sdl_ctrl.save_idx_old,
                        sdl_ctrl.sdl_window_width,
                        sdl_ctrl.sdl_window_height,
                        sdl_ctrl.output_image_buffer,
                        &sdl_ctrl));
                    sdl_ctrl.output_image_buffer = 0;

                    if (sdl_ctrl.save_idx == sdl_ctrl.save_idx_old)
                    {
                        sdl_ctrl.save_idx++;
                    }
                }
                if (sdl_ctrl.sdl_opt.write_output_images > 0)
                {
                    // 진입하지 않음. 
                    sdl_ctrl.sdl_opt.write_output_images--;
                }
            }
        }

        if (g_opts.abort_after_sec > 0)
        {
            time_t timeNow;
            time(&timeNow);
            if ((timeOld != 0) &&
                ((int)difftime(timeNow, timeOld) > g_opts.abort_after_sec))
            {
                ret = -1;
                sdl_ctrl.sdl_done = SDL_TRUE;
            }
        }

        if (g_opts.timeout_after_sec > 0)
        {
            time_t timeNow;
            time(&timeNow);
            if ((timeStart != 0) &&
                ((int)difftime(timeNow, timeStart) > g_opts.timeout_after_sec))
            {
                ret = 0;
                sdl_ctrl.sdl_done = SDL_TRUE;
            }
        }
    }   // while (!sdl_ctrl.sdl_done)

    sxpf_stop(ch->fg, SXPF_STREAM_ALL);

    if (g_opts.write_video != 0)
    {
        fclose(rawVideo);
    }

    std::for_each(workers.begin(), workers.end(),
                  [](std::thread &t) { t.join(); });

    SdlCleanup(&sdl_ctrl);

    fprintf(stderr, "\nTotal frames with bit errors: %d\n", sdl_ctrl.total_err_cnt);

    quit(ret);

    // not reached
    return ret;
}

/* vi: set ts=4 sw=4 expandtab: */
