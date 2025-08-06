#include "update_texture.h"
#include "helpers.h"

#include "csi-2.h"
#include "imx290.h"
#include "gl_fbo.h"

#include <inttypes.h>

#include <getopt.h>
#include <set>

enum Opt
{
    // ensure long-options-only parameters have a code unequal to any character
    _ = 256,
    sdl_Borderless,
    sdl_Background,
    sdl_C,
    sdl_d,
    sdl_Drop,
    sdl_DropColumn,
    sdl_f,
    sdl_F,
    sdl_g,
    sdl_Gamma,
    sdl_G,
    sdl_HFlip,
    sdl_i,
    sdl_l,
    sdl_m,
    sdl_o,
    sdl_O,
    sdl_OutputFormat,
    sdl_p,
    sdl_r,
    sdl_R,
    sdl_s,
    sdl_Saturation,
    sdl_ShowTimestamp,
    sdl_t,
    sdl_VSync,
    sdl_V,
    sdl_VFlip,
    sdl_YuvInRaw16,
    sdl_Zoom,
    sdl_8,
    sdl_Process,
};

static struct option sdl_long_options[] =
{
    { "sdl-borderless",    no_argument,       0,  Opt::sdl_Borderless },
    { "sdl-bg",            required_argument, 0,  Opt::sdl_Background },
    { "sdl-C",             required_argument, 0,  Opt::sdl_C },
    { "sdl-d",             required_argument, 0,  Opt::sdl_d },
    { "sdl-drop",          required_argument, 0,  Opt::sdl_Drop },
    { "sdl-drop-column",   required_argument, 0,  Opt::sdl_DropColumn },
    { "sdl-f",             no_argument,       0,  Opt::sdl_f },
    { "sdl-F",             required_argument, 0,  Opt::sdl_F },
    { "sdl-g",             required_argument, 0,  Opt::sdl_g },
    { "sdl-gamma",         required_argument, 0,  Opt::sdl_Gamma },
    { "sdl-G",             required_argument, 0,  Opt::sdl_G },
    { "sdl-hflip",         no_argument,       0,  Opt::sdl_HFlip },
    { "sdl-i",             no_argument,       0,  Opt::sdl_i },
    { "sdl-l",             required_argument, 0,  Opt::sdl_l },
    { "sdl-m",             required_argument, 0,  Opt::sdl_m },
    { "sdl-o",             no_argument,       0,  Opt::sdl_o },
    { "sdl-O",             required_argument, 0,  Opt::sdl_O },
    { "sdl-output-format", required_argument, 0,  Opt::sdl_OutputFormat },
    { "sdl-p",             required_argument, 0,  Opt::sdl_p },
    { "sdl-r",             no_argument,       0,  Opt::sdl_r },
    { "sdl-R",             required_argument, 0,  Opt::sdl_R },
    { "sdl-s",             required_argument, 0,  Opt::sdl_s },
    { "sdl-saturation",    required_argument, 0,  Opt::sdl_Saturation },
    { "sdl-show-ts",       no_argument,       0,  Opt::sdl_ShowTimestamp },
    { "sdl-t",             no_argument,       0,  Opt::sdl_t },
    { "sdl-vflip",         no_argument,       0,  Opt::sdl_VFlip },
    { "sdl-no-vsync",      no_argument,       0,  Opt::sdl_VSync },
    { "sdl-V",             no_argument,       0,  Opt::sdl_V },
    { "sdl-yuv-in-raw16",  no_argument,       0,  Opt::sdl_YuvInRaw16 },
    { "sdl-zoom",          required_argument, 0,  Opt::sdl_Zoom },
    { "sdl-8",             no_argument,       0,  Opt::sdl_8 },
    { "sdl-process",       required_argument, 0,  Opt::sdl_Process},
    { 0, 0, 0, 0 }
};

static void gl_debug_callback(GLenum source,
                              GLenum type,
                              GLuint id,
                              GLenum severity,
                              GLsizei length,
                              const GLchar *message,
                              GLvoid *usrParam);


void SdlUsage()
{
    printf("\nSDL options:\n"
           "\t--sdl-borderless       Show SDL window in borderless mode\n"
           "\t--sdl-bg 0xrrggbb      Set background color for SDL window (default: 0x4c4cff)\n"
           "\t--sdl-C 10|12|14|16    Assume CSI-2 format with 10, 12, 14 or 16 bits per component\n"
           "\t--sdl-d dt             CSI-2 datatype/virtual channel to parse RAW CSI-2 data format\n"
           "\t                       Bit 7:6 = virtual channel\n"
           "\t                       Bit 5:0 = datatype\n"
           "\t--sdl-drop x           Drop/show every x frame\n"
           "\t--sdl-drop-column x    Only use every x column for the CSI-2 decoded image output\n"
           "\t--sdl-f                Start fullscreen\n"
           "\t--sdl-F num            Write number of filtered CSI-2 images to disk (-1 = save all)\n"
           "\t                       Filtering is based on -d parameter\n"
           "\t--sdl-g WxH[@XxY]      Set window geometry, e.g. -g 1280x720\n"
           "\t                       Optionally add window position, e.g. -g 1280x720@0x0\n"
           "\t--sdl-gamma x          Set gamma value (0.5-2.0, default: 1.2)\n"
           "\t--sdl-G WxH            Force image geometry and disregard geometry of received data\n"
           "\t--sdl-hflip            Flip image horizontally\n"
           "\t--sdl-i                Input is 12bpp packed raw DOL2 from SONY IMX290\n"
           "\t--sdl-l num            Shift input pixels left by this amount (default: 0)\n"
           "\t--sdl-m file           Mix incoming data bits based on the given mapping file\n"
           "\t--sdl-o                Input is RGB24 in 32 bit per pixel OpenLDI-scrambled\n"
           "\t--sdl-output-format x  Set format of rendered output images when using -O parameter to\n"
           "\t                       jpg, png or bmp (default: jpg:90)\n"
           "\t                       Optionally set\n"
           "\t                       jpg quality (0...100) using parameter 'jpg:quality'\n"
           "\t                       png compression level (0..10) using parameter 'png:compression'\n"
           "\t--sdl-O num            Write number of rendered OpenGL output images to disk (-1 = save all)\n"
           "\t                       use -g parameter to set image size\n"
           "\t--sdl-p 0..3           Select 1 out of 4 possible Bayer patterns\n"
           "\t--sdl-process <format> Select format to process data\n"
           "\t                       swap16: interprete CSI-2 data as RAW16 and swap bytes\n"
           "\t--sdl-r                RAW input: de-bayer received images\n"
           "\t--sdl-R 24|32          Select RGB input (24bit RGB or 32bit RGBA)\n"
           "\t--sdl-s shader         Override shader file(s): filename without extension\n"
           "\t                       Built-in shaders: %s\n"
           "\t--sdl-saturation x     Set saturation value (0.0-2.0, default: 1.0)\n"
           "\t--sdl-show-ts          Show timestamp informations\n"
           "\t--sdl-t                Enable pixel test\n"
           "\t--sdl-vflip            Flip image vertically\n"
           "\t--sdl-no-vsync         Disable V-Sync\n"
           "\t--sdl-V                Verify CSI-2 data (check for CRC errors)\n"
           "\t--sdl-yuv-in-raw16     YUV data is captured in RAW16 datatype e.g. using "
           "a camAD3 DUAL MAX96705/96706\n"
           "\t--sdl-zoom x           Set zoom value (1.0-5.0, default: 1.0)\n"
           "\t--sdl-8                Input is compacted to 8 bit per component\n"
           "\n"
           "SDL key control:\n"
           "\t'ESC'                  Close SDL window\n"
           "\t'f'                    Toggle fullscreen mode\n"
           "\t'g'                    Increase gamma\n"
           "\t'Shift+g'              Decrease gamma\n"
           "\t'o'                    Write current rendered OpenGL output image to disk\n"
           "\t'r'                    Write current image to disk\n"
           "\t's'                    Increase saturation\n"
           "\t'Shift+s'              Decrease saturation\n"
           "\t'v'                    Toggle sync to VSYNC\n"
           "\t'z'                    Zoom in\n"
           "\t'Shift+z'              Zoom out\n"
           , list_shaders().c_str());
}

// depending on wether the passed option is a long option with or without
// parameter this function returns sdlopt_valid_no_params or
// sdlopt_valid_with_params. If the passed option is not a valid sdl option,
// invalid will be returned.
sdlopt_long_option_grade_e SdlGetLongOptionGrade(const char *argv)
{
    int optIndex = 0;

    while (sdl_long_options[optIndex].name != 0)
    {
        if(strstr(argv, sdl_long_options[optIndex].name) != NULL)
        {
            return (sdl_long_options[optIndex].has_arg == required_argument) ?
                           sdlopt_valid_with_params : sdlopt_valid_no_params;
        }
        optIndex++;
    }

    return sdlopt_invalid;
}

int SdlLongOptions(int argc, char **argv, sdl_opt_t* sdl_rc)
{
    int     value;
    int     width, height, x_pos, y_pos;
    char    s[4];

    int option_index;

    while (1)
    {
        int       c, w, h;
        int curridx = 0;
        option_index = 0;

        c = -1;
        while (optind <= argc-1)
        {
            curridx = optind;
            c = getopt_long(argc, argv, "", sdl_long_options, &option_index);
            // check agains curridx is needed as getopt_long reduces optind
            // when last parameter with argument is reached
            if (c != -1 || optind < curridx)
                break;
            optind++;
        }

        if (c == -1)
            break;

        switch (c)
        {
        case Opt::sdl_Borderless:
            sdl_rc->borderless = 1;
            break;
        case Opt::sdl_Background:
            if (sdl_long_options[option_index].flag != 0)
                break;
            if (optarg)
                sdl_rc->background = strtol(optarg, NULL, 16);
            break;
        case Opt::sdl_C:   // CSI-2 data format selection
            sdl_rc->bits_per_component = atoi(optarg);
            sdl_rc->is_csi2 = 1;
            break;
        case Opt::sdl_d:   // CSI-2 data type selection
            sdl_rc->decode_csi2_datatype = strtol(optarg, NULL, 16);
            break;
        case Opt::sdl_Drop:
            if (sdl_long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_rc->drop_show_count = atoi(optarg);
            }
            break;
        case Opt::sdl_DropColumn:
            if (sdl_long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_rc->drop_column = atoi(optarg);
            }
            break;
        case Opt::sdl_f:
            sdl_rc->start_fullscreen = 1;
            break;
        case Opt::sdl_F:   // write x filtered images to disk
            sdl_rc->write_filtered_images = atoi(optarg);
            break;
        case Opt::sdl_g:
            c = sscanf(optarg, "%dx%d@%dx%d", &width, &height, &x_pos, &y_pos);
            if (c < 2)
            {
                fprintf(stderr, "invalid geometry: %s\n", optarg);
                return -1;
            }
            sdl_rc->win_width = width;
            sdl_rc->win_height = height;
            if (c >= 3)
                sdl_rc->win_x_pos = x_pos;
            if (c >= 4)
                sdl_rc->win_y_pos = y_pos;
            break;
        case Opt::sdl_Gamma:
            if (sdl_long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_rc->gl_gamma = strtof(optarg, nullptr);
                if (sdl_rc->gl_gamma < 0.5)
                    sdl_rc->gl_gamma = 0.5;
                if (sdl_rc->gl_gamma > 2)
                    sdl_rc->gl_gamma = 2;
            }
            break;
        case Opt::sdl_G:
            c = sscanf(optarg, "%dx%d", &w, &h);
            if (c != 2)
            {
                fprintf(stderr, "invalid geometry: %s\n", optarg);
                return -1;
            }
            sdl_rc->aspect = (float)w / h;
            break;
        case Opt::sdl_HFlip:
            sdl_rc->hflip = 1;
            break;
        case Opt::sdl_i:   // SONY IMX290 in DOL2 mode
            sdl_rc->is_csi2 = 1;
            sdl_rc->bits_per_component = 12;
            sdl_rc->components_per_pixel = 1;
            sdl_rc->left_shift = 4;
            sdl_rc->shader = "debayer";
            sdl_rc->bayer_pattern = 0;
            break;
        case Opt::sdl_l:   // left shift value
            value = atoi(optarg);
            if (value >= 0 && value <= 16)
                sdl_rc->left_shift = value;
            break;
        case Opt::sdl_m:   // bit mapping file
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
                sdl_rc->bit_mapping[i] = -1;
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
                                sdl_rc->bit_mapping[i] = n;
                            }
                            else
                            {
                                fprintf(stderr,
                                        "Invalid character format of mapping file!\n");
                                exit(1);
                            }
                        }
                    }
                    else
                    {
                        fprintf(stderr,
                                "Invalid line format of mapping file!\n");
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
        case Opt::sdl_o:   // OpenLDI mode: RGB24 in 32 bit
            sdl_rc->is_oldi = 1;
            sdl_rc->bits_per_component = 8;
            sdl_rc->components_per_pixel = 4;
            sdl_rc->left_shift = 0;
            sdl_rc->shader = "oldi";
            break;
        case Opt::sdl_O:   // write rendered output images
            sdl_rc->write_output_images = atoi(optarg);
            break;
        case Opt::sdl_OutputFormat:
            c = sscanf(optarg, "%3s:%d", s, &value);
            if (strstr(s, "jpg") != NULL)
            {
                sdl_rc->write_output_format = jpg;
                if (c > 1)
                {
                    sdl_rc->write_output_jpg_quality = value;
                }
            }
            else if (strstr(s, "png") != NULL)
            {
                sdl_rc->write_output_format = png;
                if (c > 1)
                {
                    sdl_rc->write_output_png_compression = value;
                }
            }
            else if (strstr(s, "bmp") != NULL)
            {
                sdl_rc->write_output_format = bmp;
            }
            break;
        case Opt::sdl_p:   // select bayer pattern
            value = atoi(optarg);
            if (value >= 0 && value < 4)
                sdl_rc->bayer_pattern = atoi(optarg);
            else
                fprintf(stderr, "invalid bayer pattern selection! using %d.\n",
                                   sdl_rc->bayer_pattern);
            break;
        case Opt::sdl_Process:
            if (strstr(optarg, "swap16") != NULL) {
                sdl_rc->process_mode = process_mode_e::swap16;
            }
            break;
        case Opt::sdl_r:  // raw input
            sdl_rc->components_per_pixel = 1;
            sdl_rc->shader = "debayer";
            break;

        case Opt::sdl_R:   // RGBA input with 32bpp
            value = atoi(optarg);   // bits per pixel
            if (value != 24 && value != 32)
            {
                fprintf(stderr, "invalid RGB format\n");
                exit(1);
            }
            sdl_rc->components_per_pixel = value / 8;
            sdl_rc->bits_per_component = 8;
            sdl_rc->left_shift = 0;
            sdl_rc->shader = "rgb";
            break;

        case Opt::sdl_s:   // override shader
            sdl_rc->shader = optarg;
            break;
        case Opt::sdl_Saturation:
            if (sdl_long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_rc->gl_saturation = strtof(optarg, nullptr);
                if (sdl_rc->gl_saturation < 0)
                    sdl_rc->gl_saturation = 0;
                if (sdl_rc->gl_saturation > 2)
                    sdl_rc->gl_saturation = 2;
            }
            break;
        case Opt::sdl_ShowTimestamp:
            sdl_rc->show_timestamp = 1;
            break;
        case Opt::sdl_t:   // enable pixel test
            sdl_rc->pix_test = 1;
            break;
        case Opt::sdl_VSync:   // disable V-Sync
            sdl_rc->v_sync = 0;
            break;
        case Opt::sdl_V:   // verify CSI-2 data
            sdl_rc->verify_csi2 = 1;
            break;
        case Opt::sdl_VFlip:
            sdl_rc->vflip = 1;
            break;
        case Opt::sdl_YuvInRaw16:
            sdl_rc->yuv_in_raw16 = 1;
            break;
        case Opt::sdl_Zoom:
            if (sdl_long_options[option_index].flag != 0)
                break;
            if (optarg)
            {
                sdl_rc->gl_zoom = strtof(optarg, nullptr);
                if (sdl_rc->gl_zoom < 1.0)
                    sdl_rc->gl_zoom = 1.0;
                if (sdl_rc->gl_zoom > 6.0)
                    sdl_rc->gl_zoom = 6.0;
            }
            break;
        case Opt::sdl_8:   // 8 bit per component
            sdl_rc->bits_per_component = 8;
            sdl_rc->components_per_pixel = 2;
            sdl_rc->left_shift = 0;
            sdl_rc->shader = "uyvy";
            break;
        case '?':
            if(optopt)
            {
                // do nothing, invalid short option already handled
                // in main module
            }
            else
            {
                // handle long option if concerning sdl, others are already
                // handled in main module
                if(strstr(argv[curridx], "--sdl-") != NULL)
                {
                    printf("invalid long option '%s'!\n", argv[curridx]);
                    return -1;
                }
            }
            break;
        default:
            printf("invalid long option '%s'!\n", argv[curridx]);
            return -1;
            break;
        }
    }

    return 0;
}

// helper function for debugging data
// should be finally removed
void printParameter(sdl_ctrl_t *sdl_ctrl)
{
    printf("UpdateTexture show_sdl=%d.\n", sdl_ctrl->sdl_opt.show_sdl);
    printf("UpdateTexture bits_per_component=%d.\n", sdl_ctrl->sdl_opt.bits_per_component);
    printf("UpdateTexture components_per_pixel=%d.\n", sdl_ctrl->sdl_opt.components_per_pixel);
    printf("UpdateTexture dbg_image=%d.\n", sdl_ctrl->sdl_opt.dbg_image);
    printf("UpdateTexture drop_show_count=%d.\n", sdl_ctrl->sdl_opt.drop_show_count);
    printf("UpdateTexture drop_column=%d.\n", sdl_ctrl->sdl_opt.drop_column);
    printf("UpdateTexture bit_mapping=%d.\n", sdl_ctrl->sdl_opt.bit_mapping[0]);
    printf("UpdateTexture pix_test=%d.\n", sdl_ctrl->sdl_opt.pix_test);
    printf("UpdateTexture decode_csi2_datatype=%d.\n", sdl_ctrl->sdl_opt.decode_csi2_datatype);
    printf("UpdateTexture is_csi2=%d.\n", sdl_ctrl->sdl_opt.is_csi2);
    printf("UpdateTexture yuv_in_raw16=%d.\n", sdl_ctrl->sdl_opt.yuv_in_raw16);
    printf("UpdateTexture left_shift=%d.\n", sdl_ctrl->sdl_opt.left_shift);
    printf("UpdateTexture field_sel=%d.\n", sdl_ctrl->field_sel);
    printf("UpdateTexture is_oldi=%d.\n", sdl_ctrl->sdl_opt.is_oldi);
    printf("UpdateTexture shader=%s.\n", sdl_ctrl->sdl_opt.shader);
    printf("UpdateTexture show_timestamp=%d.\n", sdl_ctrl->sdl_opt.show_timestamp);
    printf("UpdateTexture x_size=%d.\n", sdl_ctrl->frame_x_size);
    printf("UpdateTexture y_size=%d.\n", sdl_ctrl->frame_y_size);
    printf("UpdateTexture frame_info=%d.\n", sdl_ctrl->frame_info);
    printf("UpdateTexture new_frame_info=%d.\n", sdl_ctrl->new_frame_info);
    printf("UpdateTexture frame_cnt=%d.\n", sdl_ctrl->frame_cnt);
    printf("UpdateTexture fps=%.6f.\n", sdl_ctrl->fps);
    printf("UpdateTexture texture=%d.\n", sdl_ctrl->gl_texture);
}

int SdlPrepare(sdl_ctrl_t * sdl_ctrl)
{
    GLint          gl_x;
    const GLubyte *msg;

    /* Enable standard application logging */
    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Couldn't initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

#ifdef DO_SET_ATTRIBUTES
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    check_gl_error("set attributes");
#endif

    /* Create the window and renderer */
    sdl_ctrl->sdl_window = SDL_CreateWindow("SXPF Demo View",
                              sdl_ctrl->sdl_opt.win_x_pos,
                              sdl_ctrl->sdl_opt.win_y_pos,
                              sdl_ctrl->sdl_opt.win_width,
                              sdl_ctrl->sdl_opt.win_height,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN |
                              (sdl_ctrl->sdl_opt.borderless
                               ? SDL_WINDOW_BORDERLESS
                               : 0) |
                              (sdl_ctrl->sdl_opt.start_fullscreen
                               ? SDL_WINDOW_FULLSCREEN_DESKTOP
                               : SDL_WINDOW_RESIZABLE));
    if (!sdl_ctrl->sdl_window)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Couldn't create OpenGL window: %s\n", SDL_GetError());
        return 3; // quit(3);
    }

    SDL_GetWindowSize(sdl_ctrl->sdl_window,
                      &sdl_ctrl->sdl_window_width,
                      &sdl_ctrl->sdl_window_height);

    sdl_ctrl->sdl_glcontext = SDL_GL_CreateContext(sdl_ctrl->sdl_window);

    gl3wInit();

    msg = glGetString(GL_VERSION);
    printf("OpenGL version: %s\n", msg);

    msg = glGetString(GL_RENDERER);
    if (!check_gl_error("glGetString(GL_RENDERER)"))
        printf("OpenGL renderer: %s\n", msg);

    glDebugMessageCallbackARB((GLDEBUGPROCARB)gl_debug_callback, 0);
    glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE,
                             GL_DEBUG_SEVERITY_MEDIUM, 0, NULL, GL_TRUE);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    check_gl_error("glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS)");

    sdl_ctrl->renderer = get_shader(sdl_ctrl->sdl_opt.shader);

    sdl_ctrl->rgb_renderer = std::make_shared<GlImageRenderer>();

    GLuint glShader = sdl_ctrl->renderer->shader();
    sdl_ctrl->gl_shader_sourceSize = glGetUniformLocation(glShader,
                                                            "sourceSize");
    sdl_ctrl->gl_shader_firstRed = glGetUniformLocation(glShader,
                                                          "firstRed");
    sdl_ctrl->gl_shader_saturation = glGetUniformLocation(glShader,
                                                            "saturation");
    sdl_ctrl->gl_shader_gamma = glGetUniformLocation(glShader,
                                                       "gamma");

    glGenTextures(1, &sdl_ctrl->gl_texture);
    check_gl_error("glGenTextures");
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gl_x);
    check_gl_error("glGetIntegerv(GL_MAX_TEXTURE_SIZE)");
    printf("#%d max texture size: %d\n", sdl_ctrl->gl_texture, gl_x);

    glEnable(GL_TEXTURE_2D);
    check_gl_error("glEnable(GL_TEXTURE_2D)");

    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");

    sdl_ctrl->swap_interval = sdl_ctrl->sdl_opt.v_sync;
    // takes effect only when window is full-screen
    if (SDL_GL_SetSwapInterval(sdl_ctrl->swap_interval))
    {
        printf("SDL_GL_SetSwapInterval failed..: %s\n", SDL_GetError());
    }
    return 0;
}

void SdlCleanup(sdl_ctrl_t *sdl_ctrl)
{
    if (sdl_ctrl->sdl_opt.show_sdl != 0)
    {
        sdl_ctrl->renderer.reset();
        sdl_ctrl->rgb_renderer.reset();
        sdl_ctrl->backbuffer.cleanup();

        glDeleteTextures(1, &sdl_ctrl->gl_texture);
        glDeleteBuffers(sdl_ctrl->num_pbos, sdl_ctrl->pbo);

        SDL_GL_DeleteContext(sdl_ctrl->sdl_glcontext);

        SDL_DestroyWindow(sdl_ctrl->sdl_window);
    }
}

static void gl_debug_callback(GLenum source,
                              GLenum type,
                              GLuint id,
                              GLenum severity,
                              GLsizei length,
                              const GLchar *message,
                              GLvoid *userParam)
{
    (void)userParam;
    printf("(%x,%x,%x,%x) GL Debug: %.*s\n",
           source, type, id, severity, length, message);
}

//ToDo: offen/klären, see lin 791ff
extern sxpf_card_props_t props;


static void save_file(const char *name, uint8_t* buff, size_t bufsize)
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


int SdlUpdateTexture(sdl_ctrl_t *sdl_ctrl,
                     double capture_fps)
{
    uint8_t            *img_ptr;

    sxpf_image_header_t *img_hdr_filtered;
    uint8_t             *img_ptr_filtered = NULL;

    int                 back_pbo;
    int                 ret = 0;    // we have a new frame
    uint32_t            *tmp = NULL;
    uint32_t            frame_size;
    int                 isNewFrame =
                                 (sdl_ctrl->new_frame_info & 0x00ffffff) > 0;
    uint32_t            x_size, y_size;
    uint8_t             bpp;

    static uint64_t     ots = 0;
    uint64_t            ts;
    uint64_t            ts_end;
    double              frame_rate;
    static int          frame_rate_mean_idx = 0;
    static double       frame_rate_mean[1000];

    int                 bits_per_component = sdl_ctrl->sdl_opt.bits_per_component;

    static uint32_t     frame_counter_last = 0;
    uint32_t            frame_counter;
    uint32_t            error_cnt;
    uint32_t            sum = 0;
    char                msg[1024];
    int                 msglen = 0;

    // variables for raw csi2 decoding
    uint32_t            packet_offset;
    uint32_t            packet_size;
    uint32_t            filtered_offset;
    uint8_t             vc_dt;
    uint32_t            word_count;
    static uint32_t     align = 0;
    uint16_t            *pdst = NULL;
    uint32_t            decoded_pix;
    uint8_t             *pixels;
    uint32_t            bits_per_pixel;
    uint32_t            pixel_group_size;
    uint16_t            *tmp2 = NULL;
    uint16_t            crc;
    uint8_t             ecc;

    time_t rawtime;
    char * time_str;

    static GLint TEX_WIDTH_OLD = -1;
    static GLint TEX_HEIGHT_OLD = -1;

    std::set<uint8_t>   frame_datatypes;

    // create debug output
    static uint8_t showVariablesCnt = 0;
    if(showVariablesCnt > 0)
    {
        printParameter(sdl_ctrl);
        showVariablesCnt--;
    }

    if (!sdl_ctrl->img_hdr)
    {
        printf(" !img_hdr, exit\n");
        return -1;
    }

    x_size = sdl_ctrl->img_hdr->columns;
    y_size = sdl_ctrl->img_hdr->rows;
    bpp = sdl_ctrl->img_hdr->bpp;

    frame_size = x_size * y_size * bpp / 8; //without header 64byte
    printf("---frame size : %d\n", frame_size);
    ts = sdl_ctrl->img_hdr->ts_start_hi * 0x100000000ull +
                sdl_ctrl->img_hdr->ts_start_lo;
    ts_end = sdl_ctrl->img_hdr->ts_end_hi * 0x100000000ull +
                sdl_ctrl->img_hdr->ts_end_lo;
    frame_rate = 40e6 / (ts - ots);
    if (abs(sdl_ctrl->sdl_opt.drop_show_count) >= 2)
    {
        frame_rate_mean[frame_rate_mean_idx] = frame_rate;
        frame_rate_mean_idx =
            (frame_rate_mean_idx + 1) % (abs(sdl_ctrl->sdl_opt.drop_show_count) - 1);
        frame_rate = 0.0;
        for (int i=0; i<abs(sdl_ctrl->sdl_opt.drop_show_count) - 1; i++)
        {
            frame_rate += frame_rate_mean[i];
        }
        frame_rate /= (abs(sdl_ctrl->sdl_opt.drop_show_count) - 1);
    }

    img_ptr = (uint8_t*)sdl_ctrl->img_hdr +
                        sdl_ctrl->img_hdr->payload_offset;

    unsigned short payload_offset = sdl_ctrl->img_hdr->payload_offset;
    printf("--- payload offset : %d\n", payload_offset);

    //ToDo: offen/klären
    // ToDo: comming from sxpfapp, this check should be done there
    /*if (sdl_ctrl->img_hdr->frame_size > props.buffer_size && strlen(sdl_ctrl->dbg_image) == 0)
    {
        printf("Error: Received frame is larger than buffer (%u > %u).\n"
#ifdef _WIN32
               "Please increase the service's framesize option.\n",
#else
               "Please increase the kernel module's framesize parameter.\n",
#endif
               sdl_ctrl->img_hdr->frame_size, props.buffer_size);
        exit(1);
    }*/

    sdl_ctrl->error_frame = 0;

    printf("----bit mapping : %d\n", sdl_ctrl->sdl_opt.bit_mapping[0]);

    if (sdl_ctrl->sdl_opt.bit_mapping[0] != -2)
    {
        // 진입하지 않음. 어느경우에 들어오는 거지?? 
        uint32_t *img_ptr32 = (uint32_t*)img_ptr;

        for (int i=0; i<(int)(frame_size/4); i++)
        {
            img_ptr32[i] = ((((img_ptr32[i] >>  0) & (sdl_ctrl->sdl_opt.bit_mapping[ 0] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 0]) |
                            (((img_ptr32[i] >>  1) & (sdl_ctrl->sdl_opt.bit_mapping[ 1] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 1]) |
                            (((img_ptr32[i] >>  2) & (sdl_ctrl->sdl_opt.bit_mapping[ 2] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 2]) |
                            (((img_ptr32[i] >>  3) & (sdl_ctrl->sdl_opt.bit_mapping[ 3] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 3]) |
                            (((img_ptr32[i] >>  4) & (sdl_ctrl->sdl_opt.bit_mapping[ 4] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 4]) |
                            (((img_ptr32[i] >>  5) & (sdl_ctrl->sdl_opt.bit_mapping[ 5] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 5]) |
                            (((img_ptr32[i] >>  6) & (sdl_ctrl->sdl_opt.bit_mapping[ 6] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 6]) |
                            (((img_ptr32[i] >>  7) & (sdl_ctrl->sdl_opt.bit_mapping[ 7] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 7]) |
                            (((img_ptr32[i] >>  8) & (sdl_ctrl->sdl_opt.bit_mapping[ 8] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 8]) |
                            (((img_ptr32[i] >>  9) & (sdl_ctrl->sdl_opt.bit_mapping[ 9] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[ 9]) |
                            (((img_ptr32[i] >> 10) & (sdl_ctrl->sdl_opt.bit_mapping[10] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[10]) |
                            (((img_ptr32[i] >> 11) & (sdl_ctrl->sdl_opt.bit_mapping[11] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[11]) |
                            (((img_ptr32[i] >> 12) & (sdl_ctrl->sdl_opt.bit_mapping[12] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[12]) |
                            (((img_ptr32[i] >> 13) & (sdl_ctrl->sdl_opt.bit_mapping[13] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[13]) |
                            (((img_ptr32[i] >> 14) & (sdl_ctrl->sdl_opt.bit_mapping[14] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[14]) |
                            (((img_ptr32[i] >> 15) & (sdl_ctrl->sdl_opt.bit_mapping[15] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[15]) |
                            (((img_ptr32[i] >> 16) & (sdl_ctrl->sdl_opt.bit_mapping[16] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[16]) |
                            (((img_ptr32[i] >> 17) & (sdl_ctrl->sdl_opt.bit_mapping[17] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[17]) |
                            (((img_ptr32[i] >> 18) & (sdl_ctrl->sdl_opt.bit_mapping[18] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[18]) |
                            (((img_ptr32[i] >> 19) & (sdl_ctrl->sdl_opt.bit_mapping[19] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[19]) |
                            (((img_ptr32[i] >> 20) & (sdl_ctrl->sdl_opt.bit_mapping[20] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[20]) |
                            (((img_ptr32[i] >> 21) & (sdl_ctrl->sdl_opt.bit_mapping[21] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[21]) |
                            (((img_ptr32[i] >> 22) & (sdl_ctrl->sdl_opt.bit_mapping[22] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[22]) |
                            (((img_ptr32[i] >> 23) & (sdl_ctrl->sdl_opt.bit_mapping[23] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[23]) |
                            (((img_ptr32[i] >> 24) & (sdl_ctrl->sdl_opt.bit_mapping[24] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[24]) |
                            (((img_ptr32[i] >> 25) & (sdl_ctrl->sdl_opt.bit_mapping[25] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[25]) |
                            (((img_ptr32[i] >> 26) & (sdl_ctrl->sdl_opt.bit_mapping[26] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[26]) |
                            (((img_ptr32[i] >> 27) & (sdl_ctrl->sdl_opt.bit_mapping[27] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[27]) |
                            (((img_ptr32[i] >> 28) & (sdl_ctrl->sdl_opt.bit_mapping[28] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[28]) |
                            (((img_ptr32[i] >> 29) & (sdl_ctrl->sdl_opt.bit_mapping[29] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[29]) |
                            (((img_ptr32[i] >> 30) & (sdl_ctrl->sdl_opt.bit_mapping[30] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[30]) |
                            (((img_ptr32[i] >> 31) & (sdl_ctrl->sdl_opt.bit_mapping[11] != -1)) << sdl_ctrl->sdl_opt.bit_mapping[31]));
        }
    }

    printf("--- new frame : %d\n", isNewFrame);
    if (isNewFrame)
    {
        ++(sdl_ctrl->frame_cnt);

        if (sdl_ctrl->sdl_opt.pix_test != 0)
        {
            printf("----pixel test?\n"); // 진입하지 않음. 
            frame_counter = extractFrameCounter((uint16_t*)img_ptr);
            if (frame_counter != frame_counter_last + 1)
            {
                fprintf(stderr,
                        "Invalid frame counters detected in payload: %d %d\n",
                        frame_counter, frame_counter_last);
                sdl_ctrl->error_frame = 1;
            }
            frame_counter_last = frame_counter;

            error_cnt = checkTestData(((uint16_t*)img_ptr + 32),
                                      x_size * y_size,
                                      frame_counter);
            if (error_cnt != 0)
            {
                fprintf(stderr, "bit error detected %d\n", error_cnt);
                sdl_ctrl->error_frame = 1;
            }
        }


    }

    if (sdl_ctrl->sdl_opt.decode_csi2_datatype >= 0)
    {
#if defined(__aarch64__)
        uint8_t    stage[32768];
#endif

        printf("-----csi2 data type : %d\n", sdl_ctrl->sdl_opt.decode_csi2_datatype);

        packet_offset = 0;
        filtered_offset = 0;
        if (isNewFrame)
        {
            align = (bpp == 64) ? 7 : 3;
        }
        // set size and framesize to undefined values if datatype is not found
        x_size = 0;
        y_size = 0;
        frame_size = 0;

        bpp = 16;

        img_hdr_filtered = NULL;
        if ((sdl_ctrl->sdl_opt.write_filtered_images > 0) ||
            (sdl_ctrl->sdl_opt.write_filtered_images == -1))
        {
            printf("--- filter?\n"); // 진입하지 않음.
            img_hdr_filtered =
                (sxpf_image_header_s*)malloc(sdl_ctrl->img_hdr->frame_size *
                                              sizeof(uint8_t));
            memcpy(img_hdr_filtered,
                   sdl_ctrl->img_hdr,
                   sdl_ctrl->img_hdr->payload_offset * sizeof(uint8_t));
            img_ptr_filtered = (uint8_t*)img_hdr_filtered +
                                img_hdr_filtered->payload_offset;

            if (sdl_ctrl->sdl_opt.write_filtered_images > 0)
            {
                sdl_ctrl->sdl_opt.write_filtered_images--;
            }
        }

        for (uint32_t pkt_count = 0;
             pkt_count < sdl_ctrl->img_hdr->rows;
             pkt_count++)
        {
            // printf("---- ???\n"); // ok 여기 진입
            pixels = csi2_parse_dphy_ph(img_ptr + packet_offset,
                                        &vc_dt,
                                        &word_count);
            if (!pixels)
            {
                printf("invalid frame data\n");
                return -1;
            }

            frame_datatypes.insert(vc_dt);

#if defined(__aarch64__)
            uint32_t    ALIGN = 63;
            uintptr_t   base = (uintptr_t)pixels & ALIGN;
            uint32_t    size = (base + word_count + 2 + ALIGN) & ~ALIGN;

            if (size <= sizeof(stage))
            {
                // copy payload + CRC from slow uncached DMA memory to staging
                // buffer on stack
                memcpy(stage, pixels - base, size); // cache-align source & size
                pixels = stage + base;              // redirect to fast memory
            }
#endif

            if (sdl_ctrl->sdl_opt.verify_csi2 != 0)
            {
                // printf("---- verify\n"); // 진입하지 않음
                ecc = *(img_ptr + packet_offset + 4 + 3);
                csi2_update_ph_ECC(img_ptr + packet_offset + 4);
                if ( ecc != *(img_ptr + packet_offset + 4 + 3))
                {
                   sdl_ctrl->error_frame = 1;
                   printf("ecc error detected\n");
                }
                *(img_ptr + packet_offset + 4 + 3) = ecc;
                if ((vc_dt & 0x3f) > 0x0f)
                {
                   crc = csi2_payload_checksum(pixels, word_count);
                   if (crc != *((uint16_t*)(pixels + word_count)))
                   {
                       sdl_ctrl->error_frame = 1;
                       printf("crc error detected\n");
                   }
               }
            }

            // Advance packet_offset to start of next packet:
            if ((vc_dt & 0x3f) <= 0x0f)
            {
                // short packet
                // - 8 bytes at the start make up the short packet data, incl. preamble
                packet_size = (8 + align) & ~align;
            }
            else
            {
                //printf("--- vc dt : %d\n", vc_dt); //여기 진입. 0x1e값이 들어감.
                // - 8 bytes at the start make up the packet header, incl. preamble
                // - word_count payload data bytes follow
                // - Payload data is followed by a 2-byte checksum.
                // The following packet then starts at the next address that is an
                // integer multiple of 4 (if bpp <= 32) or 8 (if bpp == 64).
                packet_size = (8 + word_count + 2 + align) & ~align;
            }

            if (vc_dt == sdl_ctrl->sdl_opt.decode_csi2_datatype)
            {
                //printf("---- ok\n"); // 여기 진입
                uint8_t dt = vc_dt & 0x3f;

                // printf("----dt : %d\n", dt); // 0x1e (30)로 뜸

                if (sdl_ctrl->sdl_opt.process_mode == process_mode_e::swap16)
                {
                    // printf("----1\n");
                   dt = 0x2e;
                }

                if (csi2_decode_datatype(dt,
                                         &bits_per_pixel,
                                         &pixel_group_size))
                {
                    // printf("----2\n");
                    printf("unsupported image type\n");
                    return -1;
                }

                if (dt >= 0x18 && dt <= 0x1f) {
                    // printf("----3\n");
                    sdl_ctrl->sdl_opt.bits_per_component = 0;
                    bits_per_pixel /= 2; // bits_per_pixel value for yuv is not correct here
                }

                if (dt == 0x24)
                {
                    // printf("----4\n");
                    sdl_ctrl->sdl_opt.components_per_pixel  = 3;
                    sdl_ctrl->sdl_opt.bits_per_component    = 8;
                    bpp                                     = 8;
                    bits_per_pixel              /= 3;
                }

                if (tmp2 == NULL)
                {
                    // printf("----5\n");
                    x_size = word_count * 8 / bits_per_pixel;
                    // alloc memory for all rows but only lines with matching
                    // datatype will be used
                    tmp2 = (uint16_t *)malloc(sizeof(uint16_t) *
                                                     x_size *
                                                     sdl_ctrl->img_hdr->rows);

                    pdst = tmp2;
                }

                if ((pdst - tmp2 + x_size) >
                         (x_size * sdl_ctrl->img_hdr->rows))
                {
                    // printf("----6\n");
                    fprintf(stderr, "\ndecoded data exceeds allocated memory\n");
                    break;
                }
                else
                {
                    // printf("----7\n"); // 여기에 진입
                    if (dt == 0x24)
                    {
                        decoded_pix =
                            csi2_decode_raw8((uint8_t *)pdst,
                                             word_count * 8 / bits_per_pixel,
                                             pixels, bits_per_pixel);
                        pdst = (uint16_t*)((uint8_t*)pdst + decoded_pix);
                    }
                    else
                    {
                        // printf("--- ok\n"); // 여기에 진입
                        decoded_pix =
                            csi2_decode_raw16(pdst,
                                              word_count * 8 / bits_per_pixel,
                                              pixels, bits_per_pixel);

                        // printf("--- decoded pix : %d\n", decoded_pix); // 이 값이 3840이 뜸

                        if (sdl_ctrl->sdl_opt.process_mode == process_mode_e::swap16)
                        {
                            // 진입하지 않음
                            for (uint32_t i = 0; i < decoded_pix; i++)
                            {
                                pdst[i] = ((pdst[i] & 0xff) << 8) |
                                          ((pdst[i] & 0xff00) >> 8);
                            }
                        }

                        pdst += decoded_pix;
                    }

                    if (img_hdr_filtered != NULL)
                    {
                        printf("--- img hed filtered is not null\n");
                       memcpy(img_ptr_filtered + filtered_offset,
                              img_ptr + packet_offset, packet_size);
                       filtered_offset += packet_size;
                    }

                    y_size += 1;
                    // printf("--- y size : %d\n", y_size); // 1080까지 증가하다가 다시 1로..
                }
            }
            packet_offset += packet_size;
            //printf("--- packet offset : %d\n", packet_offset); // packet size가 3852로, packet size가 3852씩 증가하다가 4160160까지 올라감
        }

        if (img_hdr_filtered != NULL)
        {
            // printf("--- filtered ok\n"); // 진입하지 않음
            if (y_size > 0)
            {
                // correct img_hdr->rows
                img_hdr_filtered->frame_size =
                            img_hdr_filtered->payload_offset + filtered_offset;
                img_hdr_filtered->rows = y_size;

                char buf[1024];
                sprintf(buf, "filtered_%010d.raw", sdl_ctrl->save_idx);
                sdl_ctrl->increment_save_idx = 1;
                save_file(buf,
                          (uint8_t*)img_hdr_filtered,
                          img_hdr_filtered->frame_size);
            }

            free(img_hdr_filtered);
        }

        img_ptr = (uint8_t*)tmp2;
        frame_size = x_size * y_size * sizeof(uint16_t);

        printf("--- new frame size : %d\n", frame_size); // frame_size가 8294400으로 나오네, 3840x1080x2(yuv422이라 픽셀당 2 byte)

        if (sdl_ctrl->sdl_opt.yuv_in_raw16 != 0)
        {
            printf("--1\n"); // 진입하지 않음
            sdl_ctrl->sdl_opt.bits_per_component = 0;
            bits_per_pixel /= 2; // bits_per_pixel value for yuv is not correct here
        }

        if (sdl_ctrl->sdl_opt.drop_column >= 2)
        {
            printf("--2\n"); //진입하지 않음
            for (uint64_t i = 0;
                 i < frame_size / sizeof(uint16_t) / sdl_ctrl->sdl_opt.drop_column;
                 i++)
            {
                ((uint16_t*)img_ptr)[i] =
                           ((uint16_t*)img_ptr)[i*sdl_ctrl->sdl_opt.drop_column];
            }
            x_size /= sdl_ctrl->sdl_opt.drop_column;
            frame_size /= sdl_ctrl->sdl_opt.drop_column;
        }

        if (x_size != 0 || y_size != 0)
        {
            printf("--3\n"); // 사이즈가 있으니. 진입
            ots = ts;
        }
        else
        {
            frame_rate = -1;
        }
    }
    else
    {
        ots = ts;
    }

    if (sdl_ctrl->increment_save_idx)
    {
        printf("--- save_idx\n"); // 진입하지 않음
        sdl_ctrl->save_idx++;
        sdl_ctrl->increment_save_idx = 0;
    }

    // printf("--- bits per components : %d\n", sdl_ctrl->sdl_opt.bits_per_component ); // bits per component는 0로 뜸
    if (sdl_ctrl->sdl_opt.bits_per_component == 8)
    {
        if (sdl_ctrl->sdl_opt.components_per_pixel == 4)
        {       
            if (bpp == 16)
            {
                
                // use x_size from header as-is, but frame_size is bigger
                frame_size *= 2;
            }
        }
        else
        {
            // packed UYVY: two components in one 16bit
            x_size = x_size * bpp / 8 / sdl_ctrl->sdl_opt.components_per_pixel;
        }
    }
    else if (sdl_ctrl->sdl_opt.bits_per_component == 12 ||
             sdl_ctrl->sdl_opt.bits_per_component == 16)
    {
        // truncate only if sdl is used
        x_size = (x_size * bpp /
                   sdl_ctrl->sdl_opt.bits_per_component) &
                 (sdl_ctrl->sdl_opt.show_sdl ? ~1 : ~0);
        //if (x_size_orig != x_size && isNewFrame)
        //    printf("\nImage width truncated for opengl. Original size was %dx%d\n", x_size_orig, y_size);
        frame_size = x_size * y_size * sizeof(uint16_t);
    }
    else if (sdl_ctrl->sdl_opt.bits_per_component == 10 ||
             sdl_ctrl->sdl_opt.bits_per_component == 14)
    {
        x_size = (x_size * bpp /
                  sdl_ctrl->sdl_opt.bits_per_component) & ~3;
        frame_size = x_size * y_size * sizeof(uint16_t);
    }
    else
    {
        x_size /= sdl_ctrl->sdl_opt.components_per_pixel;
        printf("-- component per pixel : %d, x size : %d\n", sdl_ctrl->sdl_opt.components_per_pixel, x_size); // component per pixel은 2, x_size는 1920으로.. 
    }

    // printf("--- bits per component : %d\n", bits_per_component); // 0
    if (sdl_ctrl->sdl_opt.is_csi2)
    {
        // 진입하지 않음
        uint32_t   src_line_bytes = sdl_ctrl->img_hdr->columns *
                                    bpp / 8;
#if defined(__aarch64__)
        uint32_t    ALIGN = 63;
        uintptr_t   base = (uintptr_t)img_ptr & ALIGN;
        uint32_t    size = (base + y_size * src_line_bytes + ALIGN) & ~ALIGN;

        // copy raw data from slow uncached DMA memory to staging buffer on heap
        tmp2 = (uint16_t*)malloc(size);
        memcpy(tmp2, img_ptr - base, size); // cache-align source & size
        img_ptr = (uint8_t*)tmp2 + base;    // redirect to fast memory
#endif

        tmp = (uint32_t*)malloc(frame_size);
        if (!tmp)
            return -1;

        switch (sdl_ctrl->sdl_opt.bits_per_component)
        {
        default:
            return -1;  // invalid pixel format

        case 10:
        case 12:
        case 14:
        case 16:
            for (uint32_t y = 0; y < y_size; y++)
            {
                uint16_t  *dst = (uint16_t*)tmp + y * x_size;
                uint8_t   *src = img_ptr + y * src_line_bytes;

                csi2_decode_raw16_msb(dst, x_size, src,
                                      sdl_ctrl->sdl_opt.bits_per_component);

                if (sdl_ctrl->sdl_opt.left_shift > 0)
                {
                    for (uint32_t x=0; x < x_size; x++)
                    {
                        dst[x] = dst[x] << sdl_ctrl->sdl_opt.left_shift;
                    }
                }
            }
            break;
        }
        img_ptr = (uint8_t*)tmp;
        bits_per_component = 16;
        x_size /= sdl_ctrl->sdl_opt.components_per_pixel;
    }
    else if (sdl_ctrl->sdl_opt.bits_per_component == 12)
    {
        imx290_field_t  fields[3];      // decoded frame information
        uint32_t        n_fields;       // number of present fields in frame

        uint32_t    lines = analyze_dol_frame_12b(img_ptr, x_size, y_size,
                                                  fields, &n_fields);
        uint32_t    out_lines = 0;

        if (lines > 0)
        {
            lines = 1080;

            // packed 12 bit-per-pixel mode with interleaved lines of two
            // exposures
            out_lines = (sdl_ctrl->field_sel < 3) ? lines : n_fields * lines;

            tmp = (uint32_t*)malloc(1920 * out_lines * sizeof(uint16_t));
        }

        if(tmp)
        {
            int16_t     *out_16 = (int16_t*)tmp;
            uint32_t    start_line = 0;
            double      field_scales[3] = { 4.0, 16.0, 0. };

            if (sdl_ctrl->field_sel != 2)
            {
                for (uint32_t f = 0; f < n_fields; f++)
                {
                    if (sdl_ctrl->field_sel == f ||
                        sdl_ctrl->field_sel >= n_fields)
                    {
                        // copy f to output
                        decode_dol_frame_12b(out_16 + start_line * 1920,
                                             1920, 1080,
                                             fields[f].src, fields[f].pitch,
                                             x_size, y_size, 0x0f0,//def. blklvl
                                             4 /* frame info */ + 4 + 8, 8,
                                             field_scales[f]);
                        start_line += 1080;
                    }
                }
            }
            else
            {
                combine_dol_frames_12b(out_16, 1920, 1080, fields, field_scales,
                                       x_size, y_size, 0x0f0/*blk lvl*/, 16, 8);
            }

            msglen = sprintf(msg, "YS=%u (%u, %u), VBP1=%u",
                             y_size, fields[0].lines, fields[1].lines,
                             fields[1].vbp);

            x_size = 1920;
            y_size = out_lines;

            img_ptr = (uint8_t*)out_16;

            if (sdl_ctrl->doSaveImage)
                save_file("imgdecode.raw", img_ptr, x_size * out_lines * 2);
        }
    }
    else if (bits_per_component == 8 && sdl_ctrl->sdl_opt.components_per_pixel == 4)
    {
        if (sdl_ctrl->sdl_opt.is_oldi)
        {
            // 8bit RGBA mode with alpha=0
            // pre-process image data
            if (!strcmp(sdl_ctrl->sdl_opt.shader, "oldi"))
                tmp = (uint32_t*)malloc(frame_size);

            // perform OpenLDI bit-shuffling
            if (tmp)
            {
                uint32_t    i;
                uint32_t    *src = (uint32_t*)img_ptr;

                for (i = 0; i < frame_size / sizeof(uint32_t); i++)
                {
                    sum |= src[i];  // find out which data bits are active

                    uint32_t v = src[i];
                    uint32_t r = ((v & 0x00003f) << 0) | ((v & 0x0c0000) >> 12);
                    uint32_t g = ((v & 0x000fc0) << 2) | ((v & 0x300000) >> 6);
                    uint32_t b = ((v & 0x03f000) << 4) | ((v & 0xc00000) >> 0);

                    tmp[i] = r | g | b;
                }

                img_ptr = (uint8_t*)tmp;

                msglen = sprintf(msg, "sum = 0x%08x", sum);
            }
        }
    }
    else if (bits_per_component == 16 && sdl_ctrl->sdl_opt.components_per_pixel == 1)
    {
        // RAW mode with 16bits per pixel
        // -> convert to 8bpp for our GPU debayer filter to work
        tmp = (uint32_t*)malloc(frame_size);
        bits_per_component = 8;

        uint32_t    i;
        uint32_t    *src = (uint32_t*)img_ptr;
        uint16_t    *pdst = (uint16_t*)tmp;
        uint32_t    mask = 0x0000ffff;

        // remove bits that get "shifted out"
        mask = mask >> sdl_ctrl->sdl_opt.left_shift;
        mask = mask | (mask << 16); // mask two component values in one go

        for (i = 0; i < frame_size / sizeof(uint32_t); i++)
        {
            sum |= src[i];  // find out which data bits are active

            // convert two pixels in one go
            uint32_t two_pix = (src[i] & mask) << sdl_ctrl->sdl_opt.left_shift;

            *pdst++ = ((two_pix >> 8) & 0x00ff) | ((two_pix >> 16) & 0xff00);
        }

        img_ptr = (uint8_t*)tmp;

        msglen = sprintf(msg, "sum = 0x%08x", sum);
    }
    else if (sdl_ctrl->sdl_opt.left_shift != 0
                 /* TODO || sdl_ctrl->de_compand... */)
    {
        // pre-process image data
        tmp = (uint32_t*)malloc(frame_size);
        if (tmp)
        {
            uint32_t    i;
            uint32_t    *src = (uint32_t*)img_ptr;
            uint32_t    mask = 0x0000ffff;

            // remove bits that get "shifted out"
            mask = mask >> sdl_ctrl->sdl_opt.left_shift;
            mask = mask | (mask << 16); // mask two component values in one go

            for (i = 0; i < frame_size / sizeof(uint32_t); i++)
            {
                sum |= src[i];  // find out which data bits are active

                // shift 2 pixels in one go
                tmp[i] = (src[i] & mask) << sdl_ctrl->sdl_opt.left_shift;
            }

            img_ptr = (uint8_t*)tmp;

            msglen = sprintf(msg, "sum = 0x%08x", sum);
        }
    }

    if (isNewFrame || sdl_ctrl->sdl_opt.dbg_image)
    {
        // 여기에 진입 new frame이기 때문.. 
        if (x_size != sdl_ctrl->frame_x_size ||
            y_size != sdl_ctrl->frame_y_size ||
            (sdl_ctrl->frame_info & 0x00ffffff) !=
            (sdl_ctrl->new_frame_info & 0x00ffffff) ||
            (sdl_ctrl->frame_cnt > 2 && round(frame_rate * 100) !=
             round(sdl_ctrl->fps * 100))
           )
        {
            // print on new line so we can track resolution changes
            fprintf(stderr, "\n");
        }

        sdl_ctrl->frame_x_size     = x_size;
        sdl_ctrl->frame_y_size     = y_size;
        sdl_ctrl->fps        = frame_rate;
        sdl_ctrl->frame_info = sdl_ctrl->new_frame_info;

        rawtime = time(NULL);
        time_str = ctime(&rawtime);
        time_str[strlen(time_str)-1] = '\0';

        int len = 0;
        len += sprintf(sdl_ctrl->output_str + len,
                "%s #%u: card=%d ch=%d  vs=%u, %dx%d, cap=%.2ffps, vis=%.2ffps",
                time_str,
                sdl_ctrl->frame_cnt,
                sdl_ctrl->img_hdr->card_id,
                sdl_ctrl->img_hdr->cam_id,
                sdl_ctrl->new_frame_info & 0x00ffffff,
                x_size,
                y_size,
                capture_fps,
                frame_rate);

        if (sdl_ctrl->sdl_opt.decode_csi2_datatype > 0)
        {
            len += sprintf(sdl_ctrl->output_str + len,
                    ", decoded dt=0x%02x",
                    sdl_ctrl->sdl_opt.decode_csi2_datatype);
            frame_datatypes.erase(sdl_ctrl->sdl_opt.decode_csi2_datatype);
            if (frame_datatypes.size() > 0)
            {
                len += sprintf(sdl_ctrl->output_str + len, ", other dt's ");
                auto p = frame_datatypes.begin();
                len += sprintf(sdl_ctrl->output_str + len, "0x%02x", *p++);
                while (p != frame_datatypes.end())
                {
                    len += sprintf(sdl_ctrl->output_str + len, ", 0x%02x", *p++);
                }
            }
        }

        if (sdl_ctrl->sdl_opt.show_timestamp != 0)
        {
            len += sprintf(sdl_ctrl->output_str + len, ", "
                    "start=0x%016" PRIx64 ", end=0x%016" PRIx64,
                    ts, ts_end);
        }

        if (msglen > 0)
        {
            len += sprintf(sdl_ctrl->output_str + len, ", %.*s", msglen, msg);
        }
    }


    if (sdl_ctrl->sdl_opt.show_sdl != 0 &&
        sdl_ctrl->frame_x_size != 0 &&
        sdl_ctrl->frame_y_size != 0)
    {
        GLint TEX_WIDTH = (sdl_ctrl->frame_x_size + 3) & ~3;
        GLint TEX_HEIGHT = (sdl_ctrl->frame_y_size + 3) & ~3;

        if (TEX_WIDTH != TEX_WIDTH_OLD ||
            TEX_HEIGHT != TEX_HEIGHT_OLD)
        {
            TEX_WIDTH_OLD = TEX_WIDTH;
            TEX_HEIGHT_OLD = TEX_HEIGHT;

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

            glBindTexture(GL_TEXTURE_2D, sdl_ctrl->gl_texture);
            check_gl_error("glBindTexture");
            //glPixelStorei(GL_UNPACK_ALIGNMENT, 4);  // 4-byte pixel alignment
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,
                            GL_TEXTURE_WRAP_S,
                            GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D,
                            GL_TEXTURE_WRAP_T,
                            GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
            check_gl_error("glTexParameteri(GL_TEXTURE_2D, "
                           "GL_TEXTURE_MAX_LEVEL, 0)");
#if 0
            // since OpenGL 4.2
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, TEX_WIDTH, TEX_HEIGHT);
            check_gl_error("glTexStorage2D");
#else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEX_WIDTH, TEX_HEIGHT, 0,
                         GL_RG, GL_UNSIGNED_SHORT, NULL);
            check_gl_error("glTexImage2D");
#endif
            glBindTexture(GL_TEXTURE_2D, 0);

            if (sdl_ctrl->pbo[0] == 0)
            {
                glGenBuffers(2, sdl_ctrl->pbo);
            }
            sdl_ctrl->num_pbos = 0;
            if (!check_gl_error("glGenBuffers(2, sdl_ctrl->pbo)"))
            {
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sdl_ctrl->pbo[0]);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, TEX_WIDTH * TEX_HEIGHT *
                             2, 0, GL_STREAM_DRAW);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sdl_ctrl->pbo[1]);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, TEX_WIDTH * TEX_HEIGHT *
                             2, 0, GL_STREAM_DRAW);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                if (!check_gl_error("glBufferData"))
                {
                    sdl_ctrl->num_pbos = 2;
                    sdl_ctrl->front_pbo = 0;
                    sdl_ctrl->used_pbos = 0;
                }
            }
        }

        GLenum type = GL_UNSIGNED_SHORT;
        GLenum format = GL_RG;

        if (bits_per_component == 8)
        {
            type = GL_UNSIGNED_BYTE;
        }

        if (sdl_ctrl->sdl_opt.components_per_pixel == 1)
        {
            format = GL_RED;
        }
        else if (sdl_ctrl->sdl_opt.components_per_pixel == 3)
        {
            format = GL_RGB;
        }
        else if (sdl_ctrl->sdl_opt.components_per_pixel == 4)
        {
            format = GL_RGBA;
        }

#if 0
        GLuint clearColor[4] = { 0, 255, 0, 0 };
        glClearTexImage(ch->texture, 0, GL_BGRA, GL_UNSIGNED_BYTE, &clearColor);
#endif

        if (sdl_ctrl->used_pbos == 0)
        {
            // direct upload from DMA memory using glTexSubImage2D()
            //printf("texture=%d  ", sdl_ctrl->texture);
            glBindTexture(GL_TEXTURE_2D, sdl_ctrl->gl_texture);
            check_gl_error("glBindTexture");
            glTexSubImage2D(GL_TEXTURE_2D, 0,   // level 0
                0, 0,               // x-offset, y-offset
                sdl_ctrl->frame_x_size, sdl_ctrl->frame_y_size,
                format, type, img_ptr);
            check_gl_error("glTexSubImage2D");
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        else
        {
            GLubyte *pbuf;

            if (sdl_ctrl->used_pbos == 2)
            {
                back_pbo = 1 - sdl_ctrl->front_pbo;
            }
            else
            {
                back_pbo = sdl_ctrl->front_pbo;
            }

            // copy pixels from front PBO to texture
            glBindTexture(GL_TEXTURE_2D, sdl_ctrl->gl_texture);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER,
                         sdl_ctrl->pbo[sdl_ctrl->front_pbo]);

            glTexSubImage2D(GL_TEXTURE_2D, 0,   // level 0
                0, 0,               // x-offset, y-offset
                sdl_ctrl->frame_x_size, sdl_ctrl->frame_y_size,
                format, type, 0);

            // upload new video data to back PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sdl_ctrl->pbo[back_pbo]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER,
                         sdl_ctrl->frame_x_size * sdl_ctrl->frame_y_size * 2,
                         0,
                         GL_STREAM_DRAW);
            pbuf = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
            if (pbuf)
            {
                memcpy(pbuf,
                       img_ptr,
                       sdl_ctrl->frame_x_size * sdl_ctrl->frame_y_size * 2);
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
            }
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // switch buffers
            sdl_ctrl->front_pbo = back_pbo;
        }
    }




    if (tmp)
        free(tmp);
    if (tmp2)
        free(tmp2);

    return ret;
}


void SdlRenderTexture(sdl_ctrl_t* sdl_ctrl)
{
    GLint TEX_WIDTH = (sdl_ctrl->frame_x_size + 3) & ~3;
    GLint TEX_HEIGHT = (sdl_ctrl->frame_y_size + 3) & ~3;
    float obj_aspect = (float)TEX_WIDTH / TEX_HEIGHT;

    if (sdl_ctrl->sdl_opt.aspect != 0.0)
        obj_aspect = sdl_ctrl->sdl_opt.aspect;

    glClearColor(((sdl_ctrl->sdl_opt.background >> 16) & 0xff) / 255.f,
                 ((sdl_ctrl->sdl_opt.background >>  8) & 0xff) / 255.f,
                 ((sdl_ctrl->sdl_opt.background)       & 0xff) / 255.f,
                 0);              // background color
    glClear(GL_COLOR_BUFFER_BIT);

    if (sdl_ctrl->renderer)
    {
        GLuint glShader = sdl_ctrl->renderer->shader();

        if (glShader)
        {
            // select shader to use
            glUseProgram(glShader);

            if (sdl_ctrl->gl_shader_sourceSize >= 0)
                glUniform4f(sdl_ctrl->gl_shader_sourceSize,
                            1.f * TEX_WIDTH, 1.f * TEX_HEIGHT,
                            1.f / TEX_WIDTH, 1.f / TEX_HEIGHT);

            if (sdl_ctrl->gl_shader_firstRed >= 0)
                glUniform2f(sdl_ctrl->gl_shader_firstRed,
                            sdl_ctrl->sdl_opt.bayer_pattern & 2 ? 1.f : 0.f,
                            sdl_ctrl->sdl_opt.bayer_pattern & 1 ? 1.f : 0.f);

            if (sdl_ctrl->gl_shader_gamma >= 0)
                glUniform3f(sdl_ctrl->gl_shader_gamma,
                            1.f / sdl_ctrl->sdl_opt.gl_gamma,
                            1.f / sdl_ctrl->sdl_opt.gl_gamma,
                            1.f / sdl_ctrl->sdl_opt.gl_gamma);

            if (sdl_ctrl->gl_shader_saturation >= 0)
                glUniform1f(sdl_ctrl->gl_shader_saturation,
                            sdl_ctrl->sdl_opt.gl_saturation);
        }


        float win_aspect =
               (float) sdl_ctrl->sdl_window_width / sdl_ctrl->sdl_window_height;
        if (sdl_ctrl->sdl_opt.vflip)
        {
            win_aspect *= -1.f;
        }
        if (sdl_ctrl->sdl_opt.hflip)
        {
            obj_aspect *= -1.f;
        }

        clear_gl_error();

        if (sdl_ctrl->enable_two_phase_rendering)
        {
            sdl_ctrl->backbuffer.resize(sdl_ctrl->frame_x_size,
                                          sdl_ctrl->frame_y_size,
                                          5);

            sdl_ctrl->backbuffer.bind(false);

            glViewport(0, 0, sdl_ctrl->frame_x_size, sdl_ctrl->frame_y_size);

            sdl_ctrl->renderer->render(sdl_ctrl->gl_texture,
                                                 1.f,
                                                 -1.f,
                                                 0.f,
                                                 0.f,
                                                 1.f,
                                                 1.f);

            sdl_ctrl->backbuffer.unbind();

            glViewport(0,
                       0,
                       sdl_ctrl->sdl_window_width,
                       sdl_ctrl->sdl_window_height);

            sdl_ctrl->rgb_renderer->render(sdl_ctrl->backbuffer.texture(),
                                           win_aspect,
                                           -0.5f * (sdl_ctrl->sdl_opt.gl_zoom - 1.0f),
                                           -0.5f * (sdl_ctrl->sdl_opt.gl_zoom - 1.0f),
                                           1.f + (sdl_ctrl->sdl_opt.gl_zoom - 1.0f),
                                           1.f + (sdl_ctrl->sdl_opt.gl_zoom - 1.0f));
            if (get_gl_error())
            {
                printf("Missing OpenGL features. Disabling two-phase rendering. "
                       "You might get aliasing artifacts\n");
                sdl_ctrl->enable_two_phase_rendering = 0;
            }

            if (!sdl_ctrl->enable_two_phase_rendering)
            {
                sdl_ctrl->renderer->render(sdl_ctrl->gl_texture,
                                           obj_aspect,
                                           win_aspect,
                                           -0.5f * (sdl_ctrl->sdl_opt.gl_zoom - 1.0f),
                                           -0.5f * (sdl_ctrl->sdl_opt.gl_zoom - 1.0f),
                                           1.f + (sdl_ctrl->sdl_opt.gl_zoom - 1.0f),
                                           1.f + (sdl_ctrl->sdl_opt.gl_zoom - 1.0f));
            }
        }
    }

    if (sdl_ctrl->new_frame_info != -1 || sdl_ctrl->sdl_opt.dbg_image)
    {
        if ((sdl_ctrl->doSaveOutput != 0) ||
            (sdl_ctrl->sdl_opt.write_output_images > 0) ||
            (sdl_ctrl->sdl_opt.write_output_images == -1))
        {
            if(sdl_ctrl->output_image_buffer == 0)
            {
                int width = sdl_ctrl->sdl_window_width;
                int height = sdl_ctrl->sdl_window_height;
                int size = sizeof(uint8_t) * width * height * 4;
                uint8_t* buffer = (uint8_t*)malloc(size);

                glReadnPixels(0, 0,
                              width, height,
                              GL_RGBA, GL_UNSIGNED_BYTE, size, buffer);

                sdl_ctrl->output_image_buffer = buffer;
            }
        }
    }

    SDL_GL_SwapWindow(sdl_ctrl->sdl_window);
}

void SdlPollEvents(sdl_ctrl_t* sdl_ctrl)
{
    while (SDL_PollEvent(&sdl_ctrl->sdl_event))
    {
        switch (sdl_ctrl->sdl_event.type)
        {
        case SDL_KEYUP:
            switch (sdl_ctrl->sdl_event.key.keysym.sym)
            {
            case SDLK_o:
                sdl_ctrl->doSaveOutput = 0;
                break;
            case SDLK_r:
                sdl_ctrl->doSaveImage = 0;
                break;
            }
            break;

        case SDL_KEYDOWN:
            switch (sdl_ctrl->sdl_event.key.keysym.sym)
            {
            case SDLK_1:
            case SDLK_2:
            case SDLK_3:
            case SDLK_4:
                sdl_ctrl->field_sel =
                                 sdl_ctrl->sdl_event.key.keysym.sym - SDLK_1;
                break;

            case SDLK_ESCAPE:
                sdl_ctrl->sdl_done = SDL_TRUE;
                break;

            case SDLK_f:
                sdl_ctrl->is_fullscreen = !sdl_ctrl->is_fullscreen;
                SDL_SetWindowFullscreen(sdl_ctrl->sdl_window,
                                        sdl_ctrl->is_fullscreen
                                          ? SDL_WINDOW_FULLSCREEN_DESKTOP
                                          : SDL_WINDOW_RESIZABLE);
                break;

            case SDLK_b:
                if (++sdl_ctrl->used_pbos > sdl_ctrl->num_pbos)
                    sdl_ctrl->used_pbos = sdl_ctrl->front_pbo = 0;
                fprintf(stderr, "%d PBOs in use   ", sdl_ctrl->used_pbos);
                break;

            case SDLK_d:
            {
                int dir = (sdl_ctrl->sdl_event.key.keysym.mod & KMOD_SHIFT) ?
                              -1 : 1;

                sdl_ctrl->field_sel = (sdl_ctrl->field_sel + dir) % 4;
                break;
            }

            case SDLK_g:
            {
                float dir =
                    (sdl_ctrl->sdl_event.key.keysym.mod & KMOD_SHIFT) ?
                       -0.1f : 0.1f;

                sdl_ctrl->sdl_opt.gl_gamma += dir;
                if (sdl_ctrl->sdl_opt.gl_gamma < 0.5)
                    sdl_ctrl->sdl_opt.gl_gamma = 0.5;
                if (sdl_ctrl->sdl_opt.gl_gamma > 2)
                    sdl_ctrl->sdl_opt.gl_gamma = 2;

                printf("gamma = %.2f\n", sdl_ctrl->sdl_opt.gl_gamma);
                break;
            }

            case SDLK_o:
                sdl_ctrl->doSaveOutput = 1;
                break;

            case SDLK_r:
                sdl_ctrl->doSaveImage = 1;
                break;

            case SDLK_s:
            {
                float dir =
                    (sdl_ctrl->sdl_event.key.keysym.mod & KMOD_SHIFT) ?
                       -0.1f : 0.1f;

                sdl_ctrl->sdl_opt.gl_saturation += dir;
                if (sdl_ctrl->sdl_opt.gl_saturation < 0)
                    sdl_ctrl->sdl_opt.gl_saturation = 0;
                if (sdl_ctrl->sdl_opt.gl_saturation > 2)
                    sdl_ctrl->sdl_opt.gl_saturation = 2;

                printf("saturation = %.2f\n", sdl_ctrl->sdl_opt.gl_saturation);
                break;
            }

            case SDLK_v:
                sdl_ctrl->swap_interval = 1 - sdl_ctrl->swap_interval;
                SDL_GL_SetSwapInterval(sdl_ctrl->swap_interval);

                printf("Sync to VSYNC %s\n",
                       sdl_ctrl->swap_interval ? "on" : "off");
                break;

            case SDLK_z:
            {
                float dir =
                    (sdl_ctrl->sdl_event.key.keysym.mod & KMOD_SHIFT) ?
                       -0.1f : 0.1f;

                sdl_ctrl->sdl_opt.gl_zoom += dir;
                if (sdl_ctrl->sdl_opt.gl_zoom < 1.0f)
                    sdl_ctrl->sdl_opt.gl_zoom = 1.0f;
                if (sdl_ctrl->sdl_opt.gl_zoom > 6.0f)
                    sdl_ctrl->sdl_opt.gl_zoom = 6.0f;

                printf("zoom = %.2f\n", sdl_ctrl->sdl_opt.gl_zoom);
                break;
            }


            case SDLK_PAUSE:
                sdl_ctrl->is_pause = !sdl_ctrl->is_pause;
                printf("\n%s.\n",
                       sdl_ctrl->is_pause ? "PAUSED" : "CONTINUING");
                break;
            }
            break;

        case SDL_QUIT:
            sdl_ctrl->sdl_done = SDL_TRUE;
            break;

        case SDL_WINDOWEVENT:
            switch (sdl_ctrl->sdl_event.window.event)
            {
            case SDL_WINDOWEVENT_RESIZED:
            {
                if (sdl_ctrl->sdl_window_width !=
                       sdl_ctrl->sdl_event.window.data1 ||
                    sdl_ctrl->sdl_window_height !=
                       sdl_ctrl->sdl_event.window.data2)
                {
                    sdl_ctrl->sdl_window_width =
                                     sdl_ctrl->sdl_event.window.data1;
                    sdl_ctrl->sdl_window_height =
                                     sdl_ctrl->sdl_event.window.data2;

                    fprintf(stderr, "\nresized to %dx%d\n",
                            sdl_ctrl->sdl_window_width,
                            sdl_ctrl->sdl_window_height);
                }
                break;
            }
            }
        }
    }
}
