
#include "video.file.grabber.hpp"
#include <flame/log.hpp>
#include <chrono>

using namespace flame;
using namespace std;
using namespace cv;


/* create component instance */
static video_file_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new video_file_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool video_file_grabber::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* check parameters */
        if(!parameters.contains("camera") || !parameters["camera"].is_array()){
            logger::warn("[{}] Not found or Invalid 'camera' parameters. It must be valid.", get_name());
            return false;
        }
        logger::info("[{}] {} camera parameter is defined.", get_name(), parameters["camera"].size());

        /* setup pipeline */
        _use_image_stream.store(parameters.value("use_image_stream", false));
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));

        /* get video file path from first camera entry */
        json camera_parameters = parameters["camera"];
        if(camera_parameters.is_array() && !camera_parameters.empty()){
            string video_file = camera_parameters[0].value("file", "");
            if(video_file.empty()){
                logger::error("[{}] No video file specified in 'file' parameter", get_name());
                return false;
            }

            /* create video capture */
            _video_capture = make_unique<cv::VideoCapture>(video_file);

            if(!_video_capture->isOpened()){
                logger::error("[{}] Failed to open video file: {}", get_name(), video_file);
                return false;
            }

            logger::info("[{}] Video file opened successfully: {}", get_name(), video_file);
            logger::info("[{}] Video properties - Width: {}, Height: {}, FPS: {}, Total Frames: {}",
                get_name(),
                (int)_video_capture->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)_video_capture->get(cv::CAP_PROP_FRAME_HEIGHT),
                _video_capture->get(cv::CAP_PROP_FPS),
                (int)_video_capture->get(cv::CAP_PROP_FRAME_COUNT)
            );

            /* start grab worker */
            _grab_worker = thread(&video_file_grabber::_grab_task, this, camera_parameters);
        }
        else{
            logger::error("[{}] Camera parameters array is empty", get_name());
            return false;
        }

    }
    catch(json::exception& e){
        logger::error("Profile Error : {}", e.what());
        return false;
    }
    catch(cv::Exception& e){
        logger::error("OpenCV Error : {}", e.what());
        return false;
    }

    return true;
}

void video_file_grabber::on_loop(){
    /* nothing loop */
}


void video_file_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop grabbing thread */
    if(_grab_worker.joinable()){
        _grab_worker.join();
        logger::debug("[{}] grabber is now successfully stopped", get_name());
    }

    /* close video capture */
    if(_video_capture && _video_capture->isOpened()){
        _video_capture->release();
    }

}

void video_file_grabber::on_message(const message_t& msg){
    /* reserved function */
}

void video_file_grabber::_grab_task(json camera_parameters){

    /* get video file fps for frame rate control */
    double video_fps = _video_capture->get(cv::CAP_PROP_FPS);
    if(video_fps <= 0) video_fps = 30.0; // default to 30fps if not available

    auto frame_duration = chrono::milliseconds((int)(1000.0 / video_fps));

    while(!_worker_stop.load()){

        auto frame_start = chrono::high_resolution_clock::now();

        /* do grab */
        try{
            cv::Mat captured;

            if(!_video_capture->read(captured)){
                logger::info("[{}] End of video file reached, restarting from beginning", get_name());
                _video_capture->set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }

            // /* calc capture time */
            // auto now = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsed = now - last_time;
            // last_time = now;

            if (!captured.empty()) {
                logger::debug("[{}] Captured image: {}x{}, channels: {}", get_name(), captured.cols, captured.rows, captured.channels());

                /* generate meta tag */
                json tag;
                tag["id"] = camera_parameters[0].value("id", 1);
                tag["fps"] = video_fps;
                tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(frame_start.time_since_epoch()).count();

                /* image rotate (0=cw_90, 1=180, 2=ccw_90)*/
                int rotate_flag = -1;
                string portname = "";
                if(camera_parameters.is_array() && !camera_parameters.empty()){
                    rotate_flag = camera_parameters[0].value("rotate_flag", -1);
                    portname = camera_parameters[0].value("portname", "image_stream_1");
                }

                if(rotate_flag >= 0 && rotate_flag <= 2){
                    cv::rotate(captured, captured, rotate_flag);
                }

                /* push image */
                if(_use_image_stream.load()){

                    /* image encoding */
                    std::vector<unsigned char> serialized_image;
                    cv::imencode(".jpg", captured, serialized_image);

                    /* update tag */
                    tag["height"] = captured.rows;
                    tag["width"] = captured.cols;

                    /* send data */
                    if(get_port(portname.c_str())->handle()!=nullptr){
                        message_t msg_multipart;
                        msg_multipart.addstr(portname);
                        msg_multipart.addstr(tag.dump());
                        msg_multipart.addmem(serialized_image.data(), serialized_image.size());
                        if(!msg_multipart.send(*get_port(portname.c_str()), ZMQ_DONTWAIT)){
                            logger::warn("[{}] Failed to send message, queue may be full", get_name());
                        }
                        msg_multipart.clear(); 
                    }
                    else{
                        logger::warn("[{}] socket handle is not valid ", get_name());
                    }
                    serialized_image.clear();
                }

                /* publish monitoring image */
                if(_use_image_stream_monitoring.load()){

                    /* resize image */
                    cv::Mat resized;
                    int target_width = 540;
                    int target_height = 960;

                    cv::resize(captured, resized, cv::Size(target_width, target_height));

                    /* image encoding */
                    std::vector<unsigned char> serialized_monitoring_image;
                    cv::imencode(".jpg", resized, serialized_monitoring_image);

                    /* update tag */
                    tag["height"] = resized.rows;
                    tag["width"] = resized.cols;


                    /* send monitoring data */
                    string monitor_portname = fmt::format("{}_monitor", portname);
                    if(get_port(monitor_portname.c_str())->handle()!=nullptr){
                        message_t msg_multipart;
                        msg_multipart.addstr(monitor_portname);
                        msg_multipart.addstr(tag.dump());
                        msg_multipart.addmem(serialized_monitoring_image.data(), serialized_monitoring_image.size());
                        if(!msg_multipart.send(*get_port(monitor_portname.c_str()), ZMQ_DONTWAIT)){
                            logger::warn("[{}] Failed to send message, queue may be full", get_name());
                        }
                        msg_multipart.clear(); 
                    }
                    serialized_monitoring_image.clear();
                }

            }
        }
        catch(const cv::Exception& e){
            logger::debug("[{}] CV Exception {}", get_name(), e.what());
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] Pipeline Error : {}", get_name(), e.what());
        }
        catch(const json::exception& e){
            logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
        }
        catch(const std::exception& e){
            logger::error("[{}] Standard Exception : {}", get_name(), e.what());
        }

        /* frame rate control - sleep to maintain video fps */
        auto frame_end = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(frame_end - frame_start);

        if(elapsed < frame_duration){
            this_thread::sleep_for(frame_duration - elapsed);
        }

    }

    logger::debug("[{}] Stopped grab task..", get_name());

}
