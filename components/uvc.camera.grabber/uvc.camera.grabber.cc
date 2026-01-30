
#include "uvc.camera.grabber.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>

using namespace flame;
using namespace std;
using namespace cv;

/* create component instance */
static uvc_camera_grabber* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new uvc_camera_grabber(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool uvc_camera_grabber::on_init(){

    try{
        /* read profile */
        json parameters = get_profile()->parameters();

        /* init zpipe */
        _pipe = flame::pipe::create_pipe(1);

        /* set video capture instance */
        int auto_id = 1;
        if(parameters.contains("camera")){
            for(auto& dev:parameters["camera"]){
                int id = dev.value("id", auto_id++);
                dev["id"] = id; /* update camera id */

                /* assign grabber worker */
                _grab_worker[id] = thread(&uvc_camera_grabber::_grab_task, this, id, dev);        
            }
        }
        else {
            logger::warn("[{}] Cannot found camera(s) available", get_name());
            return false;
        }

        /* configure data for data pipelining  */
        _use_image_stream_monitoring.store(parameters.value("use_image_stream_monitoring", false));
        _use_image_stream.store(parameters.value("use_image_stream", false));
        _rotation_cw.store(parameters.value("rotation_cw", 0.0));
        _worker_stop.store(false);

        /* load calibration data */
        if(parameters.contains("calibration")){
            json calib = parameters["calibration"];
            if(calib.contains("focal_length") && calib.contains("principal_point") && calib.contains("distortion")){
                vector<double> f = calib["focal_length"];
                vector<double> c = calib["principal_point"];
                vector<double> d = calib["distortion"];
                
                if(f.size()>=2 && c.size()>=2 && d.size()>=4){
                    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
                    K.at<double>(0,0) = f[0]; K.at<double>(1,1) = f[1];
                    K.at<double>(0,2) = c[0]; K.at<double>(1,2) = c[1];
                    
                    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F); // Assume at least 4 coeffs, support up to 5 standard
                    for(size_t i=0; i<d.size() && i<5; ++i) D.at<double>(0,i) = d[i];

                    /* pre-compute undistort maps (will initialize size later once resolution is known or assume port resolution) */
                    // Wait.. we need image resolution to init maps. 
                    // Since resolution might depend on camera, we can defer map init or do it if we know resolution.
                    // Let's assume resolution from dataport or first frame. 
                    // Better approach: In _grab_task, check if maps are empty and init them once.
                    // So here we store K and D, or just rely on parameters access in _grab_task? 
                    // Storing K and D in member might be cleaner but thread safety?
                    // Let's parse here but compute maps in _grab_task to be safe with image size.
                    // Actually, simpler: Just enable flags here and parse in task or helper.
                    // Given the structure, parsing in on_init and storing locally to transfer to task or member variables is best.
                    // Let's just set the flag here and parse K/D inside the task or valid place.
                    // Actually, K and D are specific to camera. If multiple cameras, we need a map of K/D.
                    // The current json has "calibration" at root, implying one calibration for the component (likely single camera case).
                    // We will parse it into members (protected by mutex if needed or just read-only after init).
                    _use_undistortion.store(true);
                }
            }
        }
    }
    catch(json::exception& e){
        logger::error("[{}] Component profile read exception : {}", get_name(), e.what());
        return false;
    }
    catch(cv::Exception::exception& e){
        logger::error("[{}] Device open exception : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void uvc_camera_grabber::on_loop(){

    
}


void uvc_camera_grabber::on_close(){

    /* stop worker */
    _worker_stop.store(true);

    /* stop grabbing */
    for_each(_grab_worker.begin(), _grab_worker.end(), [](auto& t) {
        if(t.second.joinable()){
            t.second.join();
            logger::debug("Camera #{} grabber is successfully stopped", t.first);
        }
    });
    _grab_worker.clear();

    /* close sockets and pipe */
    for(auto& s : _pub_sockets) {
        s.second->close();
    }
    _pub_sockets.clear();
    flame::pipe::destroy_pipe();

}

void uvc_camera_grabber::on_message(const flame::component::message_t& msg){
    // Note: The 'msg' parameter is currently unused.
}

void uvc_camera_grabber::_grab_task(int camera_id, json camera_param){

    string device = camera_param.value("device", "");

    if(camera_id<0 || device.empty()){
        logger::warn("[{}] Undefined or Invalid Camera Configuration in the Component Profile.", get_name());
        return;
    }

    cv::VideoCapture _cap; /* camera capture */
    try{

        /* camera open */
        _cap.open(device, CAP_V4L2); /* for linux only */
        _cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        if(!_cap.isOpened()){
            logger::error("[{}] Camera #{} cannot be opened.", get_name(), camera_id);
            return;
        }

        /* read port configurations */
        string monitoring_portname = fmt::format("image_stream_monitor_{}", camera_id);

        string stream_portname = fmt::format("image_stream_{}", camera_id);

        /* Create AsyncZSocket for EPGM */
        string socket_id = fmt::format("camera_{}", camera_id);
        auto socket = std::make_shared<flame::pipe::AsyncZSocket>(socket_id, flame::pipe::Pattern::PUBLISH);
        if(socket->create(_pipe)) {
            // EPGM requires multicast address. Defaulting to eth0;239.192.1.1 if not in config.
            // Address format: "interface;multicast_group"
            string epgm_addr = "eth0;239.192.1.1"; 
            if(camera_param.contains("epgm_address")){
                epgm_addr = camera_param["epgm_address"].get<string>();
            }
            int epgm_port = 5555 + camera_id;
            
            if(socket->join(flame::pipe::Transport::EPGM, epgm_addr, epgm_port)){
                logger::info("[{}] Camera #{} publishing via EPGM: {}:{}", get_name(), camera_id, epgm_addr, epgm_port);
                _pub_sockets[camera_id] = socket;
            } else {
                logger::error("[{}] Failed to join EPGM for camera #{}", get_name(), camera_id);
            }
        } else {
             logger::error("[{}] Failed to create socket for camera #{}", get_name(), camera_id);
        }

        json dataport_config = get_profile()->dataport();
        int monitoring_width = 480; 
        int monitoring_height = 270;

        if (dataport_config.contains(monitoring_portname)) {
            monitoring_width = dataport_config.at(monitoring_portname).at("resolution").value("width", 480);
            monitoring_height = dataport_config.at(monitoring_portname).at("resolution").value("height", 270);
        }

        json tag;
        auto last_time = chrono::high_resolution_clock::now();
        logger::debug("[{}] Camera #{} grabbing is now working...", get_name(), camera_id);

        cv::Mat raw_frame;
        cv::Mat undistorted_frame;
        cv::Mat rotated_frame;
        cv::Mat monitor_image;
        std::vector<unsigned char> serialized_image;
        std::vector<unsigned char> serialized_monitor_image;
        double rotation = _rotation_cw.load();
        bool do_undistort = _use_undistortion.load();
        
        /* init undistortion maps if needed */
        cv::Mat K, D;
        if(do_undistort){
             // Parse calibration again here or pass it? parsing here is safe for thread isolation
             // Accessing get_profile() is thread safe? profile is unique_ptr, read-only usually.
             json params = get_profile()->parameters();
             if(params.contains("calibration")){
                 json calib = params["calibration"];
                 vector<double> f = calib["focal_length"];
                 vector<double> c = calib["principal_point"];
                 vector<double> d = calib["distortion"];
                 K = cv::Mat::eye(3, 3, CV_64F);
                 K.at<double>(0,0) = f[0]; K.at<double>(1,1) = f[1];
                 K.at<double>(0,2) = c[0]; K.at<double>(1,2) = c[1];
                 D = cv::Mat::zeros(1, 5, CV_64F);
                 for(size_t i=0; i<d.size() && i<5; ++i) D.at<double>(0,i) = d[i];
             }
        }

        while(!_worker_stop.load()){

            /* capture from camera */
            _cap >> raw_frame;
            if(raw_frame.empty()){
                logger::warn("[{}] Camera #{}({}) frame is empty", get_name(), camera_id, device);
                continue;
            }

            /* apply undistortion */
            if(do_undistort && !K.empty()){
                if(_map1.empty() || _map1.size() != raw_frame.size()){
                    unique_lock<mutex> lock(_calibration_mtx);
                    if(_map1.empty() || _map1.size() != raw_frame.size()){
                         cv::initUndistortRectifyMap(K, D, cv::Mat(), K, raw_frame.size(), CV_16SC2, _map1, _map2);
                         logger::info("[{}] Undistortion map initialized for {}x{}", get_name(), raw_frame.cols, raw_frame.rows);
                    }
                }
                cv::remap(raw_frame, undistorted_frame, _map1, _map2, cv::INTER_LINEAR);
            } else {
                undistorted_frame = raw_frame;
            }

            /* rotate if needed */
            if (abs(rotation) > 0.1) {
                if (abs(rotation - 90.0) < 0.1) cv::rotate(undistorted_frame, rotated_frame, cv::ROTATE_90_CLOCKWISE);
                else if (abs(rotation - 180.0) < 0.1) cv::rotate(undistorted_frame, rotated_frame, cv::ROTATE_180);
                else if (abs(rotation - 270.0) < 0.1) cv::rotate(undistorted_frame, rotated_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                else rotated_frame = undistorted_frame; 
            } else {
                rotated_frame = undistorted_frame; 
            }

            /* generate tag */
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - last_time;
            last_time = now;
            tag["fps"] = 1.0/elapsed.count();
            tag["camera_id"] = camera_id;
            tag["height"] = rotated_frame.rows;
            tag["width"] = rotated_frame.cols;
            tag["timestamp"] = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
            string tag_str = tag.dump();

            /* transfer original image to processs */
            if(_use_image_stream.load()){
                /* image encoding */
                cv::imencode(".jpg", rotated_frame, serialized_image);

                /* Send via AsyncZSocket (EPGM) */
                if(_pub_sockets.count(camera_id)){
                    auto& sock = _pub_sockets[camera_id];
                    vector<string> msg;
                    // Topic "camera_{id}" is implicitly handled by dispatch for PUBLISH pattern
                    msg.push_back(tag_str);
                    msg.push_back(string(serialized_image.begin(), serialized_image.end()));
                    
                    sock->dispatch(msg);
                }

                // Existing port logic kept or removed?
                // Request says "change structure to use zpipe ... publish via epgm"
                // Assuming replacement of the old mechanism, but I'll comment out old one or just leave it if backward compatibility needed?
                // The request says "change structure", implying a replacement. I will comment out the old port logic to be safe.
                
                /*
                auto* port = get_port(stream_portname);
                if(port != nullptr && port->handle()!=nullptr){
                    zmq::multipart_t msg_multipart_image;
                    msg_multipart_image.addstr(tag_str);
                    msg_multipart_image.addmem(serialized_image.data(), serialized_image.size());
                    msg_multipart_image.send(*port, ZMQ_DONTWAIT);
                }
                else{
                    logger::warn("[{}] {} socket handle is not valid ", get_name(), camera_id);
                }
                */
            }

            /* transfer small image for monitoring */
            if(_use_image_stream_monitoring.load()){
                
                cv::resize(rotated_frame, monitor_image, cv::Size(monitoring_width, monitoring_height));
                cv::imencode(".jpg", monitor_image, serialized_monitor_image);

                auto* port = get_port(monitoring_portname);
                if(port != nullptr && port->handle()!=nullptr){
                    zmq::multipart_t msg_multipart;
                    msg_multipart.addstr(fmt::format("{}/image_stream_monitor_{}", get_name(), camera_id));
                    msg_multipart.addstr(tag_str);
                    msg_multipart.addmem(serialized_monitor_image.data(), serialized_monitor_image.size());
                    msg_multipart.send(*port, ZMQ_DONTWAIT);
                }
            }

        } /* end while */

        /* realse */
        _cap.release();
        logger::info("[{}] Camera #{}({}) is released", get_name(), camera_id, device);
    }
    catch(const cv::Exception& e){
        logger::error("[{}] Camera #{} CV Exception : {}", get_name(), camera_id, e.err);
        logger::debug("[{}] {}", get_name(), e.what());
        _cap.release();
    }
    catch(const std::out_of_range& e){
        logger::error("[{}] Invalid parameter access", get_name());
    }
    catch(const zmq::error_t& e){
        logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
    }
    catch(const json::exception& e){
        logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
    }

}
