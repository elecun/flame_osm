
#include "uvc.camera.grabber.hpp"
#include <utility>
#include <flame/log.hpp>

vector<string> uvc_camera_grabber::find_available_camera(int n_max, const string prefix){

    std::vector<std::string> available_cameras;

    for(int i = 0; i <= n_max; ++i) {
        std::string device_path = prefix + std::to_string(i);
        if(FILE *file = fopen(device_path.c_str(), "r")){
            fclose(file);
            cv::VideoCapture cap(device_path);
            if (cap.isOpened()) {
                available_cameras.push_back(device_path);
                cap.release();

                logger::info("[{}] Camera ({}) is available", get_name(), device_path);
            }
        }
    }

    return std::move(available_cameras);
}