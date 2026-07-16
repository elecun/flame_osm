#include "osm.monolithic.inference.hpp"
#include <flame/log.hpp>
#include <dep/json.hpp>
#include <chrono>

using json = nlohmann::json;

/* create component instance */
static osm_monolithic_inference* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new osm_monolithic_inference(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

osm_monolithic_inference::osm_monolithic_inference() {
}

bool osm_monolithic_inference::onInit(){
    try{
        json parameters = getProfile()->parameters();
        
        std::string model_path = parameters.value("model_path", "bin/x86_64/models/yolo11n-face.engine");
        int gpu_id = parameters.value("gpu_id", 0);

        /* Load monitor port configuration for image_stream_1_processed_monitor */
        json dataport_cfg = getProfile()->dataPort();
        if (dataport_cfg.contains("image_stream_1_processed_monitor")) {
            const auto& port_cfg = dataport_cfg["image_stream_1_processed_monitor"];
            if (port_cfg.contains("resolution")) {
                const auto& r = port_cfg["resolution"];
                if (r.contains("width") && r.contains("height")) {
                    _has_target_resolution = true;
                    _target_width = r["width"].get<int>();
                    _target_height = r["height"].get<int>();
                    logger::info("[{}] Monitor port target resolution: {}x{}", getName(), _target_width, _target_height);
                }
            }
        }

        _face_detector = std::make_unique<face_detection>();
        if (!_face_detector->loadModel(model_path, gpu_id)) {
            logger::error("[{}] Failed to load face detection model: {}", getName(), model_path);
            return false;
        }

        /* Start Inference thread */
        _worker_stop.store(false);
        _inference_worker = std::thread(&osm_monolithic_inference::_inference_process, this);

        logger::info("[{}] Initialized osm.monolithic.inference component", getName());
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }
    catch(const std::exception& e){
        logger::error("[{}] Initialization Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void osm_monolithic_inference::onLoop(){
}

void osm_monolithic_inference::onClose(){
    logger::info("[{}] Closing osm.monolithic.inference component", getName());

    /* Stop Worker Thread */
    _worker_stop.store(true);
    if (_inference_worker.joinable()) {
        _inference_worker.join();
    }

    if (_face_detector) {
        _face_detector.reset();
        logger::info("[{}] Face detector instance successfully released", getName());
    }
}

void osm_monolithic_inference::onData(flame::component::ZData& data){
    try {
        if (data.size() >= 3) {
            std::string portname = data.popstr();
            std::string tag_str = data.popstr();
            zmq::message_t image_msg = data.pop();

            if (portname == "image_stream_1" || portname == "image_stream_2") {
                json tag = json::parse(tag_str);
                int height = tag["height"].get<int>();
                int width = tag["width"].get<int>();
                int type = tag["type"].get<int>();

                // Restore image Mat from payload
                cv::Mat raw_img(height, width, type, image_msg.data());
                cv::Mat cloned_img = raw_img.clone();

                if (portname == "image_stream_1") {
                    std::lock_guard<std::mutex> lock(_img_mutex_1);
                    _latest_image_1 = cloned_img;
                } else if (portname == "image_stream_2") {
                    std::lock_guard<std::mutex> lock(_img_mutex_2);
                    _latest_image_2 = cloned_img;
                }
            }
        }
    }
    catch (const std::exception& e) {
        logger::error("[{}] Error in onData: {}", getName(), e.what());
    }
}

void osm_monolithic_inference::_inference_process() {
    logger::info("[{}] Inference worker thread started", getName());

    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};

    while (!_worker_stop.load()) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        cv::Mat image = getLatestImage1();
        if (image.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        try {
            /* 1. Run YOLO11-Face detection */
            std::vector<cv::Rect> bboxes = _face_detector->process(image);

            /* 2. Draw bounding boxes on the image */
            for (const auto& box : bboxes) {
                cv::rectangle(image, box, cv::Scalar(0, 255, 0), 3);
            }

            /* 3. Resize if target monitor resolution is defined */
            cv::Mat out_image;
            if (_has_target_resolution && (_target_width != image.cols || _target_height != image.rows)) {
                cv::resize(image, out_image, cv::Size(_target_width, _target_height), 0, 0, cv::INTER_LINEAR);
            } else {
                out_image = image;
            }

            /* 4. Encode as JPEG */
            std::vector<uchar> jpeg_buf;
            if (cv::imencode(".jpg", out_image, jpeg_buf, encode_params)) {
                
                /* 5. Construct metadata tags */
                json tag;
                tag["width"] = out_image.cols;
                tag["height"] = out_image.rows;
                tag["type"] = out_image.type();
                tag["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                tag["cam_channel"] = 1;

                /* 6. Send multipart message (from & meta automatically prepended by ZSocket::dispatch) */
                flame::component::ZData out_msg;
                out_msg.from = "image_stream_1_processed_monitor";
                out_msg.meta = tag.dump();
                out_msg.addmem(jpeg_buf.data(), jpeg_buf.size());

                if (!dispatch("image_stream_1_processed_monitor", out_msg)) {
                    logger::warn("[{}] Failed to dispatch processed image to port image_stream_1_processed_monitor", getName());
                }
            } else {
                logger::warn("[{}] JPEG encoding failed", getName());
            }

        } catch (const std::exception& e) {
            logger::error("[{}] Error in inference loop: {}", getName(), e.what());
        }

        /* Sleep to maintain maximum 30 FPS rate */
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        long long sleep_time = std::max(1LL, 33LL - duration);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }

    logger::info("[{}] Inference worker thread stopped", getName());
}

cv::Mat osm_monolithic_inference::getLatestImage1() {
    std::lock_guard<std::mutex> lock(_img_mutex_1);
    return _latest_image_1.clone();
}

cv::Mat osm_monolithic_inference::getLatestImage2() {
    std::lock_guard<std::mutex> lock(_img_mutex_2);
    return _latest_image_2.clone();
}
