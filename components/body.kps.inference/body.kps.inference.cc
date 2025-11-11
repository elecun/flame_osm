#include "body.kps.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

/* create component instance */
static body_kps_inference* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new body_kps_inference(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}

/* TensorRT Logger implementation */
void Logger::log(Severity severity, const char* msg) noexcept {
    switch(severity) {
        case Severity::kERROR:
            logger::error("[TensorRT] {}", msg);
            break;
        case Severity::kWARNING:
            logger::warn("[TensorRT] {}", msg);
            break;
        case Severity::kINFO:
            logger::info("[TensorRT] {}", msg);
            break;
        case Severity::kVERBOSE:
            logger::debug("[TensorRT] {}", msg);
            break;
    }
}

body_kps_inference::body_kps_inference() {
    _input_size = 3 * _input_width * _input_height * sizeof(float);
    _output_size = (4 + _num_keypoints * 3) * 8400 * sizeof(float); // YOLO11 output format
}

bool body_kps_inference::on_init(){
    try{
        /* read profile */
        json parameters = get_profile()->parameters();

        _model_path = parameters.value("model_path", "");
        if(_model_path.empty()){
            logger::error("[{}] TensorRT engine path is not defined", get_name());
            return false;
        }

        if(!fs::exists(_model_path)){
            logger::error("[{}] TensorRT engine file not found: {}", get_name(), _model_path);
            return false;
        }

        /* Read model parameters */
        _input_width = parameters.value("input_width", 640);
        _input_height = parameters.value("input_height", 640);
        _num_keypoints = parameters.value("num_keypoints", 17);

        /* Recalculate buffer sizes based on parameters */
        _input_size = 3 * _input_width * _input_height * sizeof(float);
        _output_size = (4 + _num_keypoints * 3) * 8400 * sizeof(float); //bounding box(x,y,w,h) x each kps(x,y,confidence), 8400 candidates

        /* Load TensorRT engine */
        if(!_load_engine(_model_path)){
            logger::error("[{}] Failed to load TensorRT engine", get_name());
            return false;
        }

        /* Allocate CUDA memory */
        _allocate_buffers();

        /* Start inference thread */
        _inference_worker = thread(&body_kps_inference::_inference_process, this);
        logger::info("[{}] Body keypoint inference component initialized successfully", get_name());

    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", get_name(), e.what());
        return false;
    }
    catch(const std::exception& e){
        logger::error("[{}] Initialization Error : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void body_kps_inference::on_loop(){
    /* Main loop - can be used for periodic tasks */
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void body_kps_inference::on_close(){
    logger::info("[{}] Shutting down body keypoint inference component", get_name());
    
    /* Stop inference thread */
    _worker_stop.store(true);
    if(_inference_worker.joinable()){
        _inference_worker.join();
    }

    /* Free CUDA memory and TensorRT resources */
    _free_buffers();
    _context.reset();
    _engine.reset();
    _runtime.reset();

    logger::info("[{}] Component successfully closed", get_name());
}

void body_kps_inference::on_message(const message_t& msg){
    /* Handle incoming messages if needed */
}

bool body_kps_inference::_load_engine(const std::string& engine_path){
    try{
        /* Read engine file */
        std::ifstream file(engine_path, std::ios::binary);
        if(!file.good()){
            logger::error("[{}] Failed to open engine file: {}", get_name(), engine_path);
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        /* Create TensorRT runtime and engine */
        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if(!_runtime){
            logger::error("[{}] Failed to create TensorRT runtime", get_name());
            return false;
        }

        _engine.reset(_runtime->deserializeCudaEngine(engine_data.data(), size));
        if(!_engine){
            logger::error("[{}] Failed to deserialize CUDA engine", get_name());
            return false;
        }

        _context.reset(_engine->createExecutionContext());
        if(!_context){
            logger::error("[{}] Failed to create execution context", get_name());
            return false;
        }

        logger::info("[{}] TensorRT engine loaded successfully", get_name());
        return true;
    }
    catch(const std::exception& e){
        logger::error("[{}] Exception in _load_engine: {}", get_name(), e.what());
        return false;
    }
}

void body_kps_inference::_allocate_buffers(){
    /* Allocate CPU buffers */
    _cpu_input_buffer = new float[_input_size / sizeof(float)];
    _cpu_output_buffer = new float[_output_size / sizeof(float)];

    /* Allocate GPU buffers */
    cudaMalloc(&_gpu_input_buffer, _input_size);
    cudaMalloc(&_gpu_output_buffer, _output_size);

    logger::info("[{}] CUDA buffers allocated - Input: {} bytes, Output: {} bytes", 
                get_name(), _input_size, _output_size);
}

void body_kps_inference::_free_buffers(){
    if(_cpu_input_buffer){
        delete[] _cpu_input_buffer;
        _cpu_input_buffer = nullptr;
    }
    
    if(_cpu_output_buffer){
        delete[] _cpu_output_buffer;
        _cpu_output_buffer = nullptr;
    }

    if(_gpu_input_buffer){
        cudaFree(_gpu_input_buffer);
        _gpu_input_buffer = nullptr;
    }

    if(_gpu_output_buffer){
        cudaFree(_gpu_output_buffer);
        _gpu_output_buffer = nullptr;
    }

    logger::debug("[{}] CUDA buffers freed", get_name());
}

cv::Mat body_kps_inference::_preprocess_image(const cv::Mat& image){
    cv::Mat processed;
    
    /* Resize to model input size */
    cv::resize(image, processed, cv::Size(_input_width, _input_height));
    
    /* Convert BGR to RGB */
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    /* Convert to float and normalize */
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    
    return processed;
}

std::vector<body_kps::PoseResult> body_kps_inference::_postprocess_output(float* output, int batch_size){
    std::vector<body_kps::PoseResult> results;
    
    /* YOLO11 pose output format: [batch, 4+kpts*3, 8400] */
    const int num_boxes = 8400;
    const float conf_threshold = 0.5f;
    const float nms_threshold = 0.4f;
    
    for(int i = 0; i < num_boxes; i++){
        float* box_data = output + i * (4 + _num_keypoints * 3);
        
        /* Extract bounding box */
        float x_center = box_data[0];
        float y_center = box_data[1];
        float width = box_data[2];
        float height = box_data[3];
        
        /* Calculate confidence (assuming it's embedded in keypoint confidences) */
        float max_kpt_conf = 0.0f;
        for(int k = 0; k < _num_keypoints; k++){
            float kpt_conf = box_data[4 + k * 3 + 2];
            max_kpt_conf = std::max(max_kpt_conf, kpt_conf);
        }
        
        if(max_kpt_conf > conf_threshold){
            body_kps::PoseResult result;
            result.bbox_confidence = max_kpt_conf;
            result.bbox = cv::Rect(
                (x_center - width/2) * 1920 / _input_width,
                (y_center - height/2) * 1080 / _input_height,
                width * 1920 / _input_width,
                height * 1080 / _input_height
            );

            /* Extract keypoints */
            for(int k = 0; k < _num_keypoints; k++){
                body_kps::KeyPoint kpt;
                kpt.x = box_data[4 + k * 3] * 1920 / _input_width;
                kpt.y = box_data[4 + k * 3 + 1] * 1080 / _input_height;
                kpt.confidence = box_data[4 + k * 3 + 2];
                result.keypoints.push_back(kpt);
            }

            results.push_back(result);
        }
    }
    
    return results;
}

void body_kps_inference::_inference_process(){

    unsigned long frame_count = 0;
    
    while(!_worker_stop.load()){
        try{
            /* Receive image from ZMQ */
            message_t msg_multipart;

            if(get_port("image_stream_1")->handle() == nullptr){
                logger::warn("[{}] image_stream_1 port handle is not valid", get_name());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            /* Receive multipart message with timeout */
            if(!msg_multipart.recv(*get_port("image_stream_1"))){
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if(msg_multipart.size()==3){

                auto process_start = chrono::high_resolution_clock::now();

                /* Extract parts: [portname, tag, image_data] */
                std::string portname = msg_multipart.popstr();
                std::string tag_str = msg_multipart.popstr();
                zmq::message_t image_msg = msg_multipart.pop();

                /* Parse tag JSON */
                json tag = json::parse(tag_str);

                vector<unsigned char> image(static_cast<unsigned char*>(image_msg.data()), static_cast<unsigned char*>(image_msg.data())+image_msg.size());
                cv::Mat decoded = cv::imdecode(image, cv::IMREAD_COLOR);

                if(decoded.empty()){
                    logger::warn("[{}] Failed to decode image", get_name());
                    continue;
                }

                logger::debug("[{}] Received image on port {}: {}x{}, tag: {}", get_name(), portname, decoded.cols, decoded.rows, tag_str);

                /* Preprocess image */
                cv::Mat processed_image = _preprocess_image(decoded);

                /* Copy to input buffer (HWC to CHW format) */
                for(int c = 0; c < 3; c++){
                    for(int h = 0; h < _input_height; h++){
                        for(int w = 0; w < _input_width; w++){
                            _cpu_input_buffer[c * _input_height * _input_width + h * _input_width + w] =
                                processed_image.at<cv::Vec3f>(h, w)[c];
                        }
                    }
                }

                /* Copy input to GPU */
                cudaMemcpy(_gpu_input_buffer, _cpu_input_buffer, _input_size, cudaMemcpyHostToDevice);

                /* Run inference */
                void* bindings[] = {_gpu_input_buffer, _gpu_output_buffer};
                bool success = _context->executeV2(bindings);

                if(success){
                    /* Copy output from GPU */
                    cudaMemcpy(_cpu_output_buffer, _gpu_output_buffer, _output_size, cudaMemcpyDeviceToHost);

                    /* Postprocess results */
                    std::vector<body_kps::PoseResult> results = _postprocess_output(_cpu_output_buffer, 1);

                    auto process_end = chrono::high_resolution_clock::now();
                    auto total_time = chrono::duration_cast<chrono::milliseconds>(process_end - process_start).count();
                    logger::info("[{}] Frame {}: processing time = {}ms, detected {} poses", get_name(), frame_count, total_time, results.size());


                    /* Create JSON output */
                    json output;
                    output["frame_id"] = frame_count;
                    output["timestamp"] = tag.value("timestamp", 0);
                    output["camera_id"] = tag.value("id", 1);
                    output["num_poses"] = results.size();

                    json poses_array = json::array();
                    for(const auto& result : results){
                        json pose_obj;
                        pose_obj["bbox"] = {
                            {"x", result.bbox.x},
                            {"y", result.bbox.y},
                            {"width", result.bbox.width},
                            {"height", result.bbox.height}
                        };
                        pose_obj["confidence"] = result.bbox_confidence;

                        json keypoints_array = json::array();
                        for(size_t i = 0; i < result.keypoints.size(); i++){
                            const auto& kpt = result.keypoints[i];
                            keypoints_array.push_back({
                                {"id", i},
                                {"x", kpt.x},
                                {"y", kpt.y},
                                {"confidence", kpt.confidence}
                            });
                        }
                        pose_obj["keypoints"] = keypoints_array;
                        poses_array.push_back(pose_obj);
                    }
                    output["poses"] = poses_array;

                    /* Send JSON result via ZMQ */
                    // if(get_port("pose_result")->handle() != nullptr){
                    //     std::string json_str = output.dump();
                    //     zmq::message_t result_msg(json_str.data(), json_str.size());
                    //     if(!result_msg.send(*get_port("pose_result"), ZMQ_DONTWAIT)){
                    //         if(!_worker_stop.load()){
                    //             logger::warn("[{}] Failed to send pose result", get_name());
                    //         }
                    //     }
                    // }


                } else {
                    logger::error("[{}] TensorRT inference failed", get_name());
                }
            }          

            frame_count++;

        }
        catch(const zmq::error_t& e){
            logger::error("[{}] ZMQ error in inference thread: {}", get_name(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        catch(const json::exception& e){
            logger::error("[{}] JSON parsing error: {}", get_name(), e.what());
        }
        catch(const std::exception& e){
            logger::error("[{}] Exception in inference thread: {}", get_name(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    logger::info("[{}] Inference thread stopped", get_name());
}
