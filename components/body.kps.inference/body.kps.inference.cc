#include "body.kps.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>

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
        _gpu_id = parameters.value("gpu_id", 0);

        /* Set CUDA device */
        cudaError_t cuda_status = cudaSetDevice(_gpu_id);
        if(cuda_status != cudaSuccess){
            logger::error("[{}] Failed to set CUDA device {}: {}", get_name(), _gpu_id, cudaGetErrorString(cuda_status));
            return false;
        }
        logger::info("[{}] Using GPU device: {}", get_name(), _gpu_id);

        /* Recalculate buffer sizes based on parameters */
        _input_size = 3 * _input_width * _input_height * sizeof(float);
        _output_size = (4 + _num_keypoints * 3) * 8400 * sizeof(float); //bounding box(x,y,w,h) x each kps(x,y,confidence), 8400 candidates

        /* Load TensorRT engine */
        if(!_load_engine(_model_path)){
            logger::error("[{}] Failed to load TensorRT engine", get_name());
            return false;
        }

        /* Create CUDA stream for async execution */
        cudaError_t stream_status = cudaStreamCreate(&_cuda_stream);
        if(stream_status != cudaSuccess){
            logger::error("[{}] Failed to create CUDA stream: {}", get_name(), cudaGetErrorString(stream_status));
            return false;
        }
        logger::info("[{}] CUDA stream created successfully", get_name());

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

        /* Print engine bindings info */
        int32_t num_bindings = _engine->getNbIOTensors();
        logger::info("[{}] TensorRT engine has {} IO tensors", get_name(), num_bindings);

        for(int i = 0; i < num_bindings; i++){
            const char* name = _engine->getIOTensorName(i);
            nvinfer1::Dims dims = _engine->getTensorShape(name);
            nvinfer1::DataType dtype = _engine->getTensorDataType(name);
            nvinfer1::TensorIOMode mode = _engine->getTensorIOMode(name);

            std::string dim_str = "[";
            for(int j = 0; j < dims.nbDims; j++){
                dim_str += std::to_string(dims.d[j]);
                if(j < dims.nbDims - 1) dim_str += ", ";
            }
            dim_str += "]";

            logger::info("[{}] Tensor {}: name='{}', shape={}, type={}, mode={}",
                        get_name(), i, name, dim_str, (int)dtype, (int)mode);
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
    /* Destroy CUDA stream */
    if(_cuda_stream){
        cudaStreamDestroy(_cuda_stream);
        _cuda_stream = nullptr;
    }

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

    logger::debug("[{}] CUDA buffers and stream freed", get_name());
}

cv::Mat body_kps_inference::_preprocess_image(const cv::Mat& image){
    cv::Mat processed;

    /* Letterbox resize to maintain aspect ratio */
    int orig_width = image.cols;
    int orig_height = image.rows;

    _letterbox_scale = std::min(
        (float)_input_width / orig_width,
        (float)_input_height / orig_height
    );

    int new_width = (int)(orig_width * _letterbox_scale);
    int new_height = (int)(orig_height * _letterbox_scale);

    /* Resize maintaining aspect ratio */
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    /* Create padded image (letterbox) */
    processed = cv::Mat::zeros(cv::Size(_input_width, _input_height), CV_8UC3);
    _letterbox_pad_top = (_input_height - new_height) / 2;
    _letterbox_pad_left = (_input_width - new_width) / 2;

    resized.copyTo(processed(cv::Rect(_letterbox_pad_left, _letterbox_pad_top, new_width, new_height)));

    /* Convert BGR to RGB */
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    /* Convert to float and normalize */
    processed.convertTo(processed, CV_32F, 1.0/255.0);

    // logger::info("Letterbox: orig={}x{}, scale={}, new={}x{}, pad=({},{})",
    //              orig_width, orig_height, _letterbox_scale, new_width, new_height, _letterbox_pad_left, _letterbox_pad_top);

    return processed;
}

std::vector<body_kps::PoseResult> body_kps_inference::_postprocess_output(float* output, int batch_size, int img_width, int img_height){
    std::vector<body_kps::PoseResult> results;

    /* YOLO11 pose output format: [boxes, channels] where channels = 4(bbox) + 1(obj_conf) + kpts*3 */
    const int num_boxes = 8400;
    const int num_channels = 4 + 1 + _num_keypoints * 3;  // 56 for YOLO11-pose
    const float conf_threshold = 0.5f;

    // logger::info("Output format: [boxes={}, channels={}]", num_boxes, num_channels);

    /* Track best result */
    float best_confidence = 0.0f;
    int best_index = -1;
    int count_above_threshold = 0;

    /* Find the detection with highest object confidence */
    for(int i = 0; i < num_boxes; i++){
        float* box_data = output + i * num_channels;

        /* Object confidence is at index 4 */
        float obj_conf = box_data[4];

        if(obj_conf > conf_threshold){
            count_above_threshold++;
        }

        if(obj_conf > best_confidence){
            best_confidence = obj_conf;
            best_index = i;
        }
    }

    // logger::info("Total detections above threshold ({}): {}, best_confidence: {:.4f}, best_index: {}", conf_threshold, count_above_threshold, best_confidence, best_index);

    /* Extract only the best detection */
    if(best_index >= 0){
        float* box_data = output + best_index * num_channels;

        /* Extract bounding box - boxes-first format */
        /* Format is [x1, y1, x2, y2] (xyxy format) not [cx, cy, w, h] */
        float x1 = box_data[0];
        float y1 = box_data[1];
        float x2 = box_data[2];
        float y2 = box_data[3];
        float obj_conf = box_data[4];

        /* Convert to center+size format and remove letterbox padding */
        float x1_unpadded = (x1 - _letterbox_pad_left) / _letterbox_scale;
        float y1_unpadded = (y1 - _letterbox_pad_top) / _letterbox_scale;
        float x2_unpadded = (x2 - _letterbox_pad_left) / _letterbox_scale;
        float y2_unpadded = (y2 - _letterbox_pad_top) / _letterbox_scale;

        float width_unpadded = x2_unpadded - x1_unpadded;
        float height_unpadded = y2_unpadded - y1_unpadded;

        body_kps::PoseResult result;
        result.bbox_confidence = obj_conf;
        result.bbox = cv::Rect(
            x1_unpadded,
            y1_unpadded,
            width_unpadded,
            height_unpadded
        );

        /* Extract keypoints - boxes-first format */
        /* Keypoints start at index 5: [vis0, x0, y0, vis1, x1, y1, ...] */
        for(int k = 0; k < _num_keypoints; k++){
            body_kps::KeyPoint kpt;
            int kpt_offset = 5 + k * 3;
            float kpt_vis = box_data[kpt_offset + 0];  // visibility/confidence
            float kpt_x = box_data[kpt_offset + 1];
            float kpt_y = box_data[kpt_offset + 2];

            /* Remove letterbox padding and scale to original image coordinates */
            kpt.x = (kpt_x - _letterbox_pad_left) / _letterbox_scale;
            kpt.y = (kpt_y - _letterbox_pad_top) / _letterbox_scale;
            kpt.confidence = kpt_vis;  // Use visibility as confidence
            result.keypoints.push_back(kpt);
        }

        results.push_back(result);
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

                /* decode & convert image */
                vector<unsigned char> image(static_cast<unsigned char*>(image_msg.data()), static_cast<unsigned char*>(image_msg.data())+image_msg.size());
                cv::Mat decoded = cv::imdecode(image, cv::IMREAD_COLOR);

                if(decoded.empty()){
                    logger::warn("[{}] Failed to decode image", get_name());
                    continue;
                }

                // logger::debug("[{}] Received image on port {}: {}x{}, tag: {}", get_name(), portname, decoded.cols, decoded.rows, tag_str);

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

                /* Copy input to GPU asynchronously */
                cudaMemcpyAsync(_gpu_input_buffer, _cpu_input_buffer, _input_size, cudaMemcpyHostToDevice, _cuda_stream);

                /* Set input shape for dynamic shape engine */
                // Input shape: [batch_size, channels, height, width] = [1, 3, 640, 640]
                nvinfer1::Dims input_dims;
                input_dims.nbDims = 4;
                input_dims.d[0] = 1;  // batch size
                input_dims.d[1] = 3;  // channels (RGB)
                input_dims.d[2] = _input_height;  // height
                input_dims.d[3] = _input_width;   // width

                if(!_context->setInputShape("images", input_dims)){
                    logger::error("[{}] Failed to set input shape", get_name());
                    continue;
                }

                /* Set tensor addresses */
                _context->setTensorAddress("images", _gpu_input_buffer);
                _context->setTensorAddress("output0", _gpu_output_buffer);

                /* Run inference asynchronously */
                bool success = _context->enqueueV3(_cuda_stream);

                if(success){
                    /* Copy output from GPU asynchronously and synchronize */
                    cudaMemcpyAsync(_cpu_output_buffer, _gpu_output_buffer, _output_size, cudaMemcpyDeviceToHost, _cuda_stream);
                    cudaStreamSynchronize(_cuda_stream);

                    /* Postprocess results (pass actual image dimensions) */
                    std::vector<body_kps::PoseResult> results = _postprocess_output(_cpu_output_buffer, 1, decoded.cols, decoded.rows);

                    auto process_end = chrono::high_resolution_clock::now();
                    auto total_time = chrono::duration_cast<chrono::milliseconds>(process_end - process_start).count();
                    logger::info("[{}] Frame {}: processing time = {}ms, detected {} poses", get_name(), frame_count, total_time, results.size());


                    /* Create JSON output */
                    // json output;
                    // output["id"] = 1;
                    // output["timestamp"] = chrono::duration_cast<chrono::milliseconds>(process_start.time_since_epoch()).count();
                    // output["fps"] = 0;
                    // output["num_poses"] = results.size();

                    // json poses_array = json::array();
                    // for(const auto& result : results){
                    //     json pose_obj;
                    //     pose_obj["bbox"] = {
                    //         {"x", result.bbox.x},
                    //         {"y", result.bbox.y},
                    //         {"width", result.bbox.width},
                    //         {"height", result.bbox.height}
                    //     };
                    //     pose_obj["confidence"] = result.bbox_confidence;

                    //     json keypoints_array = json::array();
                    //     for(size_t i = 0; i < result.keypoints.size(); i++){
                    //         const auto& kpt = result.keypoints[i];
                    //         keypoints_array.push_back({
                    //             {"id", i},
                    //             {"x", kpt.x},
                    //             {"y", kpt.y},
                    //             {"confidence", kpt.confidence}
                    //         });
                    //     }
                    //     pose_obj["keypoints"] = keypoints_array;
                    //     poses_array.push_back(pose_obj);
                    // }
                    // output["poses"] = poses_array;

                    /* Draw keypoints on image */
                    for(const auto& result : results){
                        // logger::info("[{}] Result bbox: ({}, {}, {}, {}), confidence: {}", get_name(), result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height, result.bbox_confidence);

                        /* Draw bounding box */
                        cv::rectangle(decoded, result.bbox, cv::Scalar(0, 255, 0), 2);

                        /* Draw keypoints */
                        for(size_t i = 0; i < result.keypoints.size(); i++){
                            const auto& kpt = result.keypoints[i];
                            if(kpt.confidence > 0.5f){
                                cv::circle(decoded, cv::Point(kpt.x, kpt.y), 5, cv::Scalar(0, 255, 0), -1);
                            }
                        }
                    }

                    /* Send annotated image via ZMQ */
                    if(get_port("image_stream_1_monitor")->handle() != nullptr){
                        /* Resize for monitoring */
                        cv::Mat resized;
                        cv::resize(decoded, resized, cv::Size(540, 960));

                        /* Encode image */
                        std::vector<unsigned char> encoded_image;
                        cv::imencode(".jpg", resized, encoded_image);

                        /* Create tag */
                        json monitor_tag;
                        monitor_tag["id"] = 1;
                        monitor_tag["fps"] = 0;
                        monitor_tag["timestamp"] = total_time;
                        monitor_tag["width"] = resized.cols;
                        monitor_tag["height"] = resized.rows;

                        /* Send multipart message */
                        message_t monitor_msg;
                        monitor_msg.addstr("image_stream_1_monitor");
                        monitor_msg.addstr(monitor_tag.dump());
                        monitor_msg.addmem(encoded_image.data(), encoded_image.size());

                        if(!monitor_msg.send(*get_port("image_stream_1_monitor"), ZMQ_DONTWAIT)){
                            if(!_worker_stop.load()){
                                logger::warn("[{}] Failed to send annotated image", get_name());
                            }
                        }
                        monitor_msg.clear();
                    }

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
