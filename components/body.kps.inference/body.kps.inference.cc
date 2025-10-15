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

        _engine_path = parameters.value("engine_path", "");
        if(_engine_path.empty()){
            logger::error("[{}] TensorRT engine path is not defined", get_name());
            return false;
        }

        if(!fs::exists(_engine_path)){
            logger::error("[{}] TensorRT engine file not found: {}", get_name(), _engine_path);
            return false;
        }

        /* Load TensorRT engine */
        if(!_load_engine(_engine_path)){
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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

void body_kps_inference::on_message(){
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

std::vector<PoseResult> body_kps_inference::_postprocess_output(float* output, int batch_size){
    std::vector<PoseResult> results;
    
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
            PoseResult result;
            result.bbox_confidence = max_kpt_conf;
            result.bbox = cv::Rect(
                (x_center - width/2) * 1920 / _input_width,
                (y_center - height/2) * 1080 / _input_height,
                width * 1920 / _input_width,
                height * 1080 / _input_height
            );
            
            /* Extract keypoints */
            for(int k = 0; k < _num_keypoints; k++){
                KeyPoint kpt;
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

void body_kps_inference::_draw_keypoints(cv::Mat& image, const std::vector<PoseResult>& results){
    /* COCO keypoint connections for skeleton */
    const std::vector<std::pair<int, int>> skeleton = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9},
        {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13},
        {13, 15}, {12, 14}, {14, 16}
    };
    
    for(const auto& result : results){
        /* Draw bounding box */
        cv::rectangle(image, result.bbox, cv::Scalar(0, 255, 0), 2);
        
        /* Draw keypoints */
        for(size_t i = 0; i < result.keypoints.size(); i++){
            const auto& kpt = result.keypoints[i];
            if(kpt.confidence > 0.5f){
                cv::circle(image, cv::Point(kpt.x, kpt.y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        /* Draw skeleton */
        for(const auto& connection : skeleton){
            if(connection.first < result.keypoints.size() && 
               connection.second < result.keypoints.size()){
                const auto& kpt1 = result.keypoints[connection.first];
                const auto& kpt2 = result.keypoints[connection.second];
                
                if(kpt1.confidence > 0.5f && kpt2.confidence > 0.5f){
                    cv::line(image, cv::Point(kpt1.x, kpt1.y), 
                            cv::Point(kpt2.x, kpt2.y), cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    }
}

void body_kps_inference::_inference_process(){
    logger::info("[{}] Inference thread started", get_name());
    
    int frame_count = 0;
    
    while(!_worker_stop.load()){
        try{
            /* Load test image (temporary - replace with actual image source) */
            std::string test_image_path = fmt::format("./test_image_{}.jpg", frame_count % 10);
            cv::Mat input_image;
            
            /* Try to load test image, if not found create a dummy image */
            if(fs::exists(test_image_path)){
                input_image = cv::imread(test_image_path);
            } else {
                /* Create dummy 1920x1080 image for testing */
                input_image = cv::Mat::zeros(1080, 1920, CV_8UC3);
                cv::putText(input_image, fmt::format("Test Frame {}", frame_count), 
                           cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
            }
            
            if(input_image.empty()){
                logger::warn("[{}] No input image available, skipping frame", get_name());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            /* Preprocess image */
            cv::Mat processed_image = _preprocess_image(input_image);
            
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
                std::vector<PoseResult> results = _postprocess_output(_cpu_output_buffer, 1);
                
                /* Draw keypoints on original image */
                _draw_keypoints(input_image, results);
                
                /* Save result image */
                std::string output_path = fmt::format("./keypoint_result_{}.jpg", frame_count);
                cv::imwrite(output_path, input_image);
                
                logger::debug("[{}] Processed frame {}, detected {} poses", 
                             get_name(), frame_count, results.size());
            } else {
                logger::error("[{}] TensorRT inference failed", get_name());
            }
            
            frame_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
            
        }
        catch(const std::exception& e){
            logger::error("[{}] Exception in inference thread: {}", get_name(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    logger::info("[{}] Inference thread stopped", get_name());
}
