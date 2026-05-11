
#include "face.detection.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;

/* create component instance */
static face_detection_inference* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new face_detection_inference(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

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

face_detection_inference::face_detection_inference() {
    
}

bool face_detection_inference::onInit(){
    try{
        logger::info("[{}] Initializing component", getName());

        /* Load model path from config */
        json parameters = getProfile()->parameters();

        _model_path = parameters.value("model_path", "");
        if(_model_path.empty()){
            logger::error("[{}] TensorRT engine path is not defined", getName());
            return false;
        }

        if(!fs::exists(_model_path)){
            logger::error("[{}] TensorRT engine file not found: {}", getName(), _model_path);
            return false;
        }

        /* Read model parameters */
        _input_width = parameters.value("input_width", 640);
        _input_height = parameters.value("input_height", 640);
        _gpu_id = parameters.value("gpu_id", 0);

        /* Set CUDA device */
        cudaError_t cuda_status = cudaSetDevice(_gpu_id);
        if(cuda_status != cudaSuccess){
            logger::error("[{}] Failed to set CUDA device {}: {}", getName(), _gpu_id, cudaGetErrorString(cuda_status));
            return false;
        }
        logger::info("[{}] Using GPU device: {}", getName(), _gpu_id);

        /* Load TensorRT engine */
        if(!_load_engine(_model_path)){
            logger::error("[{}] Failed to load TensorRT engine", getName());
            return false;
        }

        /* Create CUDA stream for async execution */
        cudaError_t stream_status = cudaStreamCreate(&_cuda_stream);
        if(stream_status != cudaSuccess){
            logger::error("[{}] Failed to create CUDA stream: {}", getName(), cudaGetErrorString(stream_status));
            return false;
        }
        logger::info("[{}] CUDA stream created successfully", getName());

        /* Allocate CUDA buffers */
        _allocate_buffers();

        /* Start inference worker thread */
        _worker_stop.store(false);
        _inference_worker = std::thread(&face_detection_inference::_inference_process, this);

        logger::info("[{}] Component initialized successfully", getName());
        return true;
    }
    catch(const std::exception& e){
        logger::error("[{}] Exception in onInit: {}", getName(), e.what());
        return false;
    }
}

void face_detection_inference::onLoop(){
}

void face_detection_inference::onClose(){
    logger::info("[{}] Closing component", getName());

    /* Stop worker thread */
    _worker_stop.store(true);
    _queue_cv.notify_all();
    if(_inference_worker.joinable()){
        _inference_worker.join();
    }

    /* Free CUDA buffers */
    _free_buffers();

    /* Reset TensorRT objects */
    _context.reset();
    _engine.reset();
    _runtime.reset();

    logger::info("[{}] Component closed", getName());
}

void face_detection_inference::onData(flame::component::ZData& data){
    /* Handle incoming messages: push to queue */
    {
        lock_guard<mutex> lock(_queue_mtx);
        if(_data_queue.size() < _max_queue_size) {
            _data_queue.push(std::move(data));
        }
        else {
            // Drop oldest if queue is full
            _data_queue.pop();
            _data_queue.push(std::move(data));
        }
    }
    _queue_cv.notify_one();
}

void face_detection_inference::_allocate_buffers(){
    /* Calculate buffer sizes */
    _input_size = 1 * 3 * _input_height * _input_width * sizeof(float);

    /* YOLO11 detection output: [batch, num_boxes, channels] */
    const int num_boxes = 8400;
    const int num_channels = 6;  // 4(bbox) + 1(obj_conf) + 1(face_class)
    _output_size = 1 * num_boxes * num_channels * sizeof(float);

    /* Allocate CPU buffers */
    _cpu_input_buffer = new float[_input_size / sizeof(float)];
    _cpu_output_buffer = new float[_output_size / sizeof(float)];

    /* Allocate GPU buffers */
    cudaMalloc(&_gpu_input_buffer, _input_size);
    cudaMalloc(&_gpu_output_buffer, _output_size);

    logger::info("[{}] Allocated buffers - input: {} bytes, output: {} bytes",
                 getName(), _input_size, _output_size);
}

void face_detection_inference::_free_buffers(){
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

    logger::debug("[{}] CUDA buffers and stream freed", getName());
}

bool face_detection_inference::_load_engine(const std::string& engine_path){
    try{
        logger::info("[{}] Loading TensorRT engine from: {}", getName(), engine_path);

        /* Read engine file */
        std::ifstream engine_file(engine_path, std::ios::binary);
        if(!engine_file.good()){
            logger::error("[{}] Failed to open engine file: {}", getName(), engine_path);
            return false;
        }

        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);
        engine_file.close();

        /* Create TensorRT runtime and deserialize engine */
        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if(!_runtime){
            logger::error("[{}] Failed to create TensorRT runtime", getName());
            return false;
        }

        _engine.reset(_runtime->deserializeCudaEngine(engine_data.data(), engine_size));
        if(!_engine){
            logger::error("[{}] Failed to deserialize CUDA engine", getName());
            return false;
        }

        _context.reset(_engine->createExecutionContext());
        if(!_context){
            logger::error("[{}] Failed to create execution context", getName());
            return false;
        }

        /* Print engine bindings info */
        int32_t num_bindings = _engine->getNbIOTensors();
        logger::info("[{}] TensorRT engine has {} IO tensors", getName(), num_bindings);

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
                        getName(), i, name, dim_str, (int)dtype, (int)mode);
        }

        logger::info("[{}] TensorRT engine loaded successfully", getName());
        return true;
    }
    catch(const std::exception& e){
        logger::error("[{}] Exception in _load_engine: {}", getName(), e.what());
        return false;
    }
}

cv::Mat face_detection_inference::_preprocess_image(const cv::Mat& image){
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

    return processed;
}

std::vector<face_detection::FaceResult> face_detection_inference::_postprocess_output(float* output, int batch_size, int img_width, int img_height){
    std::vector<face_detection::FaceResult> results;

    const int num_boxes = 8400;
    const int num_channels = 6;
    const float conf_threshold = 0.5f;

    float best_score = 0.0f;
    int best_index = -1;

    for(int i = 0; i < num_boxes; i++){
        float* box_data = output + i * num_channels;
        float obj_conf = box_data[4];

        if(obj_conf > conf_threshold){
            float x1 = box_data[0];
            float y1 = box_data[1];
            float x2 = box_data[2];
            float y2 = box_data[3];
            float area = (x2 - x1) * (y2 - y1);
            float score = obj_conf * sqrt(area);

            if(score > best_score){
                best_score = score;
                best_index = i;
            }
        }
    }

    if(best_index >= 0){
        float* box_data = output + best_index * num_channels;
        float x1 = box_data[0];
        float y1 = box_data[1];
        float x2 = box_data[2];
        float y2 = box_data[3];
        float obj_conf = box_data[4];

        float x1_unpadded = (x1 - _letterbox_pad_left) / _letterbox_scale;
        float y1_unpadded = (y1 - _letterbox_pad_top) / _letterbox_scale;
        float x2_unpadded = (x2 - _letterbox_pad_left) / _letterbox_scale;
        float y2_unpadded = (y2 - _letterbox_pad_top) / _letterbox_scale;

        face_detection::FaceResult result;
        result.confidence = obj_conf;
        result.bbox = cv::Rect(
            std::max(0.0f, x1_unpadded),
            std::max(0.0f, y1_unpadded),
            x2_unpadded - x1_unpadded,
            y2_unpadded - y1_unpadded
        );

        results.push_back(result);
    }

    return results;
}

void face_detection_inference::_inference_process(){
    logger::info("[{}] Inference thread started", getName());

    unsigned long frame_count = 0;

    while(!_worker_stop.load()){
        try{
            flame::component::ZData msg_multipart;
            {
                unique_lock<mutex> lock(_queue_mtx);
                _queue_cv.wait(lock, [this]{ return !_data_queue.empty() || _worker_stop.load(); });
                
                if(_worker_stop.load()) break;

                msg_multipart = std::move(_data_queue.front());
                _data_queue.pop();
            }

            if(msg_multipart.size() >= 3){

                auto process_start = chrono::high_resolution_clock::now();

                /* Extract parts: [portname, tag, image_data] */
                std::string portname = msg_multipart.popstr();
                std::string tag_str = msg_multipart.popstr();
                zmq::message_t image_msg = msg_multipart.pop();

                /* decode & convert image */
                vector<unsigned char> image(static_cast<unsigned char*>(image_msg.data()), static_cast<unsigned char*>(image_msg.data())+image_msg.size());
                cv::Mat decoded = cv::imdecode(image, cv::IMREAD_COLOR);

                if(decoded.empty()){
                    logger::warn("[{}] Failed to decode image", getName());
                    continue;
                }

                /* Preprocess image */
                cv::Mat processed_image = _preprocess_image(decoded);

                /* Copy to input buffer */
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

                /* Set input shape */
                nvinfer1::Dims input_dims;
                input_dims.nbDims = 4;
                input_dims.d[0] = 1;
                input_dims.d[1] = 3;
                input_dims.d[2] = _input_height;
                input_dims.d[3] = _input_width;

                if(!_context->setInputShape("images", input_dims)){
                    logger::error("[{}] Failed to set input shape", getName());
                    continue;
                }

                _context->setTensorAddress("images", _gpu_input_buffer);
                _context->setTensorAddress("output0", _gpu_output_buffer);

                /* Run inference */
                bool success = _context->enqueueV3(_cuda_stream);

                if(success){
                    cudaMemcpyAsync(_cpu_output_buffer, _gpu_output_buffer, _output_size, cudaMemcpyDeviceToHost, _cuda_stream);
                    cudaStreamSynchronize(_cuda_stream);

                    /* Postprocess results */
                    std::vector<face_detection::FaceResult> results = _postprocess_output(_cpu_output_buffer, 1, decoded.cols, decoded.rows);

                    auto process_end = chrono::high_resolution_clock::now();
                    auto total_time = chrono::duration_cast<chrono::milliseconds>(process_end - process_start).count();

                    logger::info("[{}] Frame {}: processing time = {}ms, detected {} face(s)",
                                getName(), frame_count, total_time, (int)results.size());

                    /* Draw bounding box on image */
                    for(const auto& result : results){
                        cv::rectangle(decoded, result.bbox, cv::Scalar(0, 255, 0), 3);
                    }
                    
                    /* Send via monitor port if needed (skipped for brevity, but same as body_kps) */
                }
                else{
                    logger::error("[{}] TensorRT inference failed", getName());
                }
            }
            frame_count++;
        }
        catch(const zmq::error_t& e){
            logger::error("[{}] ZMQ error in inference thread: {}", getName(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        catch(const std::exception& e){
            logger::error("[{}] Exception in inference thread: {}", getName(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    logger::info("[{}] Inference thread stopped", getName());
}
