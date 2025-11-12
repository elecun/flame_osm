
#include "face.detection.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>

/* create component instance */
static face_detection_inference* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new face_detection_inference(); return _instance; }
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

face_detection_inference::face_detection_inference() {
    
}

bool face_detection_inference::on_init(){
    try{
        logger::info("[{}] Initializing component", get_name());

        /* Load model path from config */
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

        /* Load TensorRT engine */
        if(!_load_engine(_model_path)){
            logger::error("[{}] Failed to load TensorRT engine", get_name());
            return false;
        }

        /* Allocate CUDA buffers */
        _allocate_buffers();

        /* Start inference worker thread */
        _worker_stop.store(false);
        _inference_worker = std::thread(&face_detection_inference::_inference_process, this);

        logger::info("[{}] Component initialized successfully", get_name());
        return true;
    }
    catch(const std::exception& e){
        logger::error("[{}] Exception in on_init: {}", get_name(), e.what());
        return false;
    }
}

void face_detection_inference::on_loop(){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void face_detection_inference::on_close(){
    logger::info("[{}] Closing component", get_name());

    /* Stop worker thread */
    _worker_stop.store(true);
    if(_inference_worker.joinable()){
        _inference_worker.join();
    }

    /* Free CUDA buffers */
    _free_buffers();

    /* Reset TensorRT objects */
    _context.reset();
    _engine.reset();
    _runtime.reset();

    logger::info("[{}] Component closed", get_name());
}

void face_detection_inference::on_message(const message_t& msg){
    // Handle messages if needed
}

void face_detection_inference::_allocate_buffers(){
    /* Calculate buffer sizes */
    _input_size = 1 * 3 * _input_height * _input_width * sizeof(float);

    /* YOLO11 detection output: [batch, num_boxes, channels]
     * num_boxes = 8400
     * channels = 4 (bbox: x1, y1, x2, y2) + 1 (confidence) + 80 (classes) = 85
     * But for face detection model trained on custom dataset, classes = 1
     * So channels = 4 + 1 + 1 = 6
     */
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
                 get_name(), _input_size, _output_size);
}

void face_detection_inference::_free_buffers(){
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

bool face_detection_inference::_load_engine(const std::string& engine_path){
    try{
        logger::info("[{}] Loading TensorRT engine from: {}", get_name(), engine_path);

        /* Read engine file */
        std::ifstream engine_file(engine_path, std::ios::binary);
        if(!engine_file.good()){
            logger::error("[{}] Failed to open engine file: {}", get_name(), engine_path);
            return false;
        }

        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);
        engine_file.close();

        logger::info("[{}] Engine file size: {} bytes", get_name(), engine_size);

        /* Create TensorRT runtime and deserialize engine */
        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if(!_runtime){
            logger::error("[{}] Failed to create TensorRT runtime", get_name());
            return false;
        }

        _engine.reset(_runtime->deserializeCudaEngine(engine_data.data(), engine_size));
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

    /* YOLO11 detection output format: [boxes, channels] where channels = 4(bbox) + 1(obj_conf) + 1(class) */
    const int num_boxes = 8400;
    const int num_channels = 6;  // 4 + 1 + 1
    const float conf_threshold = 0.5f;

    /* Track best result (largest face with highest confidence) */
    float best_score = 0.0f;
    int best_index = -1;
    float best_area = 0.0f;

    /* Find the detection with highest confidence and largest area */
    for(int i = 0; i < num_boxes; i++){
        float* box_data = output + i * num_channels;

        /* Object confidence is at index 4 */
        float obj_conf = box_data[4];

        if(obj_conf > conf_threshold){
            /* Extract bounding box - xyxy format */
            float x1 = box_data[0];
            float y1 = box_data[1];
            float x2 = box_data[2];
            float y2 = box_data[3];

            /* Calculate area */
            float width = x2 - x1;
            float height = y2 - y1;
            float area = width * height;

            /* Select based on confidence and area (prefer larger faces) */
            float score = obj_conf * sqrt(area);  // Weighted score

            if(score > best_score){
                best_score = score;
                best_index = i;
                best_area = area;
            }
        }
    }

    /* Extract only the best detection */
    if(best_index >= 0){
        float* box_data = output + best_index * num_channels;

        /* Extract bounding box - xyxy format */
        float x1 = box_data[0];
        float y1 = box_data[1];
        float x2 = box_data[2];
        float y2 = box_data[3];
        float obj_conf = box_data[4];

        /* Remove letterbox padding and scale to original image coordinates */
        float x1_unpadded = (x1 - _letterbox_pad_left) / _letterbox_scale;
        float y1_unpadded = (y1 - _letterbox_pad_top) / _letterbox_scale;
        float x2_unpadded = (x2 - _letterbox_pad_left) / _letterbox_scale;
        float y2_unpadded = (y2 - _letterbox_pad_top) / _letterbox_scale;

        float width_unpadded = x2_unpadded - x1_unpadded;
        float height_unpadded = y2_unpadded - y1_unpadded;

        face_detection::FaceResult result;
        result.confidence = obj_conf;
        result.bbox = cv::Rect(
            std::max(0.0f, x1_unpadded),
            std::max(0.0f, y1_unpadded),
            std::min((float)img_width - x1_unpadded, width_unpadded),
            std::min((float)img_height - y1_unpadded, height_unpadded)
        );

        results.push_back(result);
    }

    return results;
}

void face_detection_inference::_inference_process(){
    logger::info("[{}] Inference thread started", get_name());

    unsigned long frame_count = 0;

    while(!_worker_stop.load()){
        try{
            auto process_start = chrono::high_resolution_clock::now();

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

                /* Set input shape for dynamic shape engine */
                nvinfer1::Dims input_dims;
                input_dims.nbDims = 4;
                input_dims.d[0] = 1;  // batch size
                input_dims.d[1] = 3;  // channels (RGB)
                input_dims.d[2] = _input_height;
                input_dims.d[3] = _input_width;

                if(!_context->setInputShape("images", input_dims)){
                    logger::error("[{}] Failed to set input shape", get_name());
                    continue;
                }

                /* Set tensor addresses */
                void* bindings[] = {_gpu_input_buffer, _gpu_output_buffer};
                bool success = _context->executeV2(bindings);

                if(success){
                    /* Copy output from GPU */
                    cudaMemcpy(_cpu_output_buffer, _gpu_output_buffer, _output_size, cudaMemcpyDeviceToHost);

                    /* Postprocess results */
                    std::vector<face_detection::FaceResult> results = _postprocess_output(_cpu_output_buffer, 1, decoded.cols, decoded.rows);

                    auto process_end = chrono::high_resolution_clock::now();
                    auto total_time = chrono::duration_cast<chrono::milliseconds>(process_end - process_start).count();

                    logger::info("[{}] Frame {}: processing time = {}ms, detected {} face(s)",
                                get_name(), frame_count, total_time, results.size());

                    /* Draw bounding box on image */
                    for(const auto& result : results){
                        /* Draw bounding box */
                        cv::rectangle(decoded, result.bbox, cv::Scalar(0, 255, 0), 3);

                        /* Draw confidence text */
                        std::string conf_text = fmt::format("Face: {:.2f}", result.confidence);
                        cv::putText(decoded, conf_text,
                                cv::Point(result.bbox.x, result.bbox.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                    }

                    /* Send annotated image via ZMQ */
                    if(get_port("image_stream_1_monitor")->handle() != nullptr){
                        /* Resize for monitoring */
                        cv::Mat resized;
                        cv::resize(decoded, resized, cv::Size(540, 960));

                        /* Encode image */
                        vector<unsigned char> encoded;
                        cv::imencode(".jpg", resized, encoded);

                        /* Create tag */
                        json monitor_tag;
                        monitor_tag["id"] = 1;
                        monitor_tag["fps"] = 0;
                        monitor_tag["timestamp"] = total_time;
                        monitor_tag["width"] = resized.cols;
                        monitor_tag["height"] = resized.rows;

                        /* Create multipart message */
                        message_t monitor_msg;
                        monitor_msg.addstr("image_stream_1_monitor");
                        monitor_msg.addstr(tag_str);
                        monitor_msg.addmem(encoded.data(), encoded.size());

                        /* Send to monitor port */
                        if(!monitor_msg.send(*get_port("image_stream_1_monitor"), ZMQ_DONTWAIT)){
                            if(!_worker_stop.load()){
                                logger::warn("[{}] Failed to send annotated image", get_name());
                            }
                        }
                        monitor_msg.clear();
                    }
                }
                else{
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
