
#include "hpe.model.inference.hpp"
#include <flame/log.hpp>
#include <flame/config_def.hpp>
#include <chrono>
#include <algorithm>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static hpe_model_inference* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new hpe_model_inference(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}

hpe_model_inference::hpe_model_inference()
:_onnx_env(ORT_LOGGING_LEVEL_WARNING, "PoseEstimator"), _onnx_session_options(), _onnx_session(nullptr){

}

bool hpe_model_inference::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        string _onnx_model_path = parameters.value("onnx_model_path", "");
        if(_onnx_model_path.empty()){
            logger::error("[{}] ONNX Model path is not defined", get_name());
            return false;
        }

        /* set session options */
        _onnx_session_options.SetIntraOpNumThreads(1);

        /* create sesstion */
        _onnx_session = make_unique<Ort::Session>(_onnx_env, _onnx_model_path.c_str(), _onnx_session_options);
        if(_onnx_session==nullptr){
            logger::error("[{}] Failed ONNX Runtime session create", get_name());
            return false;
        }

        if(parameters.contains("image_stream_ids") && parameters["image_stream_ids"].is_array()){
            for(const auto& id:parameters["image_stream_ids"]){
                int stream_id = id.get<int>();
                /* start camera stream process */
                _camera_stream_process_worker = thread(&hpe_model_inference::_camera_stream_process, this, stream_id);
                logger::debug("[{}] HPE Inference processing is now running with camera stream ID #{}", get_name(), stream_id);
            }
        }
        
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void hpe_model_inference::on_loop(){
  
        
 
}


void hpe_model_inference::on_close(){
    
    _worker_stop.store(true);
    if(_camera_stream_process_worker.joinable()){
        _camera_stream_process_worker.join();
        logger::info("[{}] Component successfully closed.", get_name());
    }


}

void hpe_model_inference::on_message(){
    
}

void hpe_model_inference::_camera_stream_process(int stream_id){
    try{
        json tag;
        string portname = fmt::format("image_stream_{}", stream_id);

        while(!_worker_stop.load()){

            /* receive image stream from grabber component */
            zmq::multipart_t msg_multipart;
            bool success = msg_multipart.recv(*get_port(portname));

            /* received success */
            if(success){

                /* pop 2 data chunk from message */
                string camera_id = msg_multipart.popstr();
                zmq::message_t msg_image = msg_multipart.pop();
                string filename = fmt::format("{}_{}.jpg", camera_id, ++_stream_counter[stream_id]);

                /* save into multiple directories (camera_*)*/
                for(auto& path:_backup_dir_path){
                    fs::path camera_working_dir = path / working_dirname;
                    if(!fs::exists(camera_working_dir)){
                        fs::create_directories(camera_working_dir);
                        logger::info("[{}] Stream #{} data saves into {}", get_name(), stream_id, camera_working_dir.string());
                    }

                    std::ofstream out(fmt::format("{}/{}", camera_working_dir.string(), filename), std::ios::binary);
                    out.write(static_cast<char*>(msg_image.data()), msg_image.size());
                    out.close();
                }

                // release explicit
                msg_image = zmq::message_t();
                msg_multipart.clear();
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } /* end while */

        /* realse */
        
        logger::debug("[{}] Close camera stream process worker", get_name());
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