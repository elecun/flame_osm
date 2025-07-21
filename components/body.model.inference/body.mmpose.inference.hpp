/**
 * @file hpe.model.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Human Pose Estimation Model Inference Component
 * @version 0.1
 * @date 2025-04-03
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_HPE_MODLE_INFERENCE_HPP_INCLUDED
#define FLAME_HPE_MODLE_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <memory>
#include <onnxruntime_cxx_api.h> //version 1.22.0

class hpe_model_inference : public flame::component::object {
    public:
        hpe_model_inference();
        virtual ~hpe_model_inference() = default;

        /* default interface functions */
        bool on_init() override;
        void on_loop() override;
        void on_close() override;
        void on_message() override;

    private:
        void _camera_stream_process(int stream_id);

    private:
    /* worker related */
    atomic<bool> _worker_stop { false };
    thread _camera_stream_process_worker;

    /* ONNXRuntime Related */
    Ort::Env _onnx_env;
    Ort::SessionOptions _onnx_session_options;
    unique_ptr<Ort::Session> _onnx_session;
        

}; /* class */

EXPORT_COMPONENT_API

#endif