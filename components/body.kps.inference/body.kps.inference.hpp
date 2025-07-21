/**
 * @file body.kps.inference.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief Body Keypoint Detection Model Inference Component
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef FLAME_BODY_KPS_INFERENCE_HPP_INCLUDED
#define FLAME_BODY_KPS_INFERENCE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

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