
#include "headpose.model.inference.hpp"
#include <flame/log.hpp>
#include <flame/def.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace flame;
using namespace std;

/* create component instance */
static headpose_model_inference* _instance = nullptr;
flame::component::object* create(){ if(!_instance) _instance = new headpose_model_inference(); return _instance; }
void release(){ if(_instance){ delete _instance; _instance = nullptr; }}


bool headpose_model_inference::on_init(){

    try{

        /* read profile */
        json parameters = get_profile()->parameters();

        /* Initialize ZMQ */
        zmq_endpoint_ = parameters.value("zmq_endpoint", "tcp://localhost:5555");
        zmq_context_ = new zmq::context_t(1);
        zmq_socket_ = new zmq::socket_t(*zmq_context_, ZMQ_SUB);
        zmq_socket_->connect(zmq_endpoint_);
        zmq_socket_->setsockopt(ZMQ_SUBSCRIBE, "", 0);

        /* Initialize camera parameters */
        camera_matrix_ = (cv::Mat_<double>(3,3) <<
            800.0, 0.0, 320.0,
            0.0, 800.0, 240.0,
            0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64FC1);

        /* Initialize MediaPipe */
        init_mediapipe();

        /* Initialize 3D model points */
        init_3d_model_points();

        /* Initialize threading */
        is_running_ = true;
        processing_thread_ = std::thread(&headpose_model_inference::processing_loop, this);

        logger::info("[{}] Initialized successfully", get_name());

    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", get_name(), e.what());
        return false;
    }
    catch(std::exception& e){
        logger::error("[{}] Initialization Error : {}", get_name(), e.what());
        return false;
    }

    return true;
}

void headpose_model_inference::on_loop(){
  
        
 
}


void headpose_model_inference::on_close(){
    try{
        /* Stop processing thread */
        is_running_ = false;
        image_cv_.notify_all();
        if(processing_thread_.joinable()){
            processing_thread_.join();
        }

        /* Cleanup ZMQ */
        if(zmq_socket_){
            delete zmq_socket_;
            zmq_socket_ = nullptr;
        }
        if(zmq_context_){
            delete zmq_context_;
            zmq_context_ = nullptr;
        }

        /* Cleanup MediaPipe */
        if(graph_){
            graph_->CloseInputStream("input_video");
            graph_->WaitUntilDone();
            graph_.reset();
        }

        logger::info("[{}] Closed successfully", get_name());
    }
    catch(std::exception& e){
        logger::error("[{}] Close Error : {}", get_name(), e.what());
    }
}

void headpose_model_inference::on_message(const message_t& msg){
    // Note: The 'msg' parameter is currently unused.
}

void headpose_model_inference::init_mediapipe(){
    std::string calculator_graph_config_contents = R"(
        input_stream: "input_video"
        output_stream: "multi_face_landmarks"

        node {
            calculator: "FaceMeshCpu"
            input_stream: "IMAGE:input_video"
            output_stream: "LANDMARKS:multi_face_landmarks"
        }
    )";

    absl::Status status = mediapipe::ParseTextProto(calculator_graph_config_contents, &graph_config_);
    if (!status.ok()) {
        logger::error("[{}] Failed to parse graph config: {}", get_name(), status.ToString());
        return;
    }

    graph_ = std::make_unique<mediapipe::CalculatorGraph>();
    status = graph_->Initialize(graph_config_);
    if (!status.ok()) {
        logger::error("[{}] Failed to initialize graph: {}", get_name(), status.ToString());
        return;
    }

    status = graph_->StartRun({});
    if (!status.ok()) {
        logger::error("[{}] Failed to start graph: {}", get_name(), status.ToString());
        return;
    }
}

void headpose_model_inference::init_3d_model_points(){
    model_points_.clear();

    // Standard 3D face model points (approximated)
    model_points_.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));           // Nose tip
    model_points_.push_back(cv::Point3f(0.0f, -330.0f, -65.0f));      // Chin
    model_points_.push_back(cv::Point3f(-225.0f, 170.0f, -135.0f));   // Left eye left corner
    model_points_.push_back(cv::Point3f(225.0f, 170.0f, -135.0f));    // Right eye right corner
    model_points_.push_back(cv::Point3f(-150.0f, -150.0f, -125.0f));  // Left mouth corner
    model_points_.push_back(cv::Point3f(150.0f, -150.0f, -125.0f));   // Right mouth corner
}

void headpose_model_inference::processing_loop(){
    zmq::message_t message;

    while(is_running_){
        try{
            // Receive image from ZMQ
            if(zmq_socket_->recv(&message, ZMQ_NOBLOCK)){
                // Deserialize image
                cv::Mat image = cv::imdecode(cv::Mat(1, message.size(), CV_8UC1, message.data()), cv::IMREAD_COLOR);

                if(!image.empty()){
                    // Convert to MediaPipe format
                    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                        mediapipe::ImageFormat::SRGB, image.cols, image.rows,
                        mediapipe::ImageFrame::kDefaultAlignmentBoundary);

                    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
                    cv::cvtColor(image, input_frame_mat, cv::COLOR_BGR2RGB);

                    // Add frame to MediaPipe
                    size_t frame_timestamp_us = (std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());

                    absl::Status status = graph_->AddPacketToInputStream(
                        "input_video", mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us)));

                    if(!status.ok()){
                        logger::error("[{}] Failed to add packet: {}", get_name(), status.ToString());
                        continue;
                    }

                    // Get results
                    mediapipe::Packet packet;
                    if(graph_->GetOutputStreamPacket("multi_face_landmarks", &packet).ok() && !packet.IsEmpty()){
                        const auto& landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

                        if(!landmarks.empty()){
                            // Convert landmarks to 2D points
                            std::vector<cv::Point2f> image_points;
                            const auto& face_landmarks = landmarks[0];

                            // Get specific landmark points for pose estimation
                            int landmark_indices[] = {1, 152, 33, 263, 61, 291}; // nose, chin, left eye, right eye, left mouth, right mouth

                            for(int idx : landmark_indices){
                                if(idx < face_landmarks.landmark_size()){
                                    const auto& landmark = face_landmarks.landmark(idx);
                                    image_points.push_back(cv::Point2f(
                                        landmark.x() * image.cols,
                                        landmark.y() * image.rows
                                    ));
                                }
                            }

                            if(image_points.size() == 6){
                                // Calculate head pose
                                cv::Vec3d pose = calculate_head_pose(image_points);

                                // Print results
                                print_landmarks_and_pose(image_points, pose);
                            }
                        }
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS

        }
        catch(std::exception& e){
            logger::error("[{}] Processing Error: {}", get_name(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

cv::Vec3d headpose_model_inference::calculate_head_pose(const std::vector<cv::Point2f>& landmarks){
    if(landmarks.size() != 6){
        return cv::Vec3d(0, 0, 0);
    }

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(model_points_, landmarks, camera_matrix_, dist_coeffs_, rvec, tvec);

    if(!success){
        return cv::Vec3d(0, 0, 0);
    }

    // Convert rotation vector to Euler angles
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);

    // Extract Euler angles (in degrees)
    double pitch = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2)) * 180.0 / CV_PI;
    double yaw = atan2(-rotation_matrix.at<double>(2, 0), sqrt(pow(rotation_matrix.at<double>(2, 1), 2) + pow(rotation_matrix.at<double>(2, 2), 2))) * 180.0 / CV_PI;
    double roll = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0)) * 180.0 / CV_PI;

    return cv::Vec3d(pitch, yaw, roll);
}

void headpose_model_inference::print_landmarks_and_pose(const std::vector<cv::Point2f>& landmarks, const cv::Vec3d& pose){
    std::cout << "=== Face Landmarks ===" << std::endl;
    for(size_t i = 0; i < landmarks.size(); ++i){
        std::cout << "Point " << i << ": (" << landmarks[i].x << ", " << landmarks[i].y << ")" << std::endl;
    }

    std::cout << "=== Head Pose ===" << std::endl;
    std::cout << "Pitch: " << pose[0] << " degrees" << std::endl;
    std::cout << "Yaw: " << pose[1] << " degrees" << std::endl;
    std::cout << "Roll: " << pose[2] << " degrees" << std::endl;
    std::cout << "========================" << std::endl << std::endl;
}
