#ifndef OSM_MONOLITHIC_INFERENCE_V2_FACE_ANALYSIS_E2E_HPP_INCLUDED
#define OSM_MONOLITHIC_INFERENCE_V2_FACE_ANALYSIS_E2E_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

#ifndef OSM_HEAD_POSE_STRUCT_DEFINED
#define OSM_HEAD_POSE_STRUCT_DEFINED
namespace head_pose {
    struct PoseResult {
        cv::Mat rvec;                        // 3x1 double rotation vector
        cv::Mat tvec;                        // 3x1 double translation vector
        cv::Vec3d euler;                     // (pitch, yaw, roll) in degrees
        cv::Point2f nose_tip_2d{0.0f, 0.0f}; // Anchor coordinate at 2D nose tip
        bool success = false;
    };
}
#endif

namespace face_analysis {
    struct FaceAnalysisResult {
        cv::Rect square_bbox;                        // 1:1 square bounding box
        std::vector<cv::Point2f> landmarks_68;       // 68 facial landmarks in original image space
        std::vector<cv::Point2f> landmarks_191;      // 191 head landmarks in original image space
        head_pose::PoseResult pose;                  // 3D Head pose (pitch, yaw, roll)
        std::vector<float> params_3dmm;              // 413 FLAME 3DMM parameters
        std::vector<cv::Point3f> vertices_3d;        // 5023 3D mesh vertices
        std::vector<cv::Point2f> projected_vertices; // 5023 projected 2D vertices in original image space
        bool valid = false;
    };
}

class face_analysis_e2e {
public:
    face_analysis_e2e();
    ~face_analysis_e2e();

    // Load TorchScript DAD-3DHeads E2E model
    bool loadModel(const std::string& model_path, int gpu_id = 0);

    // Process single face bounding box from YOLO detection
    face_analysis::FaceAnalysisResult process(
        const cv::Mat& orig_image,
        const cv::Rect& face_bbox
    );

    // Draw visual landmarks, pose axes, and square box
    void drawResult(
        cv::Mat& image,
        const face_analysis::FaceAnalysisResult& result,
        bool draw_68 = true,
        bool draw_191 = false,
        bool draw_pose = true,
        bool draw_box = true,
        bool draw_mesh = false
    );

private:
    // Make 1:1 aspect ratio square bounding box centered around longest side
    cv::Rect makeSquareBox(const cv::Rect& bbox, int img_w, int img_h);

    // Extract Euler angles and rotation matrix from 6D rotation vector
    void computeHeadPose(
        const float* pred_3dmm_ptr,
        const cv::Point2f& nose_tip,
        head_pose::PoseResult& out_pose
    );

private:
    torch::jit::script::Module _module;
    torch::Device _device = torch::Device(torch::kCPU);
    int _gpu_id = 0;
    bool _is_loaded = false;

    static constexpr int MODEL_INPUT_SIZE = 256;
};

#endif
