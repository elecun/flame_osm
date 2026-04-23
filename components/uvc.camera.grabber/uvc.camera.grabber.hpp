/**
 * @file uvc.camera.grabber.hpp
 * @author Byunghun Hwang <bh.hwang@iae.re.kr>
 * @brief UVC Camera Component using opencv uvc camera interface
 * @version 0.1
 * @date 2025-04-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef FLAME_UVC_CAMERA_GRABBER_HPP_INCLUDED
#define FLAME_UVC_CAMERA_GRABBER_HPP_INCLUDED

#include <flame/common/zpipe.hpp>
#include <flame/component/object.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

using namespace std;

class UvcCameraGrabber : public flame::component::Object {
public:
  UvcCameraGrabber() = default;
  virtual ~UvcCameraGrabber() = default;

  /* default interface functions */
  bool onInit() override;
  void onLoop() override;
  void onClose() override;
  void onData(flame::component::ZData &data) override;

private:
  /* grabber tasks */
  void grabTask(int camera_id, json camera_param);

  /* private function */
  vector<string> findAvailableCamera(int n_max = 10, const string prefix = "/dev/video");

private:
  /* grabbing worker */
  unordered_map<int, thread> grab_worker_;

  /* zpipe */
  /* flag */
  atomic<bool> worker_stop_{false};
  atomic<bool> use_image_stream_monitoring_{false};
  atomic<bool> use_image_stream_{false};
  atomic<double> rotation_cw_{0.0};
  mutex calibration_mtx_;
  cv::Mat map1_, map2_;
  atomic<bool> use_undistortion_{false};

}; /* class */

EXPORT_COMPONENT_API

#endif