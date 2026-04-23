#include "uvc.camera.grabber.hpp"
#include <algorithm>
#include <chrono>
#include <flame/def.hpp>
#include <flame/log.hpp>
#include <thread>

using namespace flame;
using namespace std;
using namespace cv;

/* create component instance */
static UvcCameraGrabber *instance_ = nullptr;
flame::component::Object *Create() {
  if (!instance_)
    instance_ = new UvcCameraGrabber();
  return instance_;
}
void Release() {
  if (instance_) {
    delete instance_;
    instance_ = nullptr;
  }
}

bool UvcCameraGrabber::onInit() {

  try {
    /* read profile */
    json parameters = getProfile()->parameters();

    /* set video capture instance */
    int auto_id = 1;
    if (parameters.contains("camera")) {
      for (auto &dev : parameters["camera"]) {
        int id = dev.value("id", auto_id++);
        dev["id"] = id; /* update camera id */

        /* assign grabber worker */
        grab_worker_[id] = thread(&UvcCameraGrabber::grabTask, this, id, dev);
      }
    } else {
      logger::warn("[{}] Cannot found camera(s) available", getName());
      return false;
    }

    /* configure data for data pipelining  */
    use_image_stream_monitoring_.store(
        parameters.value("use_image_stream_monitoring", false));
    use_image_stream_.store(parameters.value("use_image_stream", false));
    rotation_cw_.store(parameters.value("rotation_cw", 0.0));
    worker_stop_.store(false);

    /* load calibration data */
    if (parameters.contains("calibration")) {
      json calib = parameters["calibration"];
      if (calib.contains("focal_length") && calib.contains("principal_point") &&
          calib.contains("distortion")) {
        vector<double> f = calib["focal_length"];
        vector<double> c = calib["principal_point"];
        vector<double> d = calib["distortion"];

        if (f.size() >= 2 && c.size() >= 2 && d.size() >= 4) {
          cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
          K.at<double>(0, 0) = f[0];
          K.at<double>(1, 1) = f[1];
          K.at<double>(0, 2) = c[0];
          K.at<double>(1, 2) = c[1];

          cv::Mat D = cv::Mat::zeros(
              1, 5,
              CV_64F); // Assume at least 4 coeffs, support up to 5 standard
          for (size_t i = 0; i < d.size() && i < 5; ++i)
            D.at<double>(0, i) = d[i];

          use_undistortion_.store(true);
        }
      }
    }
  } catch (json::exception &e) {
    logger::error("[{}] Component profile read exception : {}", getName(),
                  e.what());
    return false;
  } catch (cv::Exception::exception &e) {
    logger::error("[{}] Device open exception : {}", getName(), e.what());
    return false;
  }

  return true;
}

void UvcCameraGrabber::onLoop() {
  if (worker_stop_.load())
    return;

  flame::component::ZData msg;
  msg.addstr("status");
  msg.addstr(""); // broadcast
  msg.addstr("json");

  json status;
  status["component"] = getName();
  status["state"] = "running";
  status["timestamp"] = chrono::duration_cast<chrono::milliseconds>(
                            chrono::system_clock::now().time_since_epoch())
                            .count();
  msg.addstr(status.dump());

  if (!dispatch("status", msg)) {
    logger::warn("[{}] status socket is not valid", getName());
  }
}

void UvcCameraGrabber::onClose() {

  /* stop worker */
  worker_stop_.store(true);

  /* stop grabbing */
  for_each(grab_worker_.begin(), grab_worker_.end(), [](auto &t) {
    if (t.second.joinable()) {
      t.second.join();
      logger::debug("Camera #{} grabber is successfully stopped", t.first);
    }
  });
  grab_worker_.clear();
}

void UvcCameraGrabber::onData(flame::component::ZData& data) {
  // data[0] = src_port, data[1] = dst_port, data[2] = type, data[3] = payload
}

void UvcCameraGrabber::grabTask(int camera_id, json camera_param) {

  string device = camera_param.value("device", "");

  if (camera_id < 0 || device.empty()) {
    logger::warn("[{}] Undefined or Invalid Camera Configuration in the "
                 "Component Profile.",
                 getName());
    return;
  }

  cv::VideoCapture cap; /* camera capture */
  try {

    /* camera open */
    cap.open(device, CAP_V4L2); /* for linux only */
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) {
      logger::error("[{}] Camera #{} cannot be opened.", getName(), camera_id);
      return;
    }

    /* read port configurations */
    string stream_portname = camera_param.value("dataport", "");
    string monitoring_portname = camera_param.value("monitorport", "");
    json dataport_config = getProfile()->dataPort();

    // Check socket for streaming original image
    if (stream_portname.empty() || !dataport_config.contains(stream_portname)) {
      logger::warn(
          "[{}] stream dataport for camera #{} is invalid or not defined",
          getName(), camera_id);
    }

    // Check socket for monitoring image
    if (monitoring_portname.empty() || !dataport_config.contains(monitoring_portname)) {
      logger::warn(
          "[{}] monitor dataport for camera #{} is invalid or not defined",
          getName(), camera_id);
    }
    int monitoring_width = 480;
    int monitoring_height = 270;

    if (dataport_config.contains(monitoring_portname)) {
      monitoring_width = dataport_config.at(monitoring_portname)
                             .at("resolution")
                             .value("width", 480);
      monitoring_height = dataport_config.at(monitoring_portname)
                              .at("resolution")
                              .value("height", 270);
    }

    json tag;
    auto last_time = chrono::high_resolution_clock::now();
    logger::debug("[{}] Camera #{} grabbing is now working...", getName(),
                  camera_id);

    cv::Mat raw_frame;
    cv::Mat undistorted_frame;
    cv::Mat rotated_frame;
    cv::Mat monitor_image;
    std::vector<unsigned char> serialized_image;
    std::vector<unsigned char> serialized_monitor_image;
    double rotation = rotation_cw_.load();
    bool do_undistort = use_undistortion_.load();

    /* init undistortion maps if needed */
    cv::Mat K, D;
    if (do_undistort) {
      json params = getProfile()->parameters();
      if (params.contains("calibration")) {
        json calib = params["calibration"];
        vector<double> f = calib["focal_length"];
        vector<double> c = calib["principal_point"];
        vector<double> d = calib["distortion"];
        K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = f[0];
        K.at<double>(1, 1) = f[1];
        K.at<double>(0, 2) = c[0];
        K.at<double>(1, 2) = c[1];
        D = cv::Mat::zeros(1, 5, CV_64F);
        for (size_t i = 0; i < d.size() && i < 5; ++i)
          D.at<double>(0, i) = d[i];
      }
    }

    while (!worker_stop_.load()) {

      /* capture from camera */
      cap >> raw_frame;
      if (raw_frame.empty()) {
        logger::warn("[{}] Camera #{}({}) frame is empty", getName(),
                     camera_id, device);
        continue;
      }

      /* apply undistortion */
      if (do_undistort && !K.empty()) {
        if (map1_.empty() || map1_.size() != raw_frame.size()) {
          unique_lock<mutex> lock(calibration_mtx_);
          if (map1_.empty() || map1_.size() != raw_frame.size()) {
            cv::initUndistortRectifyMap(K, D, cv::Mat(), K, raw_frame.size(),
                                        CV_16SC2, map1_, map2_);
            logger::info("[{}] Undistortion map initialized for {}x{}",
                         getName(), raw_frame.cols, raw_frame.rows);
          }
        }
        cv::remap(raw_frame, undistorted_frame, map1_, map2_, cv::INTER_LINEAR);
      } else {
        undistorted_frame = raw_frame;
      }

      /* rotate if needed */
      if (abs(rotation) > 0.1) {
        if (abs(rotation - 90.0) < 0.1)
          cv::rotate(undistorted_frame, rotated_frame, cv::ROTATE_90_CLOCKWISE);
        else if (abs(rotation - 180.0) < 0.1)
          cv::rotate(undistorted_frame, rotated_frame, cv::ROTATE_180);
        else if (abs(rotation - 270.0) < 0.1)
          cv::rotate(undistorted_frame, rotated_frame,
                     cv::ROTATE_90_COUNTERCLOCKWISE);
        else
          rotated_frame = undistorted_frame;
      } else {
        rotated_frame = undistorted_frame;
      }

      /* generate tag */
      auto now = chrono::high_resolution_clock::now();
      chrono::duration<double> elapsed = now - last_time;
      last_time = now;
      tag["fps"] = 1.0 / elapsed.count();
      tag["camera_id"] = camera_id;
      tag["height"] = rotated_frame.rows;
      tag["width"] = rotated_frame.cols;
      tag["timestamp"] =
          chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch())
              .count();
      string tag_str = tag.dump();

      /* transfer original image to processs */
      if (use_image_stream_.load()) {
        /* image encoding */
        cv::imencode(".jpg", rotated_frame, serialized_image);

        /* Send via dispatch - addmem avoids string copy for binary data */
        flame::component::ZData msg;
        msg.addstr(stream_portname);
        msg.addstr(""); // broadcast
        msg.addstr("binary");
        msg.addmem(serialized_image.data(), serialized_image.size());

        if (!dispatch(stream_portname, msg)) {
          logger::warn("[{}] {} socket is not valid", getName(),
                       stream_portname);
        }
      }

      /* transfer small image for monitoring */
      if (use_image_stream_monitoring_.load()) {

        cv::resize(rotated_frame, monitor_image,
                   cv::Size(monitoring_width, monitoring_height));
        cv::imencode(".jpg", monitor_image, serialized_monitor_image);

        /* Send monitor image via dispatch - addmem for binary zero-copy */
        flame::component::ZData msg;
        msg.addstr(monitoring_portname);
        msg.addstr(""); // broadcast
        msg.addstr("binary");
        msg.addmem(serialized_monitor_image.data(), serialized_monitor_image.size());

        if (!dispatch(monitoring_portname, msg)) {
          logger::warn("[{}] {} socket is not valid", getName(),
                       monitoring_portname);
        }
      }

    } /* end while */

    /* realse */
    cap.release();
    logger::info("[{}] Camera #{}({}) is released", getName(), camera_id,
                 device);
  } catch (const cv::Exception &e) {
    logger::error("[{}] Camera #{} CV Exception : {}", getName(), camera_id,
                  e.err);
    logger::debug("[{}] {}", getName(), e.what());
    cap.release();
  } catch (const std::out_of_range &e) {
    logger::error("[{}] Invalid parameter access", getName());
  } catch (const zmq::error_t &e) {
    logger::error("[{}] Piepeline Error : {}", getName(), e.what());
  } catch (const json::exception &e) {
    logger::error("[{}] Data Parse Error : {}", getName(), e.what());
  }
}