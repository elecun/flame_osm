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
static uvc_camera_grabber *_instance = nullptr;
flame::component::object *create() {
  if (!_instance)
    _instance = new uvc_camera_grabber();
  return _instance;
}
void release() {
  if (_instance) {
    delete _instance;
    _instance = nullptr;
  }
}

bool uvc_camera_grabber::on_init() {

  try {
    /* read profile */
    json parameters = get_profile()->parameters();

    /* init zpipe */
    _pipe = flame::pipe::create_pipe(1);

    /* create status socket */
    json dataport_config = get_profile()->dataport();
    if (dataport_config.contains("status")) {
      auto socket = _create_socket("status", dataport_config);
      if (socket)
        _pub_sockets["status"] = socket;
    }

    /* set video capture instance */
    int auto_id = 1;
    if (parameters.contains("camera")) {
      for (auto &dev : parameters["camera"]) {
        int id = dev.value("id", auto_id++);
        dev["id"] = id; /* update camera id */

        /* assign grabber worker */
        _grab_worker[id] =
            thread(&uvc_camera_grabber::_grab_task, this, id, dev);
      }
    } else {
      logger::warn("[{}] Cannot found camera(s) available", get_name());
      return false;
    }

    /* configure data for data pipelining  */
    _use_image_stream_monitoring.store(
        parameters.value("use_image_stream_monitoring", false));
    _use_image_stream.store(parameters.value("use_image_stream", false));
    _rotation_cw.store(parameters.value("rotation_cw", 0.0));
    _worker_stop.store(false);

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

          /* pre-compute undistort maps (will initialize size later once
           * resolution is known or assume port resolution) */
          // Wait.. we need image resolution to init maps.
          // Since resolution might depend on camera, we can defer map init or
          // do it if we know resolution. Let's assume resolution from dataport
          // or first frame. Better approach: In _grab_task, check if maps are
          // empty and init them once. So here we store K and D, or just rely on
          // parameters access in _grab_task? Storing K and D in member might be
          // cleaner but thread safety? Let's parse here but compute maps in
          // _grab_task to be safe with image size. Actually, simpler: Just
          // enable flags here and parse in task or helper. Given the structure,
          // parsing in on_init and storing locally to transfer to task or
          // member variables is best. Let's just set the flag here and parse
          // K/D inside the task or valid place. Actually, K and D are specific
          // to camera. If multiple cameras, we need a map of K/D. The current
          // json has "calibration" at root, implying one calibration for the
          // component (likely single camera case). We will parse it into
          // members (protected by mutex if needed or just read-only after
          // init).
          _use_undistortion.store(true);
        }
      }
    }
  } catch (json::exception &e) {
    logger::error("[{}] Component profile read exception : {}", get_name(),
                  e.what());
    return false;
  } catch (cv::Exception::exception &e) {
    logger::error("[{}] Device open exception : {}", get_name(), e.what());
    return false;
  }

  return true;
}

void uvc_camera_grabber::on_loop() {
  if (_worker_stop.load())
    return;

  if (_pub_sockets.count("status")) {
    auto &sock = _pub_sockets["status"];

    json status;
    status["component"] = get_name();
    status["state"] = "running";
    status["timestamp"] = chrono::duration_cast<chrono::milliseconds>(
                              chrono::system_clock::now().time_since_epoch())
                              .count();

    vector<string> msg;
    msg.push_back(fmt::format("{}/status", get_name()));
    msg.push_back(status.dump());

    // Prevent segfault if socket was cleared off map concurrently
    if (sock != nullptr) {
      sock->dispatch(msg);
    }
  }
}

void uvc_camera_grabber::on_close() {

  /* stop worker */
  _worker_stop.store(true);

  /* stop grabbing */
  for_each(_grab_worker.begin(), _grab_worker.end(), [](auto &t) {
    if (t.second.joinable()) {
      t.second.join();
      logger::debug("Camera #{} grabber is successfully stopped", t.first);
    }
  });
  _grab_worker.clear();

  /* close sockets and pipe */
  for (auto &s : _pub_sockets) {
    s.second->close();
  }
  _pub_sockets.clear();
}

void uvc_camera_grabber::on_message(const flame::component::message_t &msg) {
  // Note: The 'msg' parameter is currently unused.
}

std::shared_ptr<flame::pipe::AsyncZSocket>
uvc_camera_grabber::_create_socket(const std::string &name,
                                   const json &dataport_config) {
  if (!dataport_config.contains(name)) {
    logger::warn("[{}] Dataport '{}' is not defined in the configuration.",
                 get_name(), name);
    return nullptr;
  }

  auto port_config = dataport_config[name];
  std::string transport_str = port_config.value("transport", "tcp");
  std::string socket_type_str = port_config.value("socket_type", "pub");
  std::string host = port_config.value("host", "*");
  int port = port_config.value("port", 5555);

  flame::pipe::Transport transport = flame::pipe::Transport::TCP;
  if (transport_str == "epgm")
    transport = flame::pipe::Transport::EPGM;
  else if (transport_str == "pgm")
    transport = flame::pipe::Transport::PGM;
  else if (transport_str == "ipc")
    transport = flame::pipe::Transport::IPC;
  else if (transport_str == "inproc")
    transport = flame::pipe::Transport::INPROC;

  flame::pipe::Pattern pattern = flame::pipe::Pattern::PUBLISH;
  if (socket_type_str == "sub")
    pattern = flame::pipe::Pattern::SUBSCRIBE;
  else if (socket_type_str == "push")
    pattern = flame::pipe::Pattern::PUSH;
  else if (socket_type_str == "pull")
    pattern = flame::pipe::Pattern::PULL;

  auto socket = std::make_shared<flame::pipe::AsyncZSocket>(name, pattern);
  if (socket->create(_pipe)) {
    try {
      auto *old_port = get_port(name);
      if (old_port) {
        old_port->close();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        logger::info("[{}] Closed existing framework socket for '{}'",
                     get_name(), name);
      }
    } catch (...) {
      // Port not found or already closed
    }

    if (socket->join(transport, host, port)) {
      logger::info("[{}] Socket '{}' created and joined at {}://{}:{}",
                   get_name(), name, transport_str, host, port);
      return socket;
    } else {
      logger::error("[{}] Failed to join socket '{}'", get_name(), name);
    }
  } else {
    logger::error("[{}] Failed to create socket '{}'", get_name(), name);
  }
  return nullptr;
}

void uvc_camera_grabber::_grab_task(int camera_id, json camera_param) {

  string device = camera_param.value("device", "");

  if (camera_id < 0 || device.empty()) {
    logger::warn("[{}] Undefined or Invalid Camera Configuration in the "
                 "Component Profile.",
                 get_name());
    return;
  }

  cv::VideoCapture _cap; /* camera capture */
  try {

    /* camera open */
    _cap.open(device, CAP_V4L2); /* for linux only */
    _cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!_cap.isOpened()) {
      logger::error("[{}] Camera #{} cannot be opened.", get_name(), camera_id);
      return;
    }

    /* read port configurations */
    string stream_portname = camera_param.value("dataport", "");
    string monitoring_portname = camera_param.value("monitorport", "");
    json dataport_config = get_profile()->dataport();

    // Create socket for streaming original image
    if (!stream_portname.empty() && dataport_config.contains(stream_portname)) {
      auto socket = _create_socket(stream_portname, dataport_config);
      if (socket)
        _pub_sockets[stream_portname] = socket;
    } else {
      logger::warn(
          "[{}] stream dataport for camera #{} is invalid or not defined",
          get_name(), camera_id);
    }

    // Create socket for monitoring image
    if (!monitoring_portname.empty() &&
        dataport_config.contains(monitoring_portname)) {
      auto socket = _create_socket(monitoring_portname, dataport_config);
      if (socket)
        _pub_sockets[monitoring_portname] = socket;
    } else {
      logger::warn(
          "[{}] monitor dataport for camera #{} is invalid or not defined",
          get_name(), camera_id);
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
    logger::debug("[{}] Camera #{} grabbing is now working...", get_name(),
                  camera_id);

    cv::Mat raw_frame;
    cv::Mat undistorted_frame;
    cv::Mat rotated_frame;
    cv::Mat monitor_image;
    std::vector<unsigned char> serialized_image;
    std::vector<unsigned char> serialized_monitor_image;
    double rotation = _rotation_cw.load();
    bool do_undistort = _use_undistortion.load();

    /* init undistortion maps if needed */
    cv::Mat K, D;
    if (do_undistort) {
      // Parse calibration again here or pass it? parsing here is safe for
      // thread isolation Accessing get_profile() is thread safe? profile is
      // unique_ptr, read-only usually.
      json params = get_profile()->parameters();
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

    while (!_worker_stop.load()) {

      /* capture from camera */
      _cap >> raw_frame;
      if (raw_frame.empty()) {
        logger::warn("[{}] Camera #{}({}) frame is empty", get_name(),
                     camera_id, device);
        continue;
      }

      /* apply undistortion */
      if (do_undistort && !K.empty()) {
        if (_map1.empty() || _map1.size() != raw_frame.size()) {
          unique_lock<mutex> lock(_calibration_mtx);
          if (_map1.empty() || _map1.size() != raw_frame.size()) {
            cv::initUndistortRectifyMap(K, D, cv::Mat(), K, raw_frame.size(),
                                        CV_16SC2, _map1, _map2);
            logger::info("[{}] Undistortion map initialized for {}x{}",
                         get_name(), raw_frame.cols, raw_frame.rows);
          }
        }
        cv::remap(raw_frame, undistorted_frame, _map1, _map2, cv::INTER_LINEAR);
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
      if (_use_image_stream.load()) {
        /* image encoding */
        cv::imencode(".jpg", rotated_frame, serialized_image);

        /* Send via AsyncZSocket */
        if (_pub_sockets.count(stream_portname)) {
          auto &sock = _pub_sockets[stream_portname];
          vector<string> msg;
          msg.push_back(tag_str);
          msg.push_back(
              string(serialized_image.begin(), serialized_image.end()));

          sock->dispatch(msg);
        } else {
          logger::warn("[{}] {} socket is not valid", get_name(),
                       stream_portname);
        }
      }

      /* transfer small image for monitoring */
      if (_use_image_stream_monitoring.load()) {

        cv::resize(rotated_frame, monitor_image,
                   cv::Size(monitoring_width, monitoring_height));
        cv::imencode(".jpg", monitor_image, serialized_monitor_image);

        /* Send relative monitor image via AsyncZSocket */
        if (_pub_sockets.count(monitoring_portname)) {
          auto &sock = _pub_sockets[monitoring_portname];
          vector<string> msg;
          msg.push_back(
              fmt::format("{}/image_stream_monitor_{}", get_name(), camera_id));
          msg.push_back(tag_str);
          msg.push_back(string(serialized_monitor_image.begin(),
                               serialized_monitor_image.end()));
          sock->dispatch(msg);
        } else {
          logger::warn("[{}] {} socket is not valid", get_name(),
                       monitoring_portname);
        }
      }

    } /* end while */

    /* realse */
    _cap.release();
    logger::info("[{}] Camera #{}({}) is released", get_name(), camera_id,
                 device);
  } catch (const cv::Exception &e) {
    logger::error("[{}] Camera #{} CV Exception : {}", get_name(), camera_id,
                  e.err);
    logger::debug("[{}] {}", get_name(), e.what());
    _cap.release();
  } catch (const std::out_of_range &e) {
    logger::error("[{}] Invalid parameter access", get_name());
  } catch (const zmq::error_t &e) {
    logger::error("[{}] Piepeline Error : {}", get_name(), e.what());
  } catch (const json::exception &e) {
    logger::error("[{}] Data Parse Error : {}", get_name(), e.what());
  }
}
