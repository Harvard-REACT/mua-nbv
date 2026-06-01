#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <mutex>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <rclcpp/rclcpp.hpp>

// FIX: Use .hpp for ROS 2 TF2 headers
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ring_generator.hpp"

namespace {

static std::string stampStr(const rclcpp::Time &t) {
  const int64_t ns = t.nanoseconds();
  const int64_t sec = ns / 1000000000LL;
  const int64_t nsec = ns % 1000000000LL;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%lld.%09lld", static_cast<long long>(sec),
                static_cast<long long>(nsec));
  return std::string(buf);
}

static geometry_msgs::msg::Quaternion
quatFromRotation(const Eigen::Matrix3d &R) {
  Eigen::Quaterniond q(R);
  q.normalize();
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

static geometry_msgs::msg::Pose poseFromIsometry(const Eigen::Isometry3d &T) {
  geometry_msgs::msg::Pose p;
  p.position.x = T.translation().x();
  p.position.y = T.translation().y();
  p.position.z = T.translation().z();
  p.orientation = quatFromRotation(T.rotation());
  return p;
}

static Eigen::Isometry3d
isoFromTransform(const geometry_msgs::msg::Transform &t) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() << t.translation.x, t.translation.y, t.translation.z;
  Eigen::Quaterniond q(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z);
  q.normalize();
  T.linear() = q.toRotationMatrix();
  return T;
}

static double yawFromRotation(const Eigen::Matrix3d &R) {
  // For planar yaw rotations around +Z: yaw = atan2(R21, R11)
  return std::atan2(R(1, 0), R(0, 0));
}

static visualization_msgs::msg::Marker
makeCenterMarker(const std::string &frame_id, const rclcpp::Time &stamp,
                 const Eigen::Vector3d &cW, double scale_m = 0.10) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "nbv_candidates";
  m.id = 0;
  m.type = visualization_msgs::msg::Marker::SPHERE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position.x = cW.x();
  m.pose.position.y = cW.y();
  m.pose.position.z = cW.z();
  m.pose.orientation.w = 1.0;
  m.scale.x = scale_m;
  m.scale.y = scale_m;
  m.scale.z = scale_m;
  m.color.r = 1.0f;
  m.color.g = 0.5f;
  m.color.b = 0.0f;
  m.color.a = 1.0f;
  return m;
}

static visualization_msgs::msg::Marker
makeHeadingArrowMarker(const std::string &frame_id, const rclcpp::Time &stamp,
                       const Eigen::Vector3d &cW,
                       const Eigen::Vector2d &heading_unit_xy,
                       double length_m) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "nbv_candidates";
  m.id = 1;
  m.type = visualization_msgs::msg::Marker::ARROW;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.03; // shaft diameter
  m.scale.y = 0.06; // head diameter
  m.scale.z = 0.06; // head length
  m.color.r = 0.0f;
  m.color.g = 0.6f;
  m.color.b = 0.0f;
  m.color.a = 1.0f;

  geometry_msgs::msg::Point p0;
  p0.x = cW.x();
  p0.y = cW.y();
  p0.z = cW.z();
  geometry_msgs::msg::Point p1;
  p1.x = cW.x() + length_m * heading_unit_xy.x();
  p1.y = cW.y() + length_m * heading_unit_xy.y();
  p1.z = cW.z();
  m.points.push_back(p0);
  m.points.push_back(p1);
  return m;
}

static visualization_msgs::msg::Marker
makeEllipseLineMarker(const std::string &frame_id, const rclcpp::Time &stamp,
                      const Eigen::Isometry3d &T_W_P, double zW, double a_m,
                      double b_m, int samples = 80) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "nbv_candidates";
  m.id = 2;
  m.type = visualization_msgs::msg::Marker::LINE_STRIP;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.02;
  m.color.r = 0.0f;
  m.color.g = 0.0f;
  m.color.b = 0.0f;
  m.color.a = 0.8f;

  const int N = std::max(8, samples);
  for (int i = 0; i <= N; ++i) {
    const double s = static_cast<double>(i) / static_cast<double>(N);
    const double phi = 2.0 * M_PI * s;
    const Eigen::Vector3d pP(a_m * std::cos(phi), b_m * std::sin(phi), 0.0);
    Eigen::Vector3d pW = (T_W_P * pP.homogeneous()).head<3>();
    pW.z() = zW;
    geometry_msgs::msg::Point pt;
    pt.x = pW.x();
    pt.y = pW.y();
    pt.z = pW.z();
    m.points.push_back(pt);
  }
  return m;
}

static visualization_msgs::msg::Marker makeCandidatePointsMarker(
    const std::string &frame_id, const rclcpp::Time &stamp,
    const std::vector<Eigen::Isometry3d> &T_W_B_list, double scale_m = 0.08) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "nbv_candidates";
  m.id = 3;
  m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = scale_m;
  m.scale.y = scale_m;
  m.scale.z = scale_m;
  m.color.r = 0.1f;
  m.color.g = 0.1f;
  m.color.b = 1.0f;
  m.color.a = 1.0f;
  m.points.reserve(T_W_B_list.size());
  for (const auto &T_W_B : T_W_B_list) {
    geometry_msgs::msg::Point pt;
    pt.x = T_W_B.translation().x();
    pt.y = T_W_B.translation().y();
    pt.z = T_W_B.translation().z();
    m.points.push_back(pt);
  }
  return m;
}

static visualization_msgs::msg::Marker makeReachabilityDiskMarker(
    const std::string &frame_id, const rclcpp::Time &stamp,
    const Eigen::Vector2d &robot_xy, double z_center_m, double radius_m,
    double height_m = 5.0, double alpha = 0.15) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "nbv_candidates";
  m.id = 4;
  m.type = visualization_msgs::msg::Marker::CYLINDER;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position.x = robot_xy.x();
  m.pose.position.y = robot_xy.y();
  // Place cylinder center at z_center_m (typically half height above ground)
  m.pose.position.z = z_center_m;
  m.pose.orientation.w = 1.0;
  const double r = std::max(0.0, radius_m);
  m.scale.x = 2.0 * r;
  m.scale.y = 2.0 * r;
  m.scale.z = std::max(1e-3, height_m);
  m.color.r = 0.0f;
  m.color.g = 1.0f;
  m.color.b = 1.0f;
  m.color.a = static_cast<float>(std::clamp(alpha, 0.0, 1.0));
  return m;
}

static cv::Mat renderCandidateEllipseImage(
    const Eigen::Vector2d &cW_xy, const Eigen::Vector2d &heading_unit_xy,
    const std::vector<Eigen::Vector2d> &cand_xy, const Eigen::Matrix2d &R_WP,
    double a_m, double b_m, int W = 800, int H = 800) {
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  const cv::Point2d cpx(W * 0.5, H * 0.5);
  const double Rmax = std::max(1e-6, std::max(a_m, b_m));
  const double px_per_m = 0.45 * std::min(W, H) / Rmax;

  auto toPix = [&](const Eigen::Vector2d &pW) -> cv::Point2d {
    const Eigen::Vector2d d = pW - cW_xy;
    return cv::Point2d(cpx.x + d.x() * px_per_m, cpx.y - d.y() * px_per_m);
  };

  // Ellipse curve in world: p(phi) = c + R_WP * [a cos(phi), b sin(phi)]
  std::vector<cv::Point> poly;
  const int Ns = 120;
  poly.reserve(Ns + 1);
  for (int i = 0; i <= Ns; ++i) {
    const double s = static_cast<double>(i) / static_cast<double>(Ns);
    const double phi = 2.0 * M_PI * s;
    const Eigen::Vector2d qP(a_m * std::cos(phi), b_m * std::sin(phi));
    const Eigen::Vector2d pW = cW_xy + R_WP * qP;
    const cv::Point2d pt = toPix(pW);
    poly.emplace_back((int)std::lround(pt.x), (int)std::lround(pt.y));
  }
  for (size_t i = 1; i < poly.size(); ++i) {
    cv::line(img, poly[i - 1], poly[i], cv::Scalar(0, 0, 0), 2);
  }

  // Heading arrow
  {
    const Eigen::Vector2d p1 = cW_xy + heading_unit_xy * Rmax;
    cv::arrowedLine(img, toPix(cW_xy), toPix(p1), cv::Scalar(0, 180, 0), 2,
                    cv::LINE_AA, 0, 0.2);
  }

  // Candidates
  for (size_t i = 0; i < cand_xy.size(); ++i) {
    const cv::Point2d pt = toPix(cand_xy[i]);
    cv::circle(img, pt, 4, cv::Scalar(180, 0, 0), -1);
    cv::putText(img, std::to_string(i), cv::Point((int)pt.x + 6, (int)pt.y - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    cv::putText(img, std::to_string(i), cv::Point((int)pt.x + 6, (int)pt.y - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
  }

  // Center
  cv::circle(img, cpx, 6, cv::Scalar(0, 128, 255), -1);
  return img;
}

} // namespace

class RingGeneratorService : public rclcpp::Node {
public:
  RingGeneratorService()
      : Node("ring_generator_node"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    // ---- Params ----
    // Frames
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    base_frame_ =
        declare_parameter<std::string>("base_frame", "pursuer/base_link");
    camera_frame_ =
        declare_parameter<std::string>("camera_frame", "pursuer/camera_link");
    optical_frame_ = declare_parameter<std::string>(
        "optical_frame", "pursuer/camera_depth_optical_frame");
    target_source_ = declare_parameter<std::string>(
        "target_source",
        "prediction_state"); // "prediction_state" | "estimate_state" | "tf"
    target_frame_ =
        declare_parameter<std::string>("target_frame", "target/base_link");
    tf_lookup_timeout_sec_ =
        declare_parameter<double>("tf_lookup_timeout_sec", 0.2);
    // On hardware, token times can be ahead of TF; optionally use latest TF.
    use_latest_tf_ = declare_parameter<bool>("use_latest_tf", false);
    // Topics
    step_stamp_topic_ = declare_parameter<std::string>(
        "input_step_stamp_topic", "/experiment/step_stamp");
    prediction_topic_ = declare_parameter<std::string>(
        "input_pred_state_topic", "/nbv/target_prediction_state");
    estimation_topic_ = declare_parameter<std::string>(
        "input_est_state_topic", "/nbv/target_estimation_state");
    optical_cand_topic_ = declare_parameter<std::string>(
        "output_optical_topic", "/nbv/candidates_world_optical");
    base_cand_topic_ = declare_parameter<std::string>(
        "output_base_topic", "/nbv/candidates_world_base");
    service_name_ = declare_parameter<std::string>("service_name",
                                                   "/nbv/generate/candidates");
    // Ring params
    num_candidates_ = declare_parameter<int>("num_candidates", 8);
    base_height_m_ = declare_parameter<double>("base_height_m", 0.0);
    v_min_ = declare_parameter<double>("v_min", 0.05);
    // Dynamic Radius Params
    r_off_m_ = declare_parameter<double>("r_off_m", 2.0);
    kappa_ = declare_parameter<double>("kappa", 2.0);
    candidate_mode_ = declare_parameter<std::string>("candidate_mode",
                                                     "ring"); // ring|ellipse

    reachability_max_dist_m_ =
        declare_parameter<double>("reachability_max_dist_m", -1.0);
    reachability_max_yaw_rad_ =
        declare_parameter<double>("reachability_max_yaw_rad", -1.0);
    // Allow reachability filtering even in static (TF) target mode.
    // (distance-only filtering; yaw filtering is ignored in TF mode)
    enable_reachability_in_tf_mode_ =
        declare_parameter<bool>("enable_reachability_in_tf_mode", false);

    // Debug visualization
    debug_publish_markers_ =
        declare_parameter<bool>("debug_publish_markers", true);
    debug_publish_image_ = declare_parameter<bool>("debug_publish_image", true);
    debug_save_image_png_ =
        declare_parameter<bool>("debug_save_image_png", false);
    debug_dir_ =
        declare_parameter<std::string>("debug_dir", "debug/mua_nbv_planner");
    markers_topic_ = declare_parameter<std::string>(
        "output_markers_topic", "/nbv/debug/candidates_markers");
    image_topic_ = declare_parameter<std::string>("output_debug_image_topic",
                                                  "/nbv/debug/candidates_plot");

    // ---- Subscriber / Publisher / Service ----
    // Subscribers
    sub_step_stamp_ = create_subscription<builtin_interfaces::msg::Time>(
        step_stamp_topic_, rclcpp::QoS(10).reliable(),
        [this](const builtin_interfaces::msg::Time::SharedPtr msg) {
          if (!msg)
            return;
          std::lock_guard<std::mutex> lk(mtx_);
          step_stamp_ = rclcpp::Time(msg->sec, msg->nanosec,
                                     get_clock()->get_clock_type());
          have_step_stamp_ = true;
          if (!have_last_step_logged_ || step_stamp_ != last_step_logged_) {
            last_step_logged_ = step_stamp_;
            have_last_step_logged_ = true;
            RCLCPP_INFO(get_logger(), "🟣 TOKEN recv: %s",
                        stampStr(step_stamp_).c_str());
          }
        });
    sub_pred_state_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        prediction_topic_, rclcpp::QoS(10).reliable(),
        std::bind(&RingGeneratorService::onPredictionState, this,
                  std::placeholders::_1));
    sub_est_state_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        estimation_topic_, rclcpp::QoS(10).reliable(),
        std::bind(&RingGeneratorService::onEstimationState, this,
                  std::placeholders::_1));
    // Publishers
    pub_optical_candidates_ = create_publisher<geometry_msgs::msg::PoseArray>(
        optical_cand_topic_, 10);
    pub_base_candidates_ =
        create_publisher<geometry_msgs::msg::PoseArray>(base_cand_topic_, 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        markers_topic_, 1);
    pub_image_ = create_publisher<sensor_msgs::msg::Image>(image_topic_, 1);
    // Service
    srv_ = create_service<std_srvs::srv::Trigger>(
        service_name_, std::bind(&RingGeneratorService::onGenerate, this,
                                 std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(
        get_logger(),
        "----------------------------------------------------------------");
    RCLCPP_INFO(get_logger(), "📡 Listening to: %s, %s",
                step_stamp_topic_.c_str(), prediction_topic_.c_str());
    RCLCPP_INFO(get_logger(), "📡 (estimate_state) Listening to: %s",
                estimation_topic_.c_str());
    RCLCPP_INFO(get_logger(), "📢 Publishing to: %s, %s",
                optical_cand_topic_.c_str(), base_cand_topic_.c_str());
    RCLCPP_INFO(get_logger(), "🔧 Service: %s", service_name_.c_str());
    RCLCPP_INFO(get_logger(),
                "📊 num_candidates=%d, base_height_m=%.3f, v_min=%.3f",
                num_candidates_, base_height_m_, v_min_);
    RCLCPP_INFO(get_logger(), "📊 r_off_m=%.3f, kappa=%.3f", r_off_m_, kappa_);
    RCLCPP_INFO(get_logger(), "📊 candidate_mode='%s'",
                candidate_mode_.c_str());
    RCLCPP_INFO(get_logger(), "📊 use_latest_tf=%s",
                use_latest_tf_ ? "true" : "false");
    RCLCPP_INFO(get_logger(),
                "📊 reachability: max_dist_m=%.3f max_yaw_rad=%.3f",
                reachability_max_dist_m_, reachability_max_yaw_rad_);
    RCLCPP_INFO(
        get_logger(),
        "----------------------------------------------------------------");
  }

private:
  void
  onPredictionState(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    if (msg->data.size() < 22)
      return;

    // Decode Stamp (Simple cast approach)
    int64_t sec = static_cast<int64_t>(std::round(msg->data[0]));
    uint32_t nsec = static_cast<uint32_t>(std::round(msg->data[1]));

    // Decode State
    Eigen::Vector4d mu;
    mu << msg->data[2], msg->data[3], msg->data[4], msg->data[5];

    Eigen::Matrix4d S;
    int k = 6;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        S(r, c) = msg->data[k++];

    // Update Cache
    {
      std::lock_guard<std::mutex> lk(mtx_);
      pred_stamp_ = rclcpp::Time(static_cast<int32_t>(sec), nsec,
                                 get_clock()->get_clock_type());
      pred_mu_ = mu;
      pred_Sigma_ = S;
      have_pred_ = true;
    }
  }

  void
  onEstimationState(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    if (msg->data.size() < 22)
      return;

    int64_t sec = static_cast<int64_t>(std::round(msg->data[0]));
    uint32_t nsec = static_cast<uint32_t>(std::round(msg->data[1]));

    Eigen::Vector4d mu;
    mu << msg->data[2], msg->data[3], msg->data[4], msg->data[5];

    Eigen::Matrix4d S;
    int k = 6;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        S(r, c) = msg->data[k++];

    {
      std::lock_guard<std::mutex> lk(mtx_);
      est_stamp_ = rclcpp::Time(static_cast<int32_t>(sec), nsec,
                                get_clock()->get_clock_type());
      est_mu_ = mu;
      est_Sigma_ = S;
      have_est_ = true;
    }
  }

  void onGenerate(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                  std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    rclcpp::Time step_stamp_local(0, 0, get_clock()->get_clock_type());
    rclcpp::Time pred_stamp_local(0, 0, get_clock()->get_clock_type());
    Eigen::Vector4d pred_mu_local;
    Eigen::Matrix4d pred_Sigma_local;
    rclcpp::Time est_stamp_local(0, 0, get_clock()->get_clock_type());
    Eigen::Vector4d est_mu_local;
    Eigen::Matrix4d est_Sigma_local;

    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!have_step_stamp_) {
        res->success = false;
        res->message = "No step token received yet on " + step_stamp_topic_;
        return;
      }
      step_stamp_local = step_stamp_;
      pred_stamp_local = pred_stamp_;
      pred_mu_local = pred_mu_;
      pred_Sigma_local = pred_Sigma_;
      est_stamp_local = est_stamp_;
      est_mu_local = est_mu_;
      est_Sigma_local = est_Sigma_;
    }

    RCLCPP_INFO(
        get_logger(),
        "🟣 TOKEN generate: step=%s target_source=%s pred=%s have_pred=%s "
        "est=%s have_est=%s",
        stampStr(step_stamp_local).c_str(), target_source_.c_str(),
        stampStr(pred_stamp_local).c_str(), have_pred_ ? "true" : "false",
        stampStr(est_stamp_local).c_str(), have_est_ ? "true" : "false");

    if (target_source_ == "prediction_state") {
      if (!have_pred_) {
        res->success = false;
        res->message = "No prediction received yet on " + prediction_topic_;
        return;
      }
      // Require prediction to be at/after the step token (no stale reuse).
      if (pred_stamp_local < step_stamp_local) {
        res->success = false;
        res->message = "Prediction stamp is older than step token; waiting for "
                       "predictor update.";
        return;
      }
    } else if (target_source_ == "estimate_state") {
      if (!have_est_) {
        res->success = false;
        res->message = "No estimate received yet on " + estimation_topic_;
        return;
      }
      // Require estimate to be at/after the step token (no stale reuse).
      if (est_stamp_local < step_stamp_local) {
        res->success = false;
        res->message = "Estimate stamp is older than step token; waiting for "
                       "estimator update.";
        return;
      }
    } else if (target_source_ != "tf") {
      res->success = false;
      res->message = "Invalid target_source: " + target_source_;
      return;
    }

    // 1. Get Static TFs (CameraLink/BaseLink/Optical)
    // We assume the camera is fixed on the robot, and Optical is fixed to
    // CameraLink.
    geometry_msgs::msg::TransformStamped tf_B_C;
    geometry_msgs::msg::TransformStamped tf_C_O;
    try {
      if (!tf_buffer_.canTransform(base_frame_, camera_frame_,
                                   tf2::TimePointZero)) {
        res->success = false;
        res->message =
            "Waiting for TF: " + base_frame_ + " -> " + camera_frame_;
        return;
      }
      // Get Base <- Camera
      tf_B_C = tf_buffer_.lookupTransform(base_frame_, camera_frame_,
                                          tf2::TimePointZero);

      if (!tf_buffer_.canTransform(camera_frame_, optical_frame_,
                                   tf2::TimePointZero)) {
        res->success = false;
        res->message =
            "Waiting for TF: " + camera_frame_ + " -> " + optical_frame_;
        return;
      }
      // Get CameraLink <- Optical
      tf_C_O = tf_buffer_.lookupTransform(camera_frame_, optical_frame_,
                                          tf2::TimePointZero);
    } catch (const tf2::TransformException &e) {
      res->success = false;
      res->message = std::string("TF Error: ") + e.what();
      return;
    }
    Eigen::Isometry3d T_B_C =
        isoFromTransform(tf_B_C.transform); // Base <- Camera
    Eigen::Isometry3d T_C_O =
        isoFromTransform(tf_C_O.transform); // Camera <- Optical

    // 2. Determine Ring Center & Radius (PredictionState OR TF target frame)
    Eigen::Isometry3d T_W_P = Eigen::Isometry3d::Identity();

    double radius_final = r_off_m_;
    mua_nbv_planner::EllipseAxes ellipse_axes;
    ellipse_axes.a_m = std::max(0.1, r_off_m_);
    ellipse_axes.b_m = std::max(0.1, r_off_m_);
    double yaw_P = 0.0;
    if (target_source_ ==
        "prediction_state") { // DYNAMIC target: use prediction state
      // Position (Prediction Mean)
      T_W_P.translation() << pred_mu_local(0), pred_mu_local(1), 0.0;

      // Heading (Velocity direction)
      const double vx = pred_mu_local(2);
      const double vy = pred_mu_local(3);
      double speed = std::hypot(vx, vy);

      double yaw = last_yaw_;
      if (speed >= v_min_) {
        yaw = std::atan2(vy, vx);
        last_yaw_ = yaw; // Cache for next time if stopped
      }
      yaw_P = yaw;
      T_W_P.linear() =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      // Uncertainty aware radius / ellipse axes (planar position covariance)
      Eigen::Matrix2d Pxy = pred_Sigma_local.block<2, 2>(0, 0);
      Pxy = 0.5 * (Pxy + Pxy.transpose());

      // Ring radius (legacy): r = r_off + kappa * sqrt(lambda_max(Pxy))
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Pxy);
      double max_eigenval =
          std::max(0.0, es.eigenvalues()(1)); // Max uncertainty axis
      double r_unc = kappa_ * std::sqrt(max_eigenval);
      radius_final = r_off_m_ + r_unc;

      // Ellipse axes (heading-aligned): a,b derived from diag(R^T Pxy R)
      ellipse_axes = mua_nbv_planner::headingAlignedEllipseAxes(
          Pxy, yaw_P, r_off_m_, kappa_, /*min_axis_m=*/0.1);
    } else if (target_source_ == "estimate_state") {
      // Baseline: use last estimated state (NOT TF ground truth, NOT one-step prediction).
      T_W_P.translation() << est_mu_local(0), est_mu_local(1), 0.0;

      const double vx = est_mu_local(2);
      const double vy = est_mu_local(3);
      double speed = std::hypot(vx, vy);

      double yaw = last_yaw_;
      if (speed >= v_min_) {
        yaw = std::atan2(vy, vx);
        last_yaw_ = yaw;
      }
      yaw_P = yaw;
      T_W_P.linear() =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      Eigen::Matrix2d Pxy = est_Sigma_local.block<2, 2>(0, 0);
      Pxy = 0.5 * (Pxy + Pxy.transpose());

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Pxy);
      double max_eigenval =
          std::max(0.0, es.eigenvalues()(1)); // Max uncertainty axis
      double r_unc = kappa_ * std::sqrt(max_eigenval);
      radius_final = r_off_m_ + r_unc;

      ellipse_axes = mua_nbv_planner::headingAlignedEllipseAxes(
          Pxy, yaw_P, r_off_m_, kappa_, /*min_axis_m=*/0.1);
    } else { // STATIC target: center from TF world<-target_frame
      try {
        const auto timeout = tf2::durationFromSec(tf_lookup_timeout_sec_);
        geometry_msgs::msg::TransformStamped tf_W_T;
        if (use_latest_tf_) {
          tf_W_T = tf_buffer_.lookupTransform(world_frame_, target_frame_,
                                              tf2::TimePointZero, timeout);
        } else {
          tf_W_T = tf_buffer_.lookupTransform(world_frame_, target_frame_,
                                              step_stamp_local, timeout);
        }
        Eigen::Isometry3d T_W_T = isoFromTransform(tf_W_T.transform);
        T_W_P = T_W_T; // treat target as ring center frame
      } catch (const tf2::TransformException &e) {
        res->success = false;
        res->message = std::string("TF error (world<-target): ") + e.what();
        return;
      }
    }

    // 3. Generate Candidates
    // Generate view poses for the ROBOT BASE in the local frame P: (P <- Base).
    // CameraLink/Optical poses are derived from static TF (Base <- Optical).
    std::vector<Eigen::Isometry3d> views_P_B;
    const bool want_ellipse = (candidate_mode_ == "ellipse") &&
                              ((target_source_ == "prediction_state") ||
                               (target_source_ == "estimate_state"));
    if (want_ellipse) {
      mua_nbv_planner::EllipseParams ep;
      ep.center_x = 0.0;
      ep.center_y = 0.0;
      ep.a_m = std::max(0.1, ellipse_axes.a_m);
      ep.b_m = std::max(0.1, ellipse_axes.b_m);
      ep.num_points = num_candidates_;
      mua_nbv_planner::EllipseGenerator gen(ep);
      views_P_B = gen.generate(base_height_m_);
    } else {
      if (candidate_mode_ != "ring" && candidate_mode_ != "ellipse") {
        RCLCPP_WARN(
            get_logger(),
            "Unknown candidate_mode='%s' (expected 'ring' or 'ellipse'); "
            "falling back to ring.",
            candidate_mode_.c_str());
      }
      mua_nbv_planner::RingParams rp;
      rp.center_x = 0.0;
      rp.center_y = 0.0; // Local to T_W_P
      rp.radius_m = std::max(0.1, radius_final);
      rp.num_points = num_candidates_;
      mua_nbv_planner::RingGenerator gen(rp);
      views_P_B = gen.generate(base_height_m_);
    }

    // 3.5 Reachability filtering.
    // Filter candidates based on robot pose (testbed: typically latest TF).
    const bool reachability_requested =
        (reachability_max_dist_m_ > 0.0) || (reachability_max_yaw_rad_ > 0.0);
    const bool reachability_allowed_for_mode =
        (target_source_ == "prediction_state") ||
        (target_source_ == "estimate_state") ||
        ((target_source_ == "tf") && enable_reachability_in_tf_mode_);
    const bool enable_reachability =
        reachability_requested && reachability_allowed_for_mode;
    Eigen::Vector2d p_robot_xy(0.0, 0.0);
    double yaw_robot = 0.0;
    if (enable_reachability) {
      try {
        const auto timeout = tf2::durationFromSec(tf_lookup_timeout_sec_);
        geometry_msgs::msg::TransformStamped tf_W_R;
        if (use_latest_tf_) {
          tf_W_R = tf_buffer_.lookupTransform(world_frame_, base_frame_,
                                              tf2::TimePointZero, timeout);
        } else {
          tf_W_R = tf_buffer_.lookupTransform(world_frame_, base_frame_,
                                              step_stamp_local, timeout);
        }
        const Eigen::Isometry3d T_W_R = isoFromTransform(tf_W_R.transform);
        p_robot_xy << T_W_R.translation().x(), T_W_R.translation().y();
        yaw_robot = yawFromRotation(T_W_R.rotation());
      } catch (const tf2::TransformException &e) {
        res->success = false;
        res->message =
            std::string("TF error (world<-robot) for reachability: ") +
            e.what();
        return;
      }
    }

    // 4. Transform to World Frame & Robot Base Frame
    geometry_msgs::msg::PoseArray msg_base_out;
    msg_base_out.header.stamp = step_stamp_local; // Match step token time
    msg_base_out.header.frame_id = world_frame_;  // World frame

    geometry_msgs::msg::PoseArray msg_opt_out = msg_base_out;

    const Eigen::Isometry3d T_B_O = T_B_C * T_C_O; // Base <- Optical
    std::vector<Eigen::Isometry3d> views_W_B;
    views_W_B.reserve(views_P_B.size());
    for (const auto &T_P_B : views_P_B) {
      // T_W_B = T_W_P * T_P_B  (World <- Base)
      Eigen::Isometry3d T_W_B = T_W_P * T_P_B;

      if (enable_reachability) {
        const Eigen::Vector2d p_cand_xy(T_W_B.translation().x(),
                                        T_W_B.translation().y());
        const double dist = (p_cand_xy - p_robot_xy).norm();
        if (reachability_max_dist_m_ > 0.0 && dist > reachability_max_dist_m_) {
          continue;
        }
        // In static TF mode, ignore yaw reachability (distance-only is safer and
        // more predictable on hardware).
        if (reachability_max_yaw_rad_ > 0.0 && target_source_ != "tf") {
          const double yaw_cand = yawFromRotation(T_W_B.rotation());
          const double dyaw =
              mua_nbv_planner::angleDiffAbsRad(yaw_cand, yaw_robot);
          if (dyaw > reachability_max_yaw_rad_) {
            continue;
          }
        }
      }

      // T_W_O = T_W_B * T_B_O  (World <- Optical)
      Eigen::Isometry3d T_W_O = T_W_B * T_B_O;
      msg_opt_out.poses.push_back(poseFromIsometry(T_W_O));
      msg_base_out.poses.push_back(poseFromIsometry(T_W_B));
      views_W_B.push_back(T_W_B);
    }

    if (views_W_B.empty()) {
      res->success = false;
      res->message =
          "No reachable candidates (reachability constraints too strict).";
      return;
    }
    pub_optical_candidates_->publish(msg_opt_out);
    pub_base_candidates_->publish(msg_base_out);

    // 5. Debug visualization: MarkerArray + plot image
    if (debug_publish_markers_ || debug_publish_image_ ||
        debug_save_image_png_) {
      const Eigen::Vector3d cW(T_W_P.translation().x(), T_W_P.translation().y(),
                               base_height_m_);
      const Eigen::Vector2d heading_unit(std::cos(yaw_P), std::sin(yaw_P));

      // For ellipse marker/image, use the effective a,b (ring treated as
      // a=b=radius).
      const double a_eff = want_ellipse ? std::max(0.1, ellipse_axes.a_m)
                                        : std::max(0.1, radius_final);
      const double b_eff = want_ellipse ? std::max(0.1, ellipse_axes.b_m)
                                        : std::max(0.1, radius_final);

      if (debug_publish_markers_) {
        visualization_msgs::msg::MarkerArray ma;
        ma.markers.push_back(
            makeCenterMarker(world_frame_, step_stamp_local, cW));
        if (enable_reachability && reachability_max_dist_m_ > 0.0) {
          // Semi-transparent reachability disk (as a cylinder) around robot.
          ma.markers.push_back(makeReachabilityDiskMarker(
              world_frame_, step_stamp_local, p_robot_xy,
              /*z_center_m=*/0.025, /*radius_m=*/reachability_max_dist_m_,
              /*height_m=*/0.05, /*alpha=*/0.15));
        }
        ma.markers.push_back(makeHeadingArrowMarker(
            world_frame_, step_stamp_local, cW, heading_unit,
            /*length_m=*/std::max(a_eff, b_eff)));
        ma.markers.push_back(
            makeEllipseLineMarker(world_frame_, step_stamp_local, T_W_P,
                                  base_height_m_, a_eff, b_eff));
        ma.markers.push_back(makeCandidatePointsMarker(
            world_frame_, step_stamp_local, views_W_B));
        pub_markers_->publish(ma);
      }

      if (debug_publish_image_ || debug_save_image_png_) {
        std::vector<Eigen::Vector2d> cand_xy;
        cand_xy.reserve(views_W_B.size());
        for (const auto &T_W_B : views_W_B) {
          cand_xy.emplace_back(T_W_B.translation().x(),  T_W_B.translation().y());
        }

        Eigen::Matrix2d R_WP;
        R_WP << std::cos(yaw_P), -std::sin(yaw_P), std::sin(yaw_P),
            std::cos(yaw_P);
        const cv::Mat img = renderCandidateEllipseImage(
            Eigen::Vector2d(cW.x(), cW.y()), heading_unit, cand_xy, R_WP, a_eff,
            b_eff);
        if (debug_publish_image_) {
          std_msgs::msg::Header hdr;
          hdr.stamp = step_stamp_local;
          hdr.frame_id = world_frame_;
          auto msg = cv_bridge::CvImage(hdr, "bgr8", img).toImageMsg();
          pub_image_->publish(*msg);
        }
        if (debug_save_image_png_) {
          const std::string fname =
              debug_dir_ + "/candidates_" + stampStr(step_stamp_local) + ".png";
          try {
            cv::imwrite(fname, img);
          } catch (const std::exception &e) {
            RCLCPP_WARN(get_logger(), "Failed to write debug image '%s': %s",
                        fname.c_str(), e.what());
          }
        }
      }
    }

    // Consume the step token only after publish succeeds (so retry is possible
    // on failure).
    {
      std::lock_guard<std::mutex> lk(mtx_);
      have_step_stamp_ = false;
    }
    res->success = true;
    res->message =
        "Generated " + std::to_string(views_W_B.size()) + " candidates.";
  }

  // Member variables
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string world_frame_, base_frame_, camera_frame_, optical_frame_;
  std::string target_source_;
  std::string target_frame_;
  double tf_lookup_timeout_sec_{0.2};
  bool use_latest_tf_{false};
  std::string step_stamp_topic_, prediction_topic_, estimation_topic_,
      optical_cand_topic_, base_cand_topic_;
  std::string service_name_;

  int num_candidates_;
  double base_height_m_{0.0}, v_min_;
  double r_off_m_, kappa_;
  std::string candidate_mode_;
  double reachability_max_dist_m_{-1.0};
  double reachability_max_yaw_rad_{-1.0};
  bool enable_reachability_in_tf_mode_{false};

  bool debug_publish_markers_{true};
  bool debug_publish_image_{true};
  bool debug_save_image_png_{false};
  std::string debug_dir_;
  std::string markers_topic_, image_topic_;

  // Cache for the latest prediction / estimate-state
  std::mutex mtx_;

  bool have_step_stamp_{false};
  rclcpp::Time step_stamp_;

  bool have_pred_ = false;
  rclcpp::Time pred_stamp_;
  Eigen::Vector4d pred_mu_;
  Eigen::Matrix4d pred_Sigma_;

  bool have_est_ = false;
  rclcpp::Time est_stamp_;
  Eigen::Vector4d est_mu_;
  Eigen::Matrix4d est_Sigma_;

  double last_yaw_ = 0.0;
  bool have_last_step_logged_{false};
  rclcpp::Time last_step_logged_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<builtin_interfaces::msg::Time>::SharedPtr
      sub_step_stamp_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
      sub_pred_state_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
      sub_est_state_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr
      pub_optical_candidates_,
      pub_base_candidates_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      pub_markers_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RingGeneratorService>());
  rclcpp::shutdown();
  return 0;
}