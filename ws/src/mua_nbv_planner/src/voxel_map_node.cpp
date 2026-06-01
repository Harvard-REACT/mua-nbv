#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <string>
#include <vector>

#include <condition_variable>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <tf2/time.h>

#include <rcl/time.h>
#include <rclcpp/rclcpp.hpp>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <octomap/ColorOcTree.h>
#include <octomap/OcTreeKey.h>

#include <Eigen/Dense>

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

inline bool finite3(double x, double y, double z) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

inline double wrapAngleRad(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

inline double yawFromQuat(const geometry_msgs::msg::Quaternion &q_msg) {
  // yaw from quaternion (x,y,z,w)
  const double x = q_msg.x;
  const double y = q_msg.y;
  const double z = q_msg.z;
  const double w = q_msg.w;
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

inline double angleDiffAbsRad(double a, double b) {
  return std::abs(wrapAngleRad(a - b));
}

inline Eigen::Vector3d clampVec(const Eigen::Vector3d &v,
                                const Eigen::Vector3d &mn,
                                const Eigen::Vector3d &mx) {
  return v.cwiseMax(mn).cwiseMin(mx);
}

inline builtin_interfaces::msg::Time timeMsg(const rclcpp::Time &t) {
  builtin_interfaces::msg::Time m;
  const int64_t ns = t.nanoseconds();
  m.sec = static_cast<int32_t>(ns / 1000000000LL);
  m.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return m;
}

inline bool intersectRayAABB(const Eigen::Vector3d &o, const Eigen::Vector3d &d,
                             const Eigen::Vector3d &bmin, const Eigen::Vector3d &bmax, 
                             double &t_enter,  double &t_exit) 
{
  t_enter = 0.0;
  t_exit = std::numeric_limits<double>::infinity();

  for (int axis = 0; axis < 3; ++axis) {
    const double origin = o[axis];
    const double dir = d[axis];
    const double mn = bmin[axis];
    const double mx = bmax[axis];

    if (std::abs(dir) < 1e-12) {
      if (origin < mn || origin > mx)
        return false;
      continue;
    }

    double t0 = (mn - origin) / dir;
    double t1 = (mx - origin) / dir;
    if (t0 > t1)
      std::swap(t0, t1);

    t_enter = std::max(t_enter, t0);
    t_exit = std::min(t_exit, t1);

    if (t_enter > t_exit)
      return false;
  }
  return true;
}

sensor_msgs::msg::PointCloud2 makeCloudXYZ(const std::string &frame_id, 
                                           const rclcpp::Time &stamp,
                                           const std::vector<Eigen::Vector3d> &pts) 
{
  // Create a PointCloud2 message with XYZ points
  sensor_msgs::msg::PointCloud2 out;
  out.header.frame_id = frame_id;
  out.header.stamp = timeMsg(stamp);

  sensor_msgs::PointCloud2Modifier mod(out);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(pts.size());

  sensor_msgs::PointCloud2Iterator<float> it_x(out, "x");
  sensor_msgs::PointCloud2Iterator<float> it_y(out, "y");
  sensor_msgs::PointCloud2Iterator<float> it_z(out, "z");

  for (size_t i = 0; i < pts.size(); ++i, ++it_x, ++it_y, ++it_z) {
    *it_x = static_cast<float>(pts[i].x());
    *it_y = static_cast<float>(pts[i].y());
    *it_z = static_cast<float>(pts[i].z());
  }
  return out;
}

} // namespace

class VoxelMapService : public rclcpp::Node {
public:
  VoxelMapService()
      : Node("voxel_map_node"),
        clock_type_(this->get_clock()->get_clock_type()),
        tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) 
  {
    // --- Params ---
    // Frames
    target_frame_ = declare_parameter<std::string>("target_frame", "target/est_base_link");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "pursuer/camera_depth_optical_frame");
    // Topics
    step_stamp_topic_ = declare_parameter<std::string>("input_step_stamp_topic", "/experiment/step_stamp");
    use_cam_pose_topic_ = declare_parameter<bool>("use_cam_pose_topic", false);
    cam_pose_topic_ = declare_parameter<std::string>("input_cam_pose_topic", "/experiment/captured_cam_pose");
    points_topic_ = declare_parameter<std::string>("input_points_topic", "/target/points");
    occupied_topic_ = declare_parameter<std::string>("output_occupied_topic", "/nbv/voxels/occupied");
    frontier_topic_ = declare_parameter<std::string>("output_frontier_topic", "/nbv/voxels/frontier");
    bbox_topic_ = declare_parameter<std::string>("output_bbox_topic", "/nbv/voxels/bbox");
    // Service
    update_service_ = declare_parameter<std::string>("update_service", "/nbv/update/voxels");
    // Timeouts
    tf_lookup_timeout_sec_ = declare_parameter<double>("tf_lookup_timeout_sec", 0.05);
    // Buffers
    cloud_buf_max_ = static_cast<size_t>(declare_parameter<int>("cloud_buf_max", 50));
    pose_buf_max_ = static_cast<size_t>(declare_parameter<int>("pose_buf_max", 50));
    wait_for_inputs_sec_ = declare_parameter<double>("wait_for_inputs_sec", 0.5);
    // Skip update option (still publish outputs, but don't modify internal map)
    skip_update_if_same_view_ = declare_parameter<bool>("skip_update_if_same_view", false);
    same_view_pos_tol_m_ = declare_parameter<double>("same_view_pos_tol_m", 1e-3);
    same_view_yaw_tol_rad_ = declare_parameter<double>("same_view_yaw_tol_rad", 1e-3);
    // Map Params
    bbx_half_side_m_ = declare_parameter<double>("bbx_half_side_m", 0.7);
    bbx_z_min_m_ = declare_parameter<double>("bbx_z_min_m", 0.02);
    bbx_z_max_m_ = declare_parameter<double>("bbx_z_max_m", 1.5);
    res_ = declare_parameter<double>("voxel_resolution_m", 0.05);
    // Raycasting Params
    ray_stride_px_ = declare_parameter<int>("ray_stride_px", 16);
    point_stride_ = declare_parameter<int>("point_stride", 3);
    max_ray_range_m_ = declare_parameter<double>("max_ray_range_m", 3.0);
    max_steps_per_ray_ = declare_parameter<int>("max_steps_per_ray", 0);
    // Intrinsics
    image_width_ = declare_parameter<int>("image_width", 640);
    image_height_ = declare_parameter<int>("image_height", 480);
    fx_ = declare_parameter<double>("fx", 554.38);
    fy_ = declare_parameter<double>("fy", 554.38);
    cx_ = declare_parameter<double>("cx", 320.0);
    cy_ = declare_parameter<double>("cy", 240.0);

    // Auto parameterization (geometry-driven defaults)
    auto_params_enable_ = declare_parameter<bool>("auto_params_enable", false);
    auto_voxel_resolution_enable_ = declare_parameter<bool>("auto_voxel_resolution_enable", false);
    auto_nominal_depth_m_ = declare_parameter<double>("auto_nominal_depth_m", 1.0);
    auto_target_voxels_per_diag_ = declare_parameter<int>("auto_target_voxels_per_diag", 45);
    auto_voxel_res_min_m_ = declare_parameter<double>("auto_voxel_res_min_m", 0.01);
    auto_voxel_res_max_m_ = declare_parameter<double>("auto_voxel_res_max_m", 0.05);
    auto_target_pixels_per_voxel_ = declare_parameter<double>("auto_target_pixels_per_voxel", 1.5);
    auto_min_ray_stride_px_ = declare_parameter<int>("auto_min_ray_stride_px", 2);
    auto_max_ray_stride_px_ = declare_parameter<int>("auto_max_ray_stride_px", 12);
    auto_point_stride_min_ = declare_parameter<int>("auto_point_stride_min", 1);
    auto_point_stride_max_ = declare_parameter<int>("auto_point_stride_max", 6);
    auto_point_stride_ratio_ = declare_parameter<double>("auto_point_stride_ratio", 0.5);
    auto_max_ray_range_scale_ = declare_parameter<double>("auto_max_ray_range_scale", 1.3);
    auto_max_steps_margin_ = declare_parameter<int>("auto_max_steps_margin", 8);

    // Bounding Box
    bbx_min_ = Eigen::Vector3d(-bbx_half_side_m_, -bbx_half_side_m_, bbx_z_min_m_);
    bbx_max_ = Eigen::Vector3d(+bbx_half_side_m_, +bbx_half_side_m_, bbx_z_max_m_);

    // Optional auto voxel resolution at startup (before map creation)
    if (auto_params_enable_ && auto_voxel_resolution_enable_) {
      const double bbx_diag = (bbx_max_ - bbx_min_).norm();
      const double nominal_d = std::max(0.2, auto_nominal_depth_m_);
      const double res_from_image = 1.5 * nominal_d / std::max(1.0, fx_);
      const double res_from_geom = bbx_diag / std::max(10, auto_target_voxels_per_diag_);
      const double auto_res = std::max(res_from_image, res_from_geom);
      res_ = std::clamp(auto_res, auto_voxel_res_min_m_, auto_voxel_res_max_m_);
    }

    // Init Octomap
    map_ = std::make_unique<octomap::ColorOcTree>(res_);
    map_->useBBXLimit(true);
    map_->setBBXMin(octomap::point3d(bbx_min_.x(), bbx_min_.y(), bbx_min_.z()));
    map_->setBBXMax(octomap::point3d(bbx_max_.x(), bbx_max_.y(), bbx_max_.z()));
    initGridOnce();

    // Initialize time members with correct clock type
    last_processed_stamp_ = rclcpp::Time(0, 0, clock_type_);
    required_stamp_ = rclcpp::Time(0, 0, clock_type_);
    last_token_logged_ = rclcpp::Time(0, 0, clock_type_);

    // --- Subscribers / Publishers / Services ---
    // Subscribers
    auto qos_pc = rclcpp::SensorDataQoS().keep_last(5);
    auto qos_token = rclcpp::QoS(1).reliable().transient_local();
    sub_step_stamp_ = create_subscription<builtin_interfaces::msg::Time>(
        step_stamp_topic_, qos_token,
        [this](const builtin_interfaces::msg::Time::SharedPtr msg) {
          if (!msg)
            return;
          std::lock_guard<std::mutex> lk(mtx_);
          required_stamp_ = rclcpp::Time(msg->sec, msg->nanosec, clock_type_);
          have_required_stamp_ = true;
          if (!have_last_token_logged_ ||
              required_stamp_ != last_token_logged_) {
            have_last_token_logged_ = true;
            last_token_logged_ = required_stamp_;
            RCLCPP_INFO(get_logger(), "🟣 TOKEN recv: %s",
                        stampStr(required_stamp_).c_str());
          }
          cv_.notify_all();
        });
    sub_points_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        points_topic_, qos_pc,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          if (!msg)
            return;
          std::lock_guard<std::mutex> lk(mtx_);
          cloud_buf_.push_back(
              {rclcpp::Time(msg->header.stamp, clock_type_), msg});
          while (cloud_buf_.size() > cloud_buf_max_)
            cloud_buf_.pop_front();
          cv_.notify_all();
        });
    if (use_cam_pose_topic_) {
      sub_cam_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          cam_pose_topic_, qos_token,
          [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            if (!msg)
              return;
            std::lock_guard<std::mutex> lk(mtx_);
            pose_buf_.push_back(
                {rclcpp::Time(msg->header.stamp, clock_type_), msg});
            while (pose_buf_.size() > pose_buf_max_)
              pose_buf_.pop_front();
            cv_.notify_all();
          });
    }
    // Publishers
    auto qos = rclcpp::QoS(10).reliable();
    pub_occupied_ = create_publisher<sensor_msgs::msg::PointCloud2>(occupied_topic_, qos);
    pub_frontier_ = create_publisher<sensor_msgs::msg::PointCloud2>(frontier_topic_, qos);
    pub_bbox_ = create_publisher<visualization_msgs::msg::Marker>(bbox_topic_, qos);
    // Service
    srv_update_ = create_service<std_srvs::srv::Trigger>(update_service_,
        std::bind(&VoxelMapService::onUpdateService, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "----------------------------------------------------------------");
    if (use_cam_pose_topic_) {
      RCLCPP_INFO(get_logger(), "📡 Listening to: %s, %s, %s", step_stamp_topic_.c_str(), points_topic_.c_str(), cam_pose_topic_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "📡 Listening to: %s, %s", step_stamp_topic_.c_str(), points_topic_.c_str());
    }
    RCLCPP_INFO(get_logger(), "📢 Publishing to: %s, %s, %s", occupied_topic_.c_str(), frontier_topic_.c_str(), bbox_topic_.c_str());
    RCLCPP_INFO(get_logger(), "🔧 Service: %s", update_service_.c_str());
    if (skip_update_if_same_view_) {
      RCLCPP_INFO(get_logger(), "📊 Skip same view: true, pos_tol=%.4f m, yaw_tol=%.4f rad", same_view_pos_tol_m_, same_view_yaw_tol_rad_);
    }
    RCLCPP_INFO(get_logger(), "📊 Bounding box: %.3f m x %.3f m x %.3f m", bbx_half_side_m_, bbx_half_side_m_, bbx_z_max_m_ - bbx_z_min_m_);
    RCLCPP_INFO(get_logger(), "📊 Bounding box z min and max: %.3f m, %.3f m", bbx_z_min_m_, bbx_z_max_m_);
    RCLCPP_INFO(get_logger(), "📊 Voxel resolution: %.3f m", res_);
    RCLCPP_INFO(get_logger(), "📊 Ray stride: %d px, Point stride: %d", ray_stride_px_, point_stride_);
    RCLCPP_INFO(get_logger(), "📊 Auto params: enable=%s auto_voxel_resolution=%s nominal_depth=%.3f",
                auto_params_enable_ ? "true" : "false",
                auto_voxel_resolution_enable_ ? "true" : "false",
                auto_nominal_depth_m_);
    RCLCPP_INFO(get_logger(), "📊 Max ray range: %.3f m, Max steps per ray: %d", max_ray_range_m_, max_steps_per_ray_);
    RCLCPP_INFO(get_logger(), "📊 Image width: %d px, height: %d px", image_width_, image_height_);
    RCLCPP_INFO(get_logger(), "📊 Focal length and principal point: %.3f px, %.3f px, (%.3f, %.3f) px", fx_, fy_, cx_, cy_);
    RCLCPP_INFO(get_logger(), "----------------------------------------------------------------");
  }

private:
  bool lookupCamPoseInTarget(const rclcpp::Time &stamp, geometry_msgs::msg::PoseStamped &out) {
    const auto timeout = tf2::durationFromSec(tf_lookup_timeout_sec_);
    try {
      auto tf = tf_buffer_.lookupTransform(target_frame_, camera_frame_, stamp, timeout);
      out.header.frame_id = target_frame_;
      out.header.stamp = timeMsg(stamp);
      out.pose.position.x = tf.transform.translation.x;
      out.pose.position.y = tf.transform.translation.y;
      out.pose.position.z = tf.transform.translation.z;
      out.pose.orientation = tf.transform.rotation;
      return true;
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "Camera TF lookup failed %s <- %s: %s", target_frame_.c_str(), camera_frame_.c_str(), e.what());
      return false;
    }
  }

  void initGridOnce() {
    for (double x = bbx_min_.x(); x <= bbx_max_.x(); x += res_) {
      for (double y = bbx_min_.y(); y <= bbx_max_.y(); y += res_) {
        for (double z = bbx_min_.z(); z <= bbx_max_.z(); z += res_) {
          auto *n = map_->updateNode(octomap::point3d(x, y, z), false);
          if (n)
            n->setColor(128, 128, 128); // Initialize as Unknown (Gray)
        }
      }
    }
    map_->updateInnerOccupancy();
  }

  void publishBBX(const rclcpp::Time &stamp) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = target_frame_;
    marker.header.stamp = timeMsg(stamp);
    marker.ns = "voxel_bbox";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    Eigen::Vector3d center = 0.5 * (bbx_min_ + bbx_max_);
    marker.pose.position.x = center.x();
    marker.pose.position.y = center.y();
    marker.pose.position.z = center.z();
    Eigen::Vector3d size = bbx_max_ - bbx_min_;
    marker.scale.x = size.x();
    marker.scale.y = size.y();
    marker.scale.z = size.z();
    marker.color.g = 1.0f;
    marker.color.a = 0.2f;
    pub_bbox_->publish(marker);
  }

  bool insideBBX(const Eigen::Vector3d &p) const {
    return (p.x() >= bbx_min_.x() && p.x() <= bbx_max_.x() &&
            p.y() >= bbx_min_.y() && p.y() <= bbx_max_.y() &&
            p.z() >= bbx_min_.z() && p.z() <= bbx_max_.z());
  }

  int neighbors26(const octomap::OcTreeKey &k, octomap::OcTreeKey out[26]) const {
    int idx = 0;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0)
            continue;
          const int64_t nx = static_cast<int64_t>(k[0]) + dx;
          const int64_t ny = static_cast<int64_t>(k[1]) + dy;
          const int64_t nz = static_cast<int64_t>(k[2]) + dz;
          if (nx < 0 || ny < 0 || nz < 0)
            continue;
          out[idx++] = octomap::OcTreeKey(static_cast<uint32_t>(nx),
                                          static_cast<uint32_t>(ny),
                                          static_cast<uint32_t>(nz));
        }
    return idx;
  }

  inline bool isWhite(const octomap::ColorOcTreeNode *n) const {
    auto c = n->getColor();
    return c.r == 255 && c.g == 255 && c.b == 255; // Ve
  }

  inline bool isGray(const octomap::ColorOcTreeNode *n) const {
    auto c = n->getColor();
    return c.r == 188 && c.g == 188 && c.b == 188; // Vu only
  }

  inline bool isNone(const octomap::ColorOcTreeNode *n) const {
    auto c = n->getColor();
    return c.r == 128 && c.g == 128 && c.b == 128; // Vn
  }

  void onUpdateService(const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_ptr;
    rclcpp::Time stamp(0, 0, clock_type_);
    rclcpp::Time token(0, 0, clock_type_);
    geometry_msgs::msg::PoseStamped::SharedPtr cam_pose_ptr;

    {
      std::unique_lock<std::mutex> lk(mtx_);

      auto findCloudExactLocked = [&](const rclcpp::Time &t)
          -> sensor_msgs::msg::PointCloud2::SharedPtr {
        for (auto it = cloud_buf_.rbegin(); it != cloud_buf_.rend(); ++it) {
          if (it->stamp == t)
            return it->msg;
        }
        return nullptr;
      };

      auto findPoseExactLocked = [&](const rclcpp::Time &t)
          -> geometry_msgs::msg::PoseStamped::SharedPtr {
        for (auto it = pose_buf_.rbegin(); it != pose_buf_.rend(); ++it) {
          if (it->stamp == t)
            return it->msg;
        }
        return nullptr;
      };

      auto ready = [&]() {
        if (!have_required_stamp_)
          return false;
        if (required_stamp_ <= last_processed_stamp_)
          return false;
        // We require the cloud that matches the token exactly.
        if (!findCloudExactLocked(required_stamp_))
          return false;
        if (use_cam_pose_topic_ && !findPoseExactLocked(required_stamp_))
          return false;
        return true;
      };

      if (!ready() && wait_for_inputs_sec_ > 0.0) {
        cv_.wait_for(lk, std::chrono::duration<double>(wait_for_inputs_sec_), ready);
      }
      if (!ready()) {
        res->success = false;
        res->message = "Inputs not ready for voxel update (need token + "
                       "matching cloud + matching cam pose if enabled).";
        return;
      }

      token = required_stamp_;
      stamp = required_stamp_;
      cloud_ptr = findCloudExactLocked(required_stamp_);
      if (use_cam_pose_topic_)
        cam_pose_ptr = findPoseExactLocked(required_stamp_);
    }

    if (!cloud_ptr) {
      res->success = false;
      res->message = "Latest cloud pointer was null.";
      return;
    }

    // Require cloud already in target frame (matches old contract)
    if (cloud_ptr->header.frame_id != target_frame_) {
      res->success = false;
      res->message = "Cloud not in target_frame_. Fix upstream CloudCapturer output frame.";
      return;
    }

    RCLCPP_INFO(get_logger(),
                "🟣 TOKEN voxelize: token=%s cloud_stamp=%s cloud_frame=%s "
                "camera_frame=%s target_frame=%s",
                stampStr(token).c_str(), stampStr(stamp).c_str(),
                cloud_ptr->header.frame_id.c_str(), camera_frame_.c_str(),
                target_frame_.c_str());

    geometry_msgs::msg::PoseStamped cam_pose;
    if (use_cam_pose_topic_) {
      if (!cam_pose_ptr) {
        res->success = false;
        res->message = "Cam pose topic enabled but no matching pose found.";
        return;
      }
      cam_pose = *cam_pose_ptr;
      if (cam_pose.header.frame_id != target_frame_) {
        res->success = false;
        res->message = "Cam pose not in target_frame_. Fix upstream CloudCapturer pose frame.";
        return;
      }
    } else {
      if (!lookupCamPoseInTarget(stamp, cam_pose)) {
        res->success = false;
        res->message = "Failed TF lookup for camera pose in target frame.";
        return;
      }
    }
 
    if (skip_update_if_same_view_ && last_cam_pose_valid_) {
      const Eigen::Vector3d p_prev(last_cam_pose_in_target_.pose.position.x,
                                   last_cam_pose_in_target_.pose.position.y,
                                   last_cam_pose_in_target_.pose.position.z);
      const Eigen::Vector3d p_cur(cam_pose.pose.position.x,
                                  cam_pose.pose.position.y,
                                  cam_pose.pose.position.z);
      const double dp = (p_cur - p_prev).norm();
      const double yaw_prev = yawFromQuat(last_cam_pose_in_target_.pose.orientation);
      const double yaw_cur = yawFromQuat(cam_pose.pose.orientation);
      const double dyaw = angleDiffAbsRad(yaw_cur, yaw_prev);

      const bool same_pos = (same_view_pos_tol_m_ <= 0.0) ? true : (dp <= same_view_pos_tol_m_); 
      const bool same_yaw = (same_view_yaw_tol_rad_ <= 0.0) ? true : (dyaw <= same_view_yaw_tol_rad_);
      RCLCPP_INFO(get_logger(), "🟡 dp=%.4f dyaw=%.4f", dp, dyaw);

      if (same_pos && same_yaw) {
        publishMap(stamp);
        last_processed_stamp_ = stamp;
        {
          std::lock_guard<std::mutex> lk(mtx_);
          have_required_stamp_ = false;
        }
        res->success = true;
        res->message = "Voxel map update skipped (same view).";
        return;
      }
    }

    sensor_msgs::msg::PointCloud2 cloud_copy = *cloud_ptr;

    updateFromCloud(cloud_copy, stamp, cam_pose);

    last_cam_pose_in_target_ = cam_pose;
    last_cam_pose_valid_ = true;

    last_processed_stamp_ = stamp;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      have_required_stamp_ = false;
    }
    res->success = true;
    res->message = "Voxel map updated.";
  }

  struct AutoRaycastParams {
    int ray_stride_px;
    int point_stride;
    double max_ray_range_m;
    int max_steps_per_ray;
  };

  AutoRaycastParams computeAutoRaycastParams(const Eigen::Vector3d &cam_origin) {
    AutoRaycastParams p{std::max(1, ray_stride_px_), std::max(1, point_stride_), max_ray_range_m_, max_steps_per_ray_};
    if (!auto_params_enable_) {
      return p;
    }

    const double range = std::max(0.2, cam_origin.norm());
    const double mpp = range / std::max(1.0, fx_);  // meters per pixel at nominal depth
    const double voxel_px = res_ / std::max(1e-6, mpp);
    const int stride_auto = static_cast<int>(std::lround(voxel_px / std::max(0.5, auto_target_pixels_per_voxel_)));

    p.ray_stride_px = std::clamp(stride_auto,
                                 std::max(1, auto_min_ray_stride_px_),
                                 std::max(std::max(1, auto_min_ray_stride_px_), auto_max_ray_stride_px_));

    const int point_auto = static_cast<int>(std::lround(std::max(1.0, p.ray_stride_px * auto_point_stride_ratio_)));
    p.point_stride = std::clamp(point_auto,
                                std::max(1, auto_point_stride_min_),
                                std::max(std::max(1, auto_point_stride_min_), auto_point_stride_max_));

    const double bbx_diag = (bbx_max_ - bbx_min_).norm();
    const double range_auto = std::max(0.1, auto_max_ray_range_scale_ * bbx_diag);
    if (max_ray_range_m_ > 0.0) {
      p.max_ray_range_m = std::min(max_ray_range_m_, range_auto);
    } else {
      p.max_ray_range_m = range_auto;
    }

    const int steps_auto = std::max(1, static_cast<int>(std::ceil(p.max_ray_range_m / std::max(1e-6, res_)))) +
                           std::max(0, auto_max_steps_margin_);
    p.max_steps_per_ray = (max_steps_per_ray_ > 0) ? std::min(max_steps_per_ray_, steps_auto) : steps_auto;
    return p;
  }

  void updateFromCloud(const sensor_msgs::msg::PointCloud2 &cloud,
                       const rclcpp::Time &publish_stamp,
                       const geometry_msgs::msg::PoseStamped &cam_pose_in_target) 
  {
    const Eigen::Vector3d cam_origin(cam_pose_in_target.pose.position.x,
                                     cam_pose_in_target.pose.position.y,
                                     cam_pose_in_target.pose.position.z);
    Eigen::Quaterniond q(cam_pose_in_target.pose.orientation.w,
                         cam_pose_in_target.pose.orientation.x,
                         cam_pose_in_target.pose.orientation.y,
                         cam_pose_in_target.pose.orientation.z);
    const Eigen::Matrix3d R_target_cam = q.normalized().toRotationMatrix();

    // --- 1. Insert Occupied Points ---
    const AutoRaycastParams auto_p = computeAutoRaycastParams(cam_origin);
    const int stride = std::max(1, auto_p.point_stride);
    sensor_msgs::PointCloud2ConstIterator<float> it_x(cloud, "x"), it_y(cloud, "y"), it_z(cloud, "z");

    size_t count = 0;
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z, ++count) {
      if ((count % stride) != 0)
        continue;

      Eigen::Vector3d p(*it_x, *it_y, *it_z);
      if (!finite3(p.x(), p.y(), p.z()))
        continue;
      if (!insideBBX(p))
        continue;

      auto *n = map_->updateNode(octomap::point3d(p.x(), p.y(), p.z()), true);
      if (n)
        n->setColor(0, 0, 255); // Occupied Color (Blue)
    }

    // --- 2. Raycasting (Free Space) ---
    const int step_px = std::max(1, auto_p.ray_stride_px);
    int rays_cast = 0;

    const int nu = (image_width_ + step_px - 1) / step_px;
    const int nv = (image_height_ + step_px - 1) / step_px;
    const int total = nu * nv;

    std::vector<std::unique_ptr<octomap::KeyRay>> rays(total);

#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; ++idx) {
      const int iu = idx % nu;
      const int iv = idx / nu;
      const int u = iu * step_px;
      const int v = iv * step_px;

      Eigen::Vector3d d_cam((u - cx_) / fx_, (v - cy_) / fy_, 1.0);
      const double nrm = d_cam.norm();
      if (nrm < 1e-12) continue;
      d_cam /= nrm;

      const Eigen::Vector3d d = R_target_cam * d_cam;

      double t_enter, t_exit;
      if (!intersectRayAABB(cam_origin, d, bbx_min_, bbx_max_, t_enter, t_exit)) continue;
      if (t_exit <= 0.0) continue;

      const double t0 = std::max(0.0, t_enter);
      double t1 = (auto_p.max_ray_range_m > 0.0) ? std::min(t_exit, auto_p.max_ray_range_m) : t_exit;
      if (t1 <= t0) continue;

      const Eigen::Vector3d start = clampVec(cam_origin + t0 * d, bbx_min_, bbx_max_);
      const Eigen::Vector3d end = clampVec(cam_origin + t1 * d, bbx_min_, bbx_max_);

      auto ray_ptr = std::make_unique<octomap::KeyRay>();
      if (!map_->computeRayKeys(
              octomap::point3d(start.x(), start.y(), start.z()),
              octomap::point3d(end.x(), end.y(), end.z()), *ray_ptr)) {
        continue;
      }

      rays[idx] = std::move(ray_ptr);
    }

    // Serial phase: apply colors
    for (int idx = 0; idx < total; ++idx) {
      if (!rays[idx])
        continue;
      ++rays_cast;

      bool hit_occupied = false;
      int steps = 0;

      for (const auto &k : *rays[idx]) {
        auto *n = map_->search(k);
        if (!n)
          continue;

        if (map_->isNodeOccupied(n)) {
          hit_occupied = true;
          continue;
        }

        if (!hit_occupied) {
          if (isNone(n))
            n->setColor(255, 255, 255);
        } else {
          if (isNone(n))
            n->setColor(188, 188, 188);
        }

        if (auto_p.max_steps_per_ray > 0 && ++steps >= auto_p.max_steps_per_ray)
          break;
      }
    }

    // --- 3. Extract & Publish ---
    publishMap(publish_stamp);

    RCLCPP_INFO(get_logger(), "✅ Map updated. Rays: %d stride_px=%d point_stride=%d max_range=%.3f steps_cap=%d",
                rays_cast, auto_p.ray_stride_px, auto_p.point_stride, auto_p.max_ray_range_m, auto_p.max_steps_per_ray);
  }

  void publishMap(const rclcpp::Time &stamp) {
    std::vector<Eigen::Vector3d> occ_pts, frontier_pts;

    // ---- Build occupied list ----
    for (auto it = map_->begin_leafs(), end = map_->end_leafs(); it != end; ++it) {
      Eigen::Vector3d c(it.getX(), it.getY(), it.getZ());
      if (!insideBBX(c))
        continue;

      auto *n = &(*it);
      if (map_->isNodeOccupied(n)) occ_pts.push_back(c);
    }

    // ---- Frontier: Vu with BOTH Vf and Vo ----
    octomap::OcTreeKey nb[26];
    for (auto it = map_->begin_leafs(), end = map_->end_leafs(); it != end; ++it) {
      Eigen::Vector3d c(it.getX(), it.getY(), it.getZ());
      if (!insideBBX(c)) continue;

      auto *cur = &(*it);
      if (!isGray(cur)) continue; // Vu only (188)

      const auto k = it.getKey();
      const int nnb = neighbors26(k, nb);

      bool has_free = false;
      bool has_occ = false;

      for (int i = 0; i < nnb; ++i) {
        auto *n = map_->search(nb[i]);
        if (!n)
          continue;
        if (map_->isNodeOccupied(n))
          has_occ = true;
        else if (isWhite(n))
          has_free = true;
        if (has_free && has_occ)
          break;
      }

      if (has_free && has_occ)
        frontier_pts.emplace_back(c.x(), c.y(), c.z());
    }

    pub_occupied_->publish(makeCloudXYZ(target_frame_, stamp, occ_pts));
    pub_frontier_->publish(makeCloudXYZ(target_frame_, stamp, frontier_pts));
    RCLCPP_INFO(get_logger(), "✅ Frontier points: %zu, Occupied points: %zu", frontier_pts.size(), occ_pts.size());
    publishBBX(stamp);
  }

  // Member Vars
  rcl_clock_type_t clock_type_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string target_frame_, camera_frame_;
  std::string step_stamp_topic_, points_topic_, occupied_topic_,
      frontier_topic_, bbox_topic_, cam_pose_topic_;
  std::string update_service_;

  struct StampedCloud {
    rclcpp::Time stamp;
    sensor_msgs::msg::PointCloud2::SharedPtr msg;
  };

  struct StampedPose {
    rclcpp::Time stamp;
    geometry_msgs::msg::PoseStamped::SharedPtr msg;
  };

  std::deque<StampedCloud> cloud_buf_;
  std::deque<StampedPose> pose_buf_;
  rclcpp::Time last_processed_stamp_;
  rclcpp::Time required_stamp_;
  double tf_lookup_timeout_sec_{0.05};
  size_t cloud_buf_max_{50};
  size_t pose_buf_max_{50};
  double wait_for_inputs_sec_{0.5};
  bool have_required_stamp_{false};
  bool have_last_token_logged_{false};
  rclcpp::Time last_token_logged_;
  bool use_cam_pose_topic_{false};
  bool skip_update_if_same_view_{false};
  double same_view_pos_tol_m_{1e-3};
  double same_view_yaw_tol_rad_{1e-3};
  bool last_cam_pose_valid_{false};
  geometry_msgs::msg::PoseStamped last_cam_pose_in_target_;

  std::mutex mtx_;
  std::condition_variable cv_;

  std::unique_ptr<octomap::ColorOcTree> map_;
  Eigen::Vector3d bbx_min_, bbx_max_;
  double res_, bbx_half_side_m_, bbx_z_min_m_, bbx_z_max_m_, max_ray_range_m_;
  int ray_stride_px_, point_stride_, max_steps_per_ray_;

  bool auto_params_enable_{false};
  bool auto_voxel_resolution_enable_{false};
  double auto_nominal_depth_m_{1.0};
  int auto_target_voxels_per_diag_{45};
  double auto_voxel_res_min_m_{0.01};
  double auto_voxel_res_max_m_{0.05};
  double auto_target_pixels_per_voxel_{1.5};
  int auto_min_ray_stride_px_{2};
  int auto_max_ray_stride_px_{12};
  int auto_point_stride_min_{1};
  int auto_point_stride_max_{6};
  double auto_point_stride_ratio_{0.5};
  double auto_max_ray_range_scale_{1.3};
  int auto_max_steps_margin_{8};
  int image_width_, image_height_;
  double fx_, fy_, cx_, cy_;

  rclcpp::Subscription<builtin_interfaces::msg::Time>::SharedPtr
      sub_step_stamp_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      sub_cam_pose_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_occupied_,
      pub_frontier_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_bbox_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_update_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VoxelMapService>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}