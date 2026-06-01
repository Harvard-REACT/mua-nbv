#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.hpp>

#include "ellipsoid_clustering.hpp"
#include "visualization.hpp"

#include <tf2/time.h>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

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
} // namespace

class ScorerNode : public rclcpp::Node {
public:
  ScorerNode() : Node("score_node") {
    // ---- Params ----
    // Topics
    step_stamp_topic_ = declare_parameter<std::string>(
        "input_step_stamp_topic", "/experiment/step_stamp");
    static_target_ = declare_parameter<bool>("static_target", false);
    target_frame_ =
        declare_parameter<std::string>("target_frame", "target/base_link");
    tf_lookup_timeout_sec_ =
        declare_parameter<double>("tf_lookup_timeout_sec", 0.2);
    // On hardware, token times can be ahead of TF; optionally use latest TF.
    use_latest_tf_ = declare_parameter<bool>("use_latest_tf", false);
    prediction_topic_ = declare_parameter<std::string>(
        "input_pred_topic", "/nbv/target_prediction");
    prediction_state_topic_ = declare_parameter<std::string>(
        "input_pred_state_topic", "/nbv/target_prediction_state");
    candidates_topic_ = declare_parameter<std::string>(
        "input_candidates_topic", "/nbv/candidates_world_optical");
    frontier_topic_ = declare_parameter<std::string>("input_frontier_topic",
                                                     "/nbv/voxels/frontier");
    occupied_topic_ = declare_parameter<std::string>("input_occupied_topic",
                                                     "/nbv/voxels/occupied");
    scores_topic_ =
        declare_parameter<std::string>("output_scores_topic", "/nbv/scores");
    best_idx_topic_ = declare_parameter<std::string>("output_best_index_topic",
                                                     "/nbv/best_index");
    best_candidate_topic_ = declare_parameter<std::string>(
        "output_best_candidate_topic", "/nbv/best_candidate");
    proj_img_topic_ = declare_parameter<std::string>(
        "output_proj_img_topic", "/nbv/debug/projection_image");
    ring_img_topic_ = declare_parameter<std::string>(
        "output_ring_img_topic", "/nbv/debug/score_ring_image");
    // Service
    score_service_ =
        declare_parameter<std::string>("score_service", "/nbv/score");
    // Wait budgets
    wait_for_inputs_sec_ =
        declare_parameter<double>("wait_for_inputs_sec", 0.5);
    pred_sync_tol_sec_ = declare_parameter<double>("pred_sync_tol_sec", 0.0);
    require_new_inputs_ = declare_parameter<bool>("require_new_inputs", true);
    image_width_ = declare_parameter<int>("image_width", 640);
    image_height_ = declare_parameter<int>("image_height", 480);
    fx_ = declare_parameter<double>("fx", 554.38);
    fy_ = declare_parameter<double>("fy", 554.38);
    cx_ = declare_parameter<double>("cx", 320.0);
    cy_ = declare_parameter<double>("cy", 240.0);
    K_.setZero();
    K_(0, 0) = fx_;
    K_(1, 1) = fy_;
    K_(0, 2) = cx_;
    K_(1, 2) = cy_;
    K_(2, 2) = 1.0;
    // Voxel map
    voxel_res_ = declare_parameter<double>("voxel_resolution_m", 0.05);
    min_cluster_pts_ = (size_t)declare_parameter<int>("min_cluster_pts", 20);
    max_points_per_cluster_ =
        (size_t)declare_parameter<int>("max_points_per_cluster", 2000);
    // GMM
    min_gmm_cluster_num_ = declare_parameter<int>("min_gmm_cluster_num", 5);
    max_gmm_cluster_num_ = declare_parameter<int>("max_gmm_cluster_num", 15);
    gmm_max_iters_ = declare_parameter<int>("gmm_max_iters", 100);
    gmm_term_eps_ = declare_parameter<double>("gmm_term_eps", 1e-3);
    gmm_use_bic_ = declare_parameter<bool>("gmm_use_bic", true);
    gmm_max_points_ = declare_parameter<int>("gmm_max_points", 20000);

    // Auto cluster parameterization (sample-count driven)
    auto_cluster_params_enable_ =
        declare_parameter<bool>("auto_cluster_params_enable", false);
    auto_min_cluster_pts_floor_ =
        declare_parameter<int>("auto_min_cluster_pts_floor", 8);
    auto_cluster_k_cap_ =
        declare_parameter<int>("auto_cluster_k_cap", 50);
    auto_gmm_min_points_ =
        declare_parameter<int>("auto_gmm_min_points", 200);
    // Scorer
    cgal_eps_ = declare_parameter<double>("cgal_eps", 0.01);
    depth_decay_ = declare_parameter<double>("depth_decay", 0.5);
    // Monte Carlo params
    v_min_ = declare_parameter<double>("v_min", 0.0);
    use_monte_carlo_ = declare_parameter<bool>("use_monte_carlo", true);
    mc_num_samples_ = declare_parameter<int>("mc_num_samples", 10);
    mc_seed_ = declare_parameter<int>("mc_seed", 0);
    rng_ = std::mt19937(static_cast<uint32_t>(mc_seed_));
    // Selection mode: "argmax", "random", or "softmax"
    selection_mode_ =
        declare_parameter<std::string>("selection_mode", "argmax");
    if (selection_mode_ != "argmax" && selection_mode_ != "random" &&
        selection_mode_ != "softmax") {
      RCLCPP_ERROR(
          get_logger(),
          "Unknown selection_mode '%s'; must be 'argmax', 'random', or 'softmax'",
          selection_mode_.c_str());
      throw std::invalid_argument("bad selection_mode");
    }
    softmax_temperature_ = declare_parameter<double>("softmax_temperature", 0.2);
    // Debug params
    debug_save_projection_png_ =
        declare_parameter<bool>("debug_save_projection_png", true);
    debug_save_all_candidates_ =
        declare_parameter<bool>("debug_save_all_candidates", true);
    debug_save_ellipsoids_json_ =
        declare_parameter<bool>("debug_save_ellipsoids_json", true);
    debug_dir_ =
        declare_parameter<std::string>("debug_dir", "debug/mua_nbv_planner");
    debug_group_by_run_id_ =
        declare_parameter<bool>("debug_group_by_run_id", true);
    run_id_topic_ =
        declare_parameter<std::string>("run_id_topic", "/experiment/run_id");
    if (debug_save_projection_png_ || debug_save_ellipsoids_json_) {
      std::filesystem::create_directories(debug_dir_);
    }

    // ---- Subscriber / Publisher / Service ----
    // Subscribers
    cb_sub_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = cb_sub_;
    sub_step_stamp_ = create_subscription<builtin_interfaces::msg::Time>(
        step_stamp_topic_, rclcpp::QoS(10).reliable(),
        [this](const builtin_interfaces::msg::Time::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          required_stamp_ = rclcpp::Time(msg->sec, msg->nanosec,
                                         get_clock()->get_clock_type());
          have_required_stamp_ = true;
          if (!have_last_token_logged_ ||
              required_stamp_ != last_token_logged_) {
            have_last_token_logged_ = true;
            last_token_logged_ = required_stamp_;
            RCLCPP_INFO(get_logger(), "🟣 TOKEN recv: %s",
                        stampStr(required_stamp_).c_str());
          }
          cv_.notify_all();
        },
        sub_opts);
    // Run id for grouping debug outputs per experiment run.
    sub_run_id_ = create_subscription<std_msgs::msg::String>(
        run_id_topic_, rclcpp::QoS(1).transient_local().reliable(),
        [this](const std_msgs::msg::String::SharedPtr msg) {
          if (!msg)
            return;
          std::lock_guard<std::mutex> lk(mtx_);
          run_id_ = msg->data;
        },
        sub_opts);
    sub_pred_ =
        create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            prediction_topic_, rclcpp::QoS(10),
            std::bind(&ScorerNode::onPrediction, this, std::placeholders::_1),
            sub_opts);
    sub_pred_state_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        prediction_state_topic_, rclcpp::QoS(10),
        std::bind(&ScorerNode::onPredictionState, this, std::placeholders::_1),
        sub_opts);
    sub_candidates_ = create_subscription<geometry_msgs::msg::PoseArray>(
        candidates_topic_, rclcpp::QoS(10),
        std::bind(&ScorerNode::onCandidates, this, std::placeholders::_1),
        sub_opts);
    auto qos_pc = rclcpp::SensorDataQoS().keep_last(5);
    sub_frontier_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        frontier_topic_, qos_pc,
        std::bind(&ScorerNode::onFrontier, this, std::placeholders::_1),
        sub_opts);
    sub_occupied_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        occupied_topic_, qos_pc,
        std::bind(&ScorerNode::onOccupied, this, std::placeholders::_1),
        sub_opts);
    // Publishers
    pub_scores_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        scores_topic_, rclcpp::QoS(1).reliable());
    pub_best_ = create_publisher<std_msgs::msg::Int32>(
        best_idx_topic_, rclcpp::QoS(1).reliable());
    pub_best_candidate_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        best_candidate_topic_, rclcpp::QoS(1).reliable());
    pub_proj_img_ = create_publisher<sensor_msgs::msg::Image>(
        proj_img_topic_, rclcpp::SensorDataQoS().best_effort());
    pub_ring_img_ = create_publisher<sensor_msgs::msg::Image>(
        ring_img_topic_, rclcpp::SensorDataQoS().best_effort());
    pub_score_time_ = create_publisher<std_msgs::msg::Float64>(
        "/nbv/score_time_ms", rclcpp::QoS(10).reliable());
    // Service
    cb_srv_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    srv_score_ = create_service<std_srvs::srv::Trigger>(
        score_service_,
        std::bind(&ScorerNode::onScoreService, this, std::placeholders::_1,
                  std::placeholders::_2),
        rclcpp::ServicesQoS(), cb_srv_);

    RCLCPP_INFO(
        get_logger(),
        "----------------------------------------------------------------");
    RCLCPP_INFO(get_logger(), "📡 Listening to: %s, %s, %s",
                step_stamp_topic_.c_str(), prediction_topic_.c_str(),
                prediction_state_topic_.c_str());
    RCLCPP_INFO(get_logger(), "📡 Listening to: %s, %s, %s",
                candidates_topic_.c_str(), frontier_topic_.c_str(),
                occupied_topic_.c_str());
    RCLCPP_INFO(get_logger(), "📢 Publishing to: %s, %s, %s",
                scores_topic_.c_str(), best_idx_topic_.c_str(),
                best_candidate_topic_.c_str());
    RCLCPP_INFO(get_logger(), "📢 Publishing to: %s, %s",
                proj_img_topic_.c_str(), ring_img_topic_.c_str());
    RCLCPP_INFO(get_logger(), "🔧 Service: %s", score_service_.c_str());
    if (debug_save_projection_png_) {
      RCLCPP_INFO(get_logger(), "📁 Debug PNG saving enabled: dir='%s'",
                  debug_dir_.c_str());
    }
    if (debug_save_all_candidates_) {
      RCLCPP_INFO(get_logger(),
                  "📁 Debug all candidates saving enabled: dir='%s'",
                  debug_dir_.c_str());
    }
    if (debug_save_ellipsoids_json_) {
      RCLCPP_INFO(get_logger(), "📁 Debug ellipsoids JSON saving enabled: dir='%s'",
                  debug_dir_.c_str());
    }
    RCLCPP_INFO(get_logger(), "📁 Debug group by run id: %s (topic=%s)",
                debug_group_by_run_id_ ? "true" : "false",
                run_id_topic_.c_str());
    RCLCPP_INFO(get_logger(), "📊 Image width and height: %d, %d", image_width_,
                image_height_);
    RCLCPP_INFO(
        get_logger(),
        "📊 Image focal length and principal point: %.3f, %.3f, (%.3f, %.3f)",
        fx_, fy_, cx_, cy_);
    RCLCPP_INFO(get_logger(), "📊 Voxel resolution: %.3f m", voxel_res_);
    RCLCPP_INFO(get_logger(), "📊 Min cluster points: %zu", min_cluster_pts_);
    RCLCPP_INFO(get_logger(), "📊 Max points per cluster: %zu",
                max_points_per_cluster_);
    RCLCPP_INFO(get_logger(), "📊 Min and Max GMM cluster num: %d, %d",
                min_gmm_cluster_num_, max_gmm_cluster_num_);
    RCLCPP_INFO(get_logger(), "📊 GMM max iters: %d", gmm_max_iters_);
    RCLCPP_INFO(get_logger(), "📊 GMM term eps: %.3f", gmm_term_eps_);
    RCLCPP_INFO(get_logger(), "📊 GMM use BIC: %s",
                gmm_use_bic_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "📊 GMM max points: %d", gmm_max_points_);
    RCLCPP_INFO(get_logger(), "📊 Auto cluster params: enable=%s floor=%d k_cap=%d",
                auto_cluster_params_enable_ ? "true" : "false",
                auto_min_cluster_pts_floor_, auto_cluster_k_cap_);
    RCLCPP_INFO(get_logger(), "📊 CGAL eps: %.3f", cgal_eps_);
    RCLCPP_INFO(get_logger(), "📊 Depth decay: %.3f", depth_decay_);
    RCLCPP_INFO(get_logger(), "📊 V min: %.3f", v_min_);
    if (use_monte_carlo_) {
      RCLCPP_INFO(get_logger(), "📊 Use Monte Carlo: true");
      RCLCPP_INFO(get_logger(), "📊 Monte Carlo num samples: %d",
                  mc_num_samples_);
      RCLCPP_INFO(get_logger(), "📊 Monte Carlo seed: %d", mc_seed_);
    }
    RCLCPP_INFO(get_logger(), "📊 use_latest_tf=%s",
                use_latest_tf_ ? "true" : "false");
    RCLCPP_INFO(
        get_logger(),
        "----------------------------------------------------------------");
  }

private:
  void onCandidates(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    if (!msg)
      return;
    std::lock_guard<std::mutex> lk(mtx_);
    candidates_ = *msg;
    cv_.notify_all();
  }

  void onFrontier(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!msg)
      return;
    std::lock_guard<std::mutex> lk(mtx_);
    frontier_cloud_ = *msg;
    have_frontier_ = true;
    cv_.notify_all();
  }

  void onOccupied(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!msg)
      return;
    std::lock_guard<std::mutex> lk(mtx_);
    occupied_cloud_ = *msg;
    have_occupied_ = true;
    cv_.notify_all();
  }

  struct ClusterTuneParams {
    size_t min_cluster_pts;
    int min_gmm_k;
    int max_gmm_k;
    int gmm_max_points;
  };

  ClusterTuneParams tunedClusterParams(size_t n_pts) const {
    ClusterTuneParams p{min_cluster_pts_, min_gmm_cluster_num_,
                        max_gmm_cluster_num_, gmm_max_points_};
    if (!auto_cluster_params_enable_) {
      return p;
    }
    if (n_pts <= 3) {
      p.min_cluster_pts = std::max<size_t>(1, static_cast<size_t>(auto_min_cluster_pts_floor_));
      p.min_gmm_k = 1;
      p.max_gmm_k = 1;
      p.gmm_max_points = std::max(16, auto_gmm_min_points_ / 2);
      return p;
    }

    const int n = static_cast<int>(n_pts);
    int max_k = std::min({std::max(1, max_gmm_cluster_num_),
                          std::max(1, auto_cluster_k_cap_),
                          std::max(1, n / 2)});
    if (n < 2 * max_k) {
      max_k = std::max(1, n / 2);
    }
    int min_k = std::min(std::max(1, min_gmm_cluster_num_), max_k);

    const int min_pts_auto = std::max(auto_min_cluster_pts_floor_,
                                      n / std::max(2, 3 * std::max(1, max_k)));
    p.min_cluster_pts = static_cast<size_t>(std::clamp(min_pts_auto, 1, std::max(1, n / 2)));
    p.min_gmm_k = min_k;
    p.max_gmm_k = max_k;
    p.gmm_max_points = std::min(std::max(auto_gmm_min_points_, n), gmm_max_points_);
    return p;
  }

  void saveEllipsoidsJson(
      const std::string &debug_out_dir,
      const builtin_interfaces::msg::Time &stamp,
      const std::vector<mua_nbv_planner::EllipsoidParam> &ellipsoids,
      const size_t frontier_points,
      const size_t occupied_points) {
    if (!debug_save_ellipsoids_json_)
      return;

    std::error_code ec;
    std::filesystem::create_directories(debug_out_dir, ec);

    const rclcpp::Time t(stamp.sec, stamp.nanosec, get_clock()->get_clock_type());
    const std::string stamp_label = stampStr(t);
    const std::string path = debug_out_dir + "/ellipsoids_" + stamp_label + ".json";

    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.good()) {
      RCLCPP_WARN(get_logger(), "Failed to write ellipsoid debug JSON: %s", path.c_str());
      return;
    }

    ofs << "{\n";
    ofs << "  \"stamp\": \"" << stamp_label << "\",\n";
    ofs << "  \"target_frame\": \"" << target_frame_ << "\",\n";
    ofs << "  \"frontier_points\": " << frontier_points << ",\n";
    ofs << "  \"occupied_points\": " << occupied_points << ",\n";
    ofs << "  \"ellipsoid_count\": " << ellipsoids.size() << ",\n";
    ofs << "  \"ellipsoids\": [\n";

    for (size_t i = 0; i < ellipsoids.size(); ++i) {
      const auto &e = ellipsoids[i];
      const Eigen::Matrix3d R = e.pose.block<3, 3>(0, 0);
      const Eigen::Vector3d t3 = e.pose.block<3, 1>(0, 3);
      Eigen::Quaterniond q(R);
      q.normalize();

      ofs << "    {\n";
      ofs << "      \"id\": " << i << ",\n";
      ofs << "      \"type\": \"" << e.type << "\",\n";
      ofs << "      \"center\": [" << t3.x() << ", " << t3.y() << ", " << t3.z() << "],\n";
      ofs << "      \"radii\": [" << e.radii.x() << ", " << e.radii.y() << ", " << e.radii.z() << "],\n";
      ofs << "      \"quat_wxyz\": [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "],\n";
      ofs << "      \"rotation_row_major\": ["
          << R(0, 0) << ", " << R(0, 1) << ", " << R(0, 2) << ", "
          << R(1, 0) << ", " << R(1, 1) << ", " << R(1, 2) << ", "
          << R(2, 0) << ", " << R(2, 1) << ", " << R(2, 2) << "]\n";
      ofs << "    }";
      if (i + 1 < ellipsoids.size())
        ofs << ",";
      ofs << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";
  }

  int softmaxSample(const std::vector<double> &raw_scores) {
    const size_t C = raw_scores.size();
    if (C == 0) return -1;
    if (C == 1) return 0;

    // Normalize scores to [0, 1]
    double smin = *std::min_element(raw_scores.begin(), raw_scores.end());
    double smax = *std::max_element(raw_scores.begin(), raw_scores.end());
    double denom = smax - smin;

    std::vector<double> probs(C);
    if (denom > 1e-12 && softmax_temperature_ > 1e-12) {
      double max_norm = 1.0;  // after normalization, max is always 1.0
      double sum_exp = 0.0;
      for (size_t i = 0; i < C; ++i) {
        double ns = (raw_scores[i] - smin) / denom;  // [0, 1]
        probs[i] = std::exp((ns - max_norm) / softmax_temperature_);
        sum_exp += probs[i];
      }
      for (auto &p : probs) p /= sum_exp;
    } else {
      std::fill(probs.begin(), probs.end(), 1.0 / static_cast<double>(C));
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng_);
  }

  double
  scoreView(const geometry_msgs::msg::Pose &cand_pose,
            const std::vector<mua_nbv_planner::EllipsoidParam> &ellipsoids) {
    // ---- Build T_target_opt from Pose (target -> optical) ----
    Eigen::Quaterniond q(cand_pose.orientation.w, cand_pose.orientation.x,
                         cand_pose.orientation.y, cand_pose.orientation.z);
    q.normalize();

    Eigen::Matrix4d T_target_opt = Eigen::Matrix4d::Identity();
    T_target_opt.block<3, 3>(0, 0) = q.toRotationMatrix();
    T_target_opt(0, 3) = cand_pose.position.x;
    T_target_opt(1, 3) = cand_pose.position.y;
    T_target_opt(2, 3) = cand_pose.position.z;
    const Eigen::Matrix4d T_opt_target =
        T_target_opt.inverse(); // optical <- target
    const Eigen::Matrix<double, 3, 4> Rt =
        T_opt_target.block<3, 4>(0, 0); // Rotation and translation matrix
    const Eigen::Matrix<double, 3, 4> P = K_ * Rt; // Projection matrix

    struct ZItem {
      size_t i;
      double z;
    };
    std::vector<ZItem> order;
    order.reserve(ellipsoids.size());
    // Iterate over all ellipsoids and compute their depth in the optical frame
    for (size_t i = 0; i < ellipsoids.size(); ++i) {
      Eigen::Vector4d c_h(ellipsoids[i].pose(0, 3), ellipsoids[i].pose(1, 3),
                          ellipsoids[i].pose(2, 3), 1.0);
      Eigen::Vector3d c_cam =
          (T_opt_target * c_h).head<3>(); // in optical frame
      if (!c_cam.allFinite())
        continue;
      order.push_back({i, c_cam.z()});
    }

    std::sort(order.begin(), order.end(), [](const ZItem &a, const ZItem &b) {
      return a.z < b.z;
    }); // near first

    // ---- weights: w = 1, depth_decay, depth_decay^2, ... in sorted order ----
    std::vector<double> weights(ellipsoids.size(), 0.0);
    double w = 1.0;
    // Compute the weights for the ellipsoids based on their depth
    for (const auto &zi : order) {
      if (zi.z <= 1e-6) {
        weights[zi.i] = 0.0;
        continue;
      }
      weights[zi.i] = w;
      w *= depth_decay_;
    }

    // Compute the mask for the i-th ellipsoid
    auto ellipseMask = [&](size_t i, cv::Mat &mask_out) -> bool {
      const Eigen::Matrix4d Qdual =
          mua_nbv_planner::createEllipsoidDualMatrix(ellipsoids[i]);
      if (Qdual.isZero(0))
        return false;

      const Eigen::Matrix3d conic =
          mua_nbv_planner::projectDualQuadricToConic(P, Qdual);
      if (conic.isZero(0))
        return false;

      cv::Point2d center;
      cv::Size2d axes;
      double angle_deg = 0.0;
      if (!mua_nbv_planner::conicToEllipse(conic, center, axes, angle_deg))
        return false;

      // Reject nonsense
      if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
          !std::isfinite(axes.width) || !std::isfinite(axes.height))
        return false;
      if (axes.width < 1.0 || axes.height < 1.0)
        return false;
      if (axes.width > 1e5 || axes.height > 1e5)
        return false;

      mask_out = cv::Mat(image_height_, image_width_, CV_8UC1, cv::Scalar(0));
      cv::ellipse(mask_out, center, axes, angle_deg, 0.0, 360.0,
                  cv::Scalar(255), -1);
      return true;
    };

    double frontier_sum = 0.0;
    double occupied_sum = 0.0;

    // Iterate over all ellipsoids and compute their contribution to the score
    for (size_t i = 0; i < ellipsoids.size(); ++i) {
      if (weights[i] <= 0.0)
        continue;

      cv::Mat mask;
      if (!ellipseMask(i, mask))
        continue;

      const double pix = (double)cv::countNonZero(
          mask); // Count the number of pixels in the mask
      const double contrib =
          pix * weights[i]; // Contribution of the ellipsoid to the score

      if (ellipsoids[i].type == "frontier")
        frontier_sum += contrib;
      else if (ellipsoids[i].type == "occupied")
        occupied_sum += contrib;
    }

    return frontier_sum - occupied_sum;
  }

  void onScoreService(const std_srvs::srv::Trigger::Request::SharedPtr,
                      const std_srvs::srv::Trigger::Response::SharedPtr res) {
    std::string why;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (have_required_stamp_) {
        RCLCPP_INFO(get_logger(), "🟣 TOKEN score request: token=%s",
                    stampStr(required_stamp_).c_str());
      } else {
        RCLCPP_INFO(get_logger(), "🟣 TOKEN score request: token=None");
      }
    }
    const bool ok = scoreOnce(why);
    res->success = ok;
    res->message =
        ok ? "Scoring computed and published." : ("Scoring skipped: " + why);
    if (ok) {
      RCLCPP_INFO(get_logger(), "✅ %s", res->message.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "⚠️ %s", res->message.c_str());
    }
  }

  static inline double wrapAngle(double a) {
    while (a > M_PI)
      a -= 2.0 * M_PI;
    while (a < -M_PI)
      a += 2.0 * M_PI;
    return a;
  }

  static inline double yawFromQuat(const geometry_msgs::msg::Quaternion &qmsg) {
    // ZYX yaw from quaternion
    const double w = qmsg.w, x = qmsg.x, y = qmsg.y, z = qmsg.z;
    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  static inline Eigen::Isometry3d
  isoFromPoseMsg(const geometry_msgs::msg::Pose &p) {
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y,
                         p.orientation.z);
    q.normalize();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() << p.position.x, p.position.y, p.position.z;
    return T;
  }

  static inline geometry_msgs::msg::Pose
  poseMsgFromIso(const Eigen::Isometry3d &T) {
    geometry_msgs::msg::Pose p;
    p.position.x = T.translation().x();
    p.position.y = T.translation().y();
    p.position.z = T.translation().z();
    Eigen::Quaterniond q(T.linear());
    q.normalize();
    p.orientation.w = q.w();
    p.orientation.x = q.x();
    p.orientation.y = q.y();
    p.orientation.z = q.z();
    return p;
  }

  static inline Eigen::Isometry3d isoFromXYYaw(double x, double y, double yaw) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() << x, y, 0.0;
    const double c = std::cos(yaw), s = std::sin(yaw);
    T.linear() << c, -s, 0, s, c, 0, 0, 0, 1;
    return T;
  }

  bool scoreOnce(std::string &why) {
    geometry_msgs::msg::PoseArray candidates;
    sensor_msgs::msg::PointCloud2 frontier, occupied;
    geometry_msgs::msg::PoseWithCovarianceStamped pred;

    auto ready = [&]() {
      if (candidates_.poses.empty())
        return false;
      if (!have_frontier_ || !have_occupied_)
        return false;
      if (!static_target_ && (!have_pred_ || !have_pred_state_))
        return false;

      const rclcpp::Time tc(candidates_.header.stamp,
                            get_clock()->get_clock_type());
      const rclcpp::Time tf(frontier_cloud_.header.stamp,
                            get_clock()->get_clock_type());
      const rclcpp::Time to(occupied_cloud_.header.stamp,
                            get_clock()->get_clock_type());
      const rclcpp::Time tp(pred_.header.stamp, get_clock()->get_clock_type());
      const rclcpp::Time ts = pred_state_stamp_;

      auto close = [&](const rclcpp::Time &a, const rclcpp::Time &b,
                       double tol_sec) {
        return std::abs((a - b).seconds()) <= tol_sec;
      };

      // Step token gating is ALWAYS enabled under the new pipeline policy.
      if (!have_required_stamp_)
        return false;

      // Everything must be >= the step token stamp (no stale reuse)
      if (tc < required_stamp_ || tf < required_stamp_ ||
          to < required_stamp_ ||
          (!static_target_ && (tp < required_stamp_ || ts < required_stamp_)))
        return false;

      if (!static_target_) {
        const double ptol = std::max(0.0, pred_sync_tol_sec_);
        if (ptol > 0.0) {
          if (!close(tp, required_stamp_, ptol))
            return false;
          if (!close(ts, required_stamp_, ptol))
            return false;
        } else {
          if (tp != required_stamp_ || ts != required_stamp_)
            return false;
        }
      }

      if (require_new_inputs_) {
        if (tc <= last_processed_stamp_)
          return false;
      }

      return true;
    };

    {
      std::unique_lock<std::mutex> lk(mtx_);
      if (!ready()) {
        cv_.wait_for(lk, std::chrono::duration<double>(wait_for_inputs_sec_),
                     ready);
      }
      if (!ready()) {
        why = "inputs not ready for this step";
        return false;
      }

      // snapshot consistent set
      candidates = candidates_;
      frontier = frontier_cloud_;
      occupied = occupied_cloud_;
      pred = pred_;

      // consume required stamp (one score per step)
      have_required_stamp_ = false;
      last_processed_stamp_ =
          rclcpp::Time(candidates.header.stamp, get_clock()->get_clock_type());
    }

    // Debug output folder: group artifacts by run_id (one experiment run = one
    // folder).
    std::string debug_out_dir = debug_dir_;
    if ((debug_save_projection_png_ || debug_save_ellipsoids_json_) &&
        debug_group_by_run_id_) {
      std::string rid;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        rid = run_id_;
      }
      if (!rid.empty()) {
        debug_out_dir = debug_dir_ + "/" + rid;
        std::filesystem::create_directories(debug_out_dir);
      }
    }

    const auto t_score_start = std::chrono::steady_clock::now();

    const auto frontier_pts = mua_nbv_planner::cloudToVec(frontier);
    const auto occupied_pts = mua_nbv_planner::cloudToVec(occupied);

    std::vector<mua_nbv_planner::EllipsoidParam> ellipsoids;
    {
      const auto fcfg = tunedClusterParams(frontier_pts.size());
      const auto ocfg = tunedClusterParams(occupied_pts.size());
      auto fe = mua_nbv_planner::fitEllipsoids(
          frontier_pts, "frontier", cgal_eps_, max_points_per_cluster_,
          fcfg.min_cluster_pts, fcfg.min_gmm_k, fcfg.max_gmm_k,
          gmm_max_iters_, gmm_term_eps_, gmm_use_bic_, fcfg.gmm_max_points);
      auto oe = mua_nbv_planner::fitEllipsoids(
          occupied_pts, "occupied", cgal_eps_, max_points_per_cluster_,
          ocfg.min_cluster_pts, ocfg.min_gmm_k, ocfg.max_gmm_k,
          gmm_max_iters_, gmm_term_eps_, gmm_use_bic_, ocfg.gmm_max_points);
      ellipsoids.reserve(fe.size() + oe.size());
      ellipsoids.insert(ellipsoids.end(), fe.begin(), fe.end());
      ellipsoids.insert(ellipsoids.end(), oe.begin(), oe.end());

      if (auto_cluster_params_enable_) {
        RCLCPP_INFO(get_logger(),
                    "🧪 auto cluster frontier(n=%zu): min_pts=%zu k=[%d,%d] gmm_max_points=%d | occupied(n=%zu): min_pts=%zu k=[%d,%d] gmm_max_points=%d",
                    frontier_pts.size(), fcfg.min_cluster_pts, fcfg.min_gmm_k,
                    fcfg.max_gmm_k, fcfg.gmm_max_points,
                    occupied_pts.size(), ocfg.min_cluster_pts, ocfg.min_gmm_k,
                    ocfg.max_gmm_k, ocfg.gmm_max_points);
      }
    }

    saveEllipsoidsJson(debug_out_dir, candidates.header.stamp, ellipsoids,
                      frontier_pts.size(), occupied_pts.size());

    if (ellipsoids.empty()) {
      why = "no ellipsoids fit";
      return false;
    }

    if (static_target_) {
      try {
        const auto timeout = tf2::durationFromSec(tf_lookup_timeout_sec_);
        geometry_msgs::msg::TransformStamped tf_T_W;
        if (use_latest_tf_) {
          tf_T_W = tf_buffer_.lookupTransform(target_frame_,
                                              candidates.header.frame_id,
                                              tf2::TimePointZero, timeout);
        } else {
          const rclcpp::Time t_req(candidates.header.stamp,
                                   get_clock()->get_clock_type());
          tf_T_W = tf_buffer_.lookupTransform(target_frame_,
                                              candidates.header.frame_id,
                                              t_req, timeout);
        }
        Eigen::Isometry3d T_T_W = Eigen::Isometry3d::Identity();
        T_T_W.translation() << tf_T_W.transform.translation.x,
            tf_T_W.transform.translation.y, tf_T_W.transform.translation.z;
        Eigen::Quaterniond q(
            tf_T_W.transform.rotation.w, tf_T_W.transform.rotation.x,
            tf_T_W.transform.rotation.y, tf_T_W.transform.rotation.z);
        q.normalize();
        T_T_W.linear() = q.toRotationMatrix();

        std_msgs::msg::Float32MultiArray scores;
        scores.data.resize(candidates.poses.size(), 0.0f);

        int best_idx = -1;
        double best_score = -std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < candidates.poses.size(); ++i) {
          const Eigen::Isometry3d T_W_O =
              isoFromPoseMsg(candidates.poses[i]);       // world->optical
          const Eigen::Isometry3d T_T_O = T_T_W * T_W_O; // target->optical
          const double J = scoreView(poseMsgFromIso(T_T_O), ellipsoids);
          scores.data[i] = static_cast<float>(J);
          if (J > best_score) {
            best_score = J;
            best_idx = static_cast<int>(i);
          }
        }

        if (selection_mode_ == "random" && !candidates.poses.empty()) {
          std::uniform_int_distribution<int> dist(
              0, static_cast<int>(candidates.poses.size()) - 1);
          best_idx = dist(rng_);
          RCLCPP_INFO(get_logger(), "Random selection: idx=%d / %zu",
                      best_idx, candidates.poses.size());
        } else if (selection_mode_ == "softmax" && !candidates.poses.empty()) {
          std::vector<double> rv(scores.data.size());
          for (size_t ii = 0; ii < scores.data.size(); ++ii)
            rv[ii] = static_cast<double>(scores.data[ii]);
          int argmax_idx = best_idx;
          best_idx = softmaxSample(rv);
          RCLCPP_INFO(get_logger(),
                      "Softmax selection (T=%.3f): idx=%d (argmax=%d) / %zu",
                      softmax_temperature_, best_idx, argmax_idx,
                      candidates.poses.size());
        }

        const auto t_score_end = std::chrono::steady_clock::now();
        const double score_ms =
            std::chrono::duration<double, std::milli>(
                t_score_end - t_score_start)
                .count();
        {
          std_msgs::msg::Float64 tmsg;
          tmsg.data = score_ms;
          pub_score_time_->publish(tmsg);
        }

        // Normalize
        std::vector<double> raw(scores.data.size());
        for (size_t i = 0; i < scores.data.size(); ++i)
          raw[i] = (double)scores.data[i];
        double smin = std::numeric_limits<double>::infinity();
        double smax = -std::numeric_limits<double>::infinity();
        for (double s : raw) {
          smin = std::min(smin, s);
          smax = std::max(smax, s);
        }
        std_msgs::msg::Float32MultiArray scores_norm;
        scores_norm.data.resize(raw.size());
        const double denom = smax - smin;
        if (denom <= 1e-12 || !std::isfinite(denom)) {
          std::fill(scores_norm.data.begin(), scores_norm.data.end(), 0.5f);
        } else {
          for (size_t i = 0; i < raw.size(); ++i) {
            double v = (raw[i] - smin) / denom;
            v = std::clamp(v, 0.0, 1.0);
            scores_norm.data[i] = static_cast<float>(v);
          }
        }

        pub_scores_->publish(scores_norm);
        std_msgs::msg::Int32 b;
        b.data = best_idx;
        pub_best_->publish(b);
        if (best_idx >= 0 &&
            best_idx < static_cast<int>(candidates.poses.size())) {
          geometry_msgs::msg::PoseStamped best;
          best.header = candidates.header;
          best.pose = candidates.poses[static_cast<size_t>(best_idx)];
          pub_best_candidate_->publish(best);
        }

        // ---- Debug images (static_target path) ----
        if (debug_save_projection_png_) {
          auto saveOne = [&](int cand_i) {
            const auto &poseW = candidates.poses[cand_i];
            const Eigen::Isometry3d T_W_O =
                isoFromPoseMsg(poseW);                     // world->optical
            const Eigen::Isometry3d T_T_O = T_T_W * T_W_O; // target->optical

            // renderProjectionImage expects T_target_opt (target -> optical)
            const Eigen::Matrix4d T_target_opt = T_T_O.matrix();

            double fsum = 0.0, osum = 0.0;
            cv::Mat proj = mua_nbv_planner::renderProjectionImage(
                T_target_opt, ellipsoids, K_, image_width_, image_height_,
                depth_decay_, fsum, osum);

            // annotate
            const double score_dbg =
                (cand_i >= 0 && (size_t)cand_i < raw.size())
                    ? raw[(size_t)cand_i]
                    : 0.0;
            cv::putText(proj,
                        "cand=" + std::to_string(cand_i) +
                            " score=" + std::to_string(score_dbg),
                        cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 0), 2);
            cv::putText(proj,
                        "frontier=" + std::to_string(fsum) +
                            " occupied=" + std::to_string(osum),
                        cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 0), 2);

            const std::string path = debug_out_dir + "/projection_img_" +
                                     std::to_string(debug_seq_) + "_cand_" +
                                     std::to_string(cand_i) + ".png";

            if (cand_i == best_idx && pub_proj_img_) {
              auto msg =
                  cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", proj)
                      .toImageMsg();
              msg->header.stamp = candidates.header.stamp;
              msg->header.frame_id = candidates.header.frame_id;
              pub_proj_img_->publish(*msg);
            }

            try {
              cv::imwrite(path, proj);
            } catch (const cv::Exception &e) {
              RCLCPP_WARN(get_logger(), "cv::imwrite failed: %s", e.what());
            }
          };

          if (debug_save_all_candidates_) {
            for (int i = 0; i < static_cast<int>(candidates.poses.size());
                 ++i) {
              saveOne(i);
            }
          } else {
            if (best_idx >= 0) {
              saveOne(best_idx);
            }
          }

          debug_seq_++;
        }

        // Ring debug image
        if (pub_ring_img_) {
          cv::Mat ring = mua_nbv_planner::renderScoreRingImage(
              candidates, scores_norm, best_idx);
          auto ring_msg =
              cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", ring)
                  .toImageMsg();
          ring_msg->header.stamp = candidates.header.stamp;
          ring_msg->header.frame_id = candidates.header.frame_id;
          pub_ring_img_->publish(*ring_msg);

          // save to disk alongside projection PNGs
          if (debug_save_projection_png_) {
            const std::string path = debug_out_dir + "/score_ring_" +
                                     std::to_string(debug_seq_) + ".png";
            try {
              cv::imwrite(path, ring);
            } catch (...) {
            }
          }
        }

        RCLCPP_INFO(get_logger(),
                    "👑 [static_target] best_idx=%d best_score=%.3f (N=%zu) "
                    "score_ms=%.1f",
                    best_idx, best_score, candidates.poses.size(), score_ms);
        return true;
      } catch (const std::exception &e) {
        why = std::string("static_target TF/scoring failed: ") + e.what();
        return false;
      }
    }

    // --- MC setup (x,y,yaw) ---
    double yaw_hat = yawFromQuat(pred.pose.pose.orientation);
    {
      const double vx0 = pred_mu4_(2), vy0 = pred_mu4_(3);
      const double sp0 = std::hypot(vx0, vy0);
      if (sp0 >= v_min_)
        yaw_hat = std::atan2(vy0, vx0);
    }

    // Build 4D Gaussian
    Eigen::Vector4d mu4 = pred_mu4_;
    Eigen::Matrix4d Sigma4 = pred_Sigma4_;

    // jitter to keep PSD-ish
    for (int i = 0; i < 4; ++i)
      Sigma4(i, i) = std::max(Sigma4(i, i), 1e-12);

    Sigma4 = 0.5 * (Sigma4 + Sigma4.transpose());
    Eigen::LLT<Eigen::Matrix4d> llt4(Sigma4);
    if (llt4.info() != Eigen::Success) {
      Sigma4 += 1e-9 * Eigen::Matrix4d::Identity();
      llt4.compute(Sigma4);
      if (llt4.info() != Eigen::Success) {
        why = "prediction_state covariance not PSD (LLT failed)";
        return false;
      }
    }
    const Eigen::Matrix4d L = llt4.matrixL();

    if (use_monte_carlo_ &&
        candidates.header.frame_id != pred.header.frame_id) {
      why = "candidates frame != pred frame";
      return false;
    }

    const int N = use_monte_carlo_ ? std::max(1, mc_num_samples_) : 1;
    const size_t C = candidates.poses.size();

    // Pre-generate all MC random samples so the parallel loop is RNG-free.
    // mc_samples[i * N + n] = 4D standard-normal vector for candidate i, sample n.
    std::vector<Eigen::Vector4d> mc_samples(C * static_cast<size_t>(N));
    if (use_monte_carlo_) {
      for (auto &s : mc_samples) {
        s << stdnorm_(rng_), stdnorm_(rng_), stdnorm_(rng_), stdnorm_(rng_);
      }
    }

    std::vector<double> raw_scores(C, 0.0);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < C; ++i) {
      double J = 0.0;

      if (!use_monte_carlo_) {
        const double vx = mu4(2), vy = mu4(3);
        const double sp = std::hypot(vx, vy);
        const double yaw = (sp >= v_min_) ? std::atan2(vy, vx) : yaw_hat;

        const Eigen::Isometry3d T_W_O = isoFromXYYaw(mu4(0), mu4(1), yaw);
        const Eigen::Isometry3d T_W_C =
            isoFromPoseMsg(candidates.poses[i]);
        const Eigen::Isometry3d T_O_C = T_W_O.inverse() * T_W_C;

        J = scoreView(poseMsgFromIso(T_O_C), ellipsoids);
      } else {
        const Eigen::Isometry3d T_W_C =
            isoFromPoseMsg(candidates.poses[i]);

        double acc = 0.0;
        for (int n = 0; n < N; ++n) {
          const Eigen::Vector4d &z = mc_samples[i * static_cast<size_t>(N) + n];
          Eigen::Vector4d x = mu4 + L * z;
          const double vx = x(2), vy = x(3);
          const double sp = std::hypot(vx, vy);
          const double yaw = (sp >= v_min_) ? std::atan2(vy, vx) : yaw_hat;
          const Eigen::Isometry3d T_W_O = isoFromXYYaw(x(0), x(1), yaw);
          const Eigen::Isometry3d T_O_C = T_W_O.inverse() * T_W_C;
          acc += scoreView(poseMsgFromIso(T_O_C), ellipsoids);
        }
        J = acc / static_cast<double>(N);
      }

      raw_scores[i] = J;
    }

    std_msgs::msg::Float32MultiArray scores;
    scores.data.resize(C, 0.0f);

    int best_idx = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < C; ++i) {
      scores.data[i] = static_cast<float>(raw_scores[i]);
      if (raw_scores[i] > best_score) {
        best_score = raw_scores[i];
        best_idx = static_cast<int>(i);
      }
    }

    if (selection_mode_ == "random" && !candidates.poses.empty()) {
      std::uniform_int_distribution<int> dist(
          0, static_cast<int>(candidates.poses.size()) - 1);
      best_idx = dist(rng_);
      RCLCPP_INFO(get_logger(), "Random selection: idx=%d / %zu", best_idx,
                  candidates.poses.size());
    } else if (selection_mode_ == "softmax" && !candidates.poses.empty()) {
      int argmax_idx = best_idx;
      best_idx = softmaxSample(raw_scores);
      RCLCPP_INFO(get_logger(),
                  "Softmax selection (T=%.3f): idx=%d (argmax=%d) / %zu",
                  softmax_temperature_, best_idx, argmax_idx,
                  candidates.poses.size());
    }

    const auto t_score_end = std::chrono::steady_clock::now();
    const double score_ms =
        std::chrono::duration<double, std::milli>(t_score_end - t_score_start)
            .count();
    {
      std_msgs::msg::Float64 tmsg;
      tmsg.data = score_ms;
      pub_score_time_->publish(tmsg);
    }

    // Normalize the scores (reuse raw_scores from parallel loop)
    const auto &raw = raw_scores;

    double smin = std::numeric_limits<double>::infinity();
    double smax = -std::numeric_limits<double>::infinity();
    for (double s : raw) {
      smin = std::min(smin, s);
      smax = std::max(smax, s);
    }

    std_msgs::msg::Float32MultiArray scores_norm;
    scores_norm.data.resize(raw.size());

    const double denom = smax - smin;
    if (denom <= 1e-12 || !std::isfinite(denom)) {
      std::fill(scores_norm.data.begin(), scores_norm.data.end(), 0.5f);
    } else {
      for (size_t i = 0; i < raw.size(); ++i) {
        double v = (raw[i] - smin) / denom;
        v = std::clamp(v, 0.0, 1.0);
        scores_norm.data[i] = static_cast<float>(v);
      }
    }

    // ---- Debug images ----
    if (debug_save_projection_png_) {
      auto saveOne = [&](int cand_i) {
        const auto &poseW = candidates.poses[cand_i];
        const Eigen::Isometry3d T_W_C = isoFromPoseMsg(poseW);

        // Use mean object pose for debug visualization
        const double vx0 = pred_mu4_(2), vy0 = pred_mu4_(3);
        const double sp0 = std::hypot(vx0, vy0);
        const double yaw_dbg = (sp0 >= v_min_)
                                   ? std::atan2(vy0, vx0)
                                   : yawFromQuat(pred.pose.pose.orientation);
        const Eigen::Isometry3d T_W_O_dbg =
            isoFromXYYaw(pred_mu4_(0), pred_mu4_(1), yaw_dbg);
        const Eigen::Isometry3d T_O_C =
            T_W_O_dbg.inverse() *
            T_W_C; // target/base_link->pursuer/camera_link

        // Candidate poses are target->optical already
        Eigen::Matrix4d T_target_opt = Eigen::Matrix4d::Identity();
        T_target_opt.block<3, 3>(0, 0) = T_O_C.linear();
        T_target_opt(0, 3) = T_O_C.translation().x();
        T_target_opt(1, 3) = T_O_C.translation().y();
        T_target_opt(2, 3) = T_O_C.translation().z();

        double fsum = 0.0, osum = 0.0;
        cv::Mat proj = mua_nbv_planner::renderProjectionImage(
            T_target_opt, ellipsoids, K_, image_width_, image_height_,
            depth_decay_, fsum, osum);
        const double score_dbg = fsum - osum;
        cv::putText(proj,
                    "cand=" + std::to_string(cand_i) +
                        " score=" + std::to_string(score_dbg),
                    cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 2);

        cv::putText(proj,
                    "frontier=" + std::to_string(fsum) +
                        " occupied=" + std::to_string(osum),
                    cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 2);

        const std::string path = debug_out_dir + "/projection_img_" +
                                 std::to_string(debug_seq_) + "_cand_" +
                                 std::to_string(cand_i) + ".png";

        if (cand_i == best_idx && pub_proj_img_) {
          auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", proj)
                         .toImageMsg();
          msg->header.stamp = candidates.header.stamp;
          msg->header.frame_id = candidates.header.frame_id;
          pub_proj_img_->publish(*msg);
        }

        try {
          cv::imwrite(path, proj);
        } catch (const cv::Exception &e) {
          RCLCPP_WARN(get_logger(), "cv::imwrite failed: %s", e.what());
        }
      };

      if (debug_save_all_candidates_) {
        for (int i = 0; i < static_cast<int>(candidates.poses.size()); ++i)
          saveOne(i);
      } else {
        if (best_idx >= 0)
          saveOne(best_idx);
      }

      debug_seq_++;
    }

    pub_scores_->publish(scores_norm);

    std_msgs::msg::Int32 b;
    b.data = best_idx;
    pub_best_->publish(b);

    if (best_idx >= 0 && best_idx < static_cast<int>(candidates.poses.size())) {
      geometry_msgs::msg::PoseStamped best;
      best.header = candidates.header;
      best.pose = candidates.poses[static_cast<size_t>(best_idx)];
      pub_best_candidate_->publish(best);
    }

    // Ring debug image
    if (pub_ring_img_) {
      cv::Mat ring = mua_nbv_planner::renderScoreRingImage(
          candidates, scores_norm, best_idx);
      auto ring_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", ring)
                          .toImageMsg();
      ring_msg->header.stamp = candidates.header.stamp;
      ring_msg->header.frame_id = candidates.header.frame_id;
      pub_ring_img_->publish(*ring_msg);

      // save to disk alongside projection PNGs
      if (debug_save_projection_png_) {
        const std::string path = debug_out_dir + "/score_ring_" +
                                 std::to_string(debug_seq_) + ".png";
        try {
          cv::imwrite(path, ring);
        } catch (...) {
        }
      }
    }

    RCLCPP_INFO(
        get_logger(),
        "💯 cands=%zu mc=%d evals=%zu ellipsoids=%zu best_idx=%d best_score=%.3f score_ms=%.1f",
        C, N, C * static_cast<size_t>(N), ellipsoids.size(), best_idx, best_score,
        score_ms);
    return true;
  }

  void onPrediction(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    pred_ = *msg;
    have_pred_ = true;
    pred_seq_++;
    cv_.notify_all();
  }

  void
  onPredictionState(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    if (!msg || msg->data.size() != 22)
      return;

    std::lock_guard<std::mutex> lk(mtx_);

    // Stamp is encoded in the message
    const int32_t sec = static_cast<int32_t>(std::lround(msg->data[0]));
    const uint32_t nsec = static_cast<uint32_t>(std::lround(msg->data[1]));
    pred_state_stamp_ = rclcpp::Time(sec, nsec, get_clock()->get_clock_type());

    // Mean [x,y,vx,vy]
    pred_mu4_ << msg->data[2], msg->data[3], msg->data[4], msg->data[5];

    // Cov 4x4 row-major
    int k = 6;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        pred_Sigma4_(r, c) = static_cast<double>(msg->data[k++]);
      }
    }

    // Symmetrize + clamp diagonals (robust)
    pred_Sigma4_ = 0.5 * (pred_Sigma4_ + pred_Sigma4_.transpose());
    for (int i = 0; i < 4; ++i)
      pred_Sigma4_(i, i) = std::max(pred_Sigma4_(i, i), 1e-12);

    have_pred_state_ = true;
    cv_.notify_all();
  }

  std::string step_stamp_topic_, prediction_topic_, prediction_state_topic_;
  std::string candidates_topic_, frontier_topic_, occupied_topic_;
  std::string scores_topic_, best_idx_topic_, best_candidate_topic_,
      proj_img_topic_, ring_img_topic_;
  std::string score_service_;
  bool static_target_{false};
  std::string target_frame_{"target/base_link"};
  double tf_lookup_timeout_sec_{0.2};
  bool use_latest_tf_{false};

  rclcpp::CallbackGroup::SharedPtr cb_sub_;
  rclcpp::Subscription<builtin_interfaces::msg::Time>::SharedPtr
      sub_step_stamp_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      sub_pred_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
      sub_pred_state_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr
      sub_candidates_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_frontier_,
      sub_occupied_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_scores_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_best_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      pub_best_candidate_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_proj_img_,
      pub_ring_img_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_score_time_;

  rclcpp::CallbackGroup::SharedPtr cb_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_score_;

  geometry_msgs::msg::PoseWithCovarianceStamped pred_;
  bool have_pred_{false};
  uint64_t pred_seq_{0};

  double pred_sync_tol_sec_{0.0};

  bool have_pred_state_{false};
  rclcpp::Time pred_state_stamp_{0, 0, RCL_ROS_TIME};
  Eigen::Vector4d pred_mu4_{Eigen::Vector4d::Zero()};
  Eigen::Matrix4d pred_Sigma4_{Eigen::Matrix4d::Zero()};

  // v_min for heading
  double v_min_{0.0};
  bool use_monte_carlo_{true};
  int mc_num_samples_{16};
  int mc_seed_{0};
  std::mt19937 rng_{0};
  std::normal_distribution<double> stdnorm_{0.0, 1.0};
  std::string selection_mode_{"argmax"};
  double softmax_temperature_{0.2};

  std::condition_variable cv_;
  rclcpp::Time required_stamp_{0, 0, RCL_ROS_TIME};
  bool have_required_stamp_{false};
  rclcpp::Time last_processed_stamp_{0, 0, RCL_ROS_TIME};
  bool require_new_inputs_{true};
  double wait_for_inputs_sec_{0.5};

  // --- State ---
  std::mutex mtx_;
  geometry_msgs::msg::PoseArray candidates_;
  sensor_msgs::msg::PointCloud2 frontier_cloud_, occupied_cloud_;
  bool have_frontier_{false}, have_occupied_{false};

  tf2_ros::Buffer tf_buffer_{this->get_clock()};
  tf2_ros::TransformListener tf_listener_{tf_buffer_};

  // --- GMM clustering  ---
  int min_gmm_cluster_num_{5};
  int max_gmm_cluster_num_{20};
  int gmm_max_iters_{100};
  double gmm_term_eps_{1e-3};
  bool gmm_use_bic_{true};
  int gmm_max_points_{20000};

  bool auto_cluster_params_enable_{false};
  int auto_min_cluster_pts_floor_{8};
  int auto_cluster_k_cap_{50};
  int auto_gmm_min_points_{200};

  int image_width_{640}, image_height_{480};
  double fx_{554.38}, fy_{554.38}, cx_{320.0}, cy_{240.0};
  Eigen::Matrix3d K_{Eigen::Matrix3d::Identity()};

  double voxel_res_{0.05};
  size_t min_cluster_pts_{20};
  size_t max_points_per_cluster_{2000};
  double cgal_eps_{0.01};
  double depth_decay_{0.5};

  // --- debug image dump ---
  bool debug_save_projection_png_{true};
  bool debug_save_all_candidates_{true};
  bool debug_save_ellipsoids_json_{true};
  std::string debug_dir_{"debug/mua_nbv_planner"};
  bool debug_group_by_run_id_{true};
  std::string run_id_topic_{"/experiment/run_id"};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_run_id_;
  std::string run_id_;
  uint64_t debug_seq_{0};

  // token debug
  bool have_last_token_logged_{false};
  rclcpp::Time last_token_logged_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScorerNode>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}