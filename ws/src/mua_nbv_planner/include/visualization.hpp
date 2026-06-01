#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_array.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "ellipsoid_clustering.hpp" // For EllipsoidParam

namespace mua_nbv_planner {

cv::Mat renderProjectionImage(const Eigen::Matrix4d &T_target_opt,
                              const std::vector<EllipsoidParam> &ellipsoids,
                              const Eigen::Matrix3d &K, int W, int H,
                              double depth_decay, double &frontier_sum_out,
                              double &occupied_sum_out) {
  frontier_sum_out = 0.0;
  occupied_sum_out = 0.0;

  cv::Mat union_frontier(H, W, CV_8UC1, cv::Scalar(0));
  cv::Mat union_occupied(H, W, CV_8UC1, cv::Scalar(0));
  cv::Mat outlines(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

  // P = K [R|t], where [R|t] is (optical <- target)
  const Eigen::Matrix4d T_opt_target = T_target_opt.inverse();
  Eigen::Matrix<double, 3, 4> Rt = T_opt_target.block<3, 4>(0, 0);
  Eigen::Matrix<double, 3, 4> P = K * Rt;

  // Depth order for weights (z in optical frame; +Z forward)
  struct ZItem {
    size_t idx;
    double z;
  };
  std::vector<ZItem> order;
  order.reserve(ellipsoids.size());

  for (size_t i = 0; i < ellipsoids.size(); ++i) {
    Eigen::Vector4d c(ellipsoids[i].pose(0, 3), ellipsoids[i].pose(1, 3),
                      ellipsoids[i].pose(2, 3), 1.0);
    Eigen::Vector3d c_opt = (T_opt_target * c).head<3>();
    if (!c_opt.allFinite())
      continue;
    order.push_back({i, c_opt.z()});
  }
  std::sort(order.begin(), order.end(),
            [](const ZItem &a, const ZItem &b) { return a.z < b.z; });

  std::vector<double> w(ellipsoids.size(), 0.0);
  double cur = 1.0;
  for (size_t r = 0; r < order.size(); ++r) {
    w[order[r].idx] = cur;
    cur *= depth_decay; // 0.5^rank (depth proxy / occlusion proxy)
  }

  // Draw filled ellipse masks and accumulate weighted pixel counts
  for (size_t i = 0; i < ellipsoids.size(); ++i) {
    if (w[i] <= 0.0)
      continue;

    const Eigen::Matrix4d Qdual = createEllipsoidDualMatrix(ellipsoids[i]);
    if (Qdual.isZero(0))
      continue;

    const Eigen::Matrix3d conic = projectDualQuadricToConic(P, Qdual);
    if (conic.isZero(0))
      continue;

    cv::Point2d center;
    cv::Size2d axes;
    double angle_deg = 0.0;

    if (!conicToEllipse(conic, center, axes, angle_deg))
      continue;

    // reject nonsense
    if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
        !std::isfinite(axes.width) || !std::isfinite(axes.height))
      continue;
    if (axes.width < 1.0 || axes.height < 1.0)
      continue;

    // outline color
    const cv::Scalar col = (ellipsoids[i].type == "frontier")
                               ? cv::Scalar(0, 0, 255)  // red
                               : cv::Scalar(255, 0, 0); // blue

    // draw outline (thicker so it shows)
    cv::ellipse(outlines, center, axes, angle_deg, 0.0, 360.0, col, 2);

    // label index at center (clamped)
    cv::Point ptxt((int)std::lround(center.x), (int)std::lround(center.y));
    ptxt.x = std::clamp(ptxt.x, 0, W - 1);
    ptxt.y = std::clamp(ptxt.y, 0, H - 1);
    cv::putText(outlines, std::to_string(i), ptxt, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 0, 0), 2);
    cv::putText(outlines, std::to_string(i), ptxt, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 255), 1);

    cv::Mat mask(H, W, CV_8UC1, cv::Scalar(0));
    cv::ellipse(mask, center, axes, angle_deg, 0.0, 360.0, cv::Scalar(255), -1);

    const double pix = static_cast<double>(cv::countNonZero(mask));
    const double contrib = pix * w[i];

    if (ellipsoids[i].type == "frontier") {
      frontier_sum_out += contrib;
      cv::bitwise_or(union_frontier, mask, union_frontier);
    } else if (ellipsoids[i].type == "occupied") {
      occupied_sum_out += contrib;
      cv::bitwise_or(union_occupied, mask, union_occupied);
    }
  }

  // Build final colored image
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

  // occupied = blue (BGR)
  img.setTo(cv::Scalar(255, 0, 0), union_occupied);

  // frontier = red
  img.setTo(cv::Scalar(0, 0, 255), union_frontier);

  // overlap = magenta
  cv::Mat overlap;
  cv::bitwise_and(union_frontier, union_occupied, overlap);
  img.setTo(cv::Scalar(255, 0, 255), overlap);

  cv::addWeighted(img, 0.90, outlines, 0.25, 0.0, img);
  return img;
}

cv::Mat renderScoreRingImage(const geometry_msgs::msg::PoseArray &candidates,
                             const std_msgs::msg::Float32MultiArray &scores,
                             int best_idx, int W = 800, int H = 800) {
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  const cv::Point2d c(W * 0.5, H * 0.5);

  // Outer ring
  const int R = (int)(0.40 * std::min(W, H)); // ring radius in pixels
  cv::circle(img, c, R, cv::Scalar(0, 0, 0), 2);

  // Important:
  // Candidates are often in a world frame (not centered at (0,0)), especially
  // in dynamic mode. Using atan2(y, x) would compute angles around the world
  // origin and makes the plot look "clustered". Instead compute angles around
  // the ring center, approximated by the candidate centroid.
  double cxw = 0.0, cyw = 0.0;
  int ncenter = 0;
  for (const auto &p : candidates.poses) {
    if (std::isfinite(p.position.x) && std::isfinite(p.position.y)) {
      cxw += p.position.x;
      cyw += p.position.y;
      ncenter++;
    }
  }
  if (ncenter > 0) {
    cxw /= static_cast<double>(ncenter);
    cyw /= static_cast<double>(ncenter);
  } else {
    cxw = 0.0;
    cyw = 0.0;
  }
  // Score range (optional, for small legend)
  float smin = std::numeric_limits<float>::infinity();
  float smax = -std::numeric_limits<float>::infinity();
  for (float s : scores.data) {
    smin = std::min(smin, s);
    smax = std::max(smax, s);
  }

  // Draw each candidate
  const int N = (int)candidates.poses.size();
  for (int i = 0; i < N; ++i) {
    const auto &p = candidates.poses[i].position;
    const double theta = std::atan2(p.y - cyw, p.x - cxw);

    const double R_i = R; // <-- fixed ring radius

    const cv::Point2d pt(c.x + R_i * std::cos(theta),
                         c.y - R_i * std::sin(theta) // y+ goes up in the image
    );

    const bool is_best = (i == best_idx);

    // point
    cv::circle(img, pt, is_best ? 7 : 4,
               is_best ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 0), -1);

    // label "i:score"
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(3);
    ss << i << ":" << scores.data[i];

    // small offset so text doesn't sit on the dot
    cv::Point text_pt((int)pt.x + 8, (int)pt.y - 8);
    cv::putText(img, ss.str(), text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 2);
  }

  // Legend
  {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(3);
    ss << "N=" << N << "  best=" << best_idx << "  score_range=[" << smin << ","
       << smax << "]";
    cv::putText(img, ss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 0), 2);
  }
  return img;
}

} // namespace mua_nbv_planner