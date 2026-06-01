#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

namespace mua_nbv_planner {

static inline double deg2rad(double d) { return d * M_PI / 180.0; }

// Wrap angle to [-pi, pi].
static inline double wrapAngleRad(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

// Smallest absolute difference between two angles (radians), in [0, pi].
static inline double angleDiffAbsRad(double a, double b) {
  return std::abs(wrapAngleRad(a - b));
}

struct EllipseAxes {
  double a_m{0.1}; // along heading axis
  double b_m{0.1}; // lateral axis
};

// Compute heading-aligned ellipse axes from planar covariance.
inline EllipseAxes headingAlignedEllipseAxes(const Eigen::Matrix2d &Sigma_p_in,
                                             double heading_yaw_rad,
                                             double r_off_m, double kappa,
                                             double min_axis_m = 0.1) {
  EllipseAxes out;
  // Preconditions (handled gracefully):
  // - r_off_m > 0
  // - kappa > 0
  // - min_axis_m > 0

  Eigen::Matrix2d Sigma_p = 0.5 * (Sigma_p_in + Sigma_p_in.transpose());

  const double c = std::cos(heading_yaw_rad);
  const double s = std::sin(heading_yaw_rad);
  Eigen::Matrix2d R;
  R << c, -s, s, c;

  const Eigen::Matrix2d Sigma_tilde = R.transpose() * Sigma_p * R;

  // Ignore off-diagonals per spec; guard against small negative due to
  // numerical issues.
  const double var_par = std::max(0.0, Sigma_tilde(0, 0));
  const double var_perp = std::max(0.0, Sigma_tilde(1, 1));
  const double sigma_parallel = std::sqrt(var_par);
  const double sigma_perp = std::sqrt(var_perp);

  const double a = r_off_m + kappa * sigma_parallel;
  const double b = r_off_m + kappa * sigma_perp;

  const double amin = std::max(1e-9, min_axis_m);
  out.a_m = std::max(amin, a);
  out.b_m = std::max(amin, b);
  return out;
}

struct RingParams {
  double radius_m = 2.0;
  int num_points = 4;

  // Interpreted as [start, end) for N>1 to avoid duplicate endpoints
  double angle_start_deg = 0.0;
  double angle_end_deg = 360.0;

  double center_x = 0.0;
  double center_y = 0.0;
};

class RingGenerator {
public:
  explicit RingGenerator(RingParams p) : p_(std::move(p)) {
    p_.num_points = std::max(1, p_.num_points);
    p_.radius_m = std::max(0.0, p_.radius_m);
  }

  // Returns candidate camera poses as T_target_cam (camera pose expressed in
  // target frame)
  std::vector<Eigen::Isometry3d> generate(double z_height_m) const {
    std::vector<Eigen::Isometry3d> out;
    out.reserve(static_cast<size_t>(p_.num_points));

    const double a0 = deg2rad(p_.angle_start_deg);
    const double a1 = deg2rad(p_.angle_end_deg);

    // Look-at center in target frame
    const Eigen::Vector3d center(p_.center_x, p_.center_y, 0.0);

    const int N = p_.num_points;

    // IMPORTANT: sample [start, end) to avoid duplicate at 360 deg
    for (int i = 0; i < N; ++i) {
      const double s =
          (N == 1)
              ? 0.0
              : (static_cast<double>(i) / static_cast<double>(N)); // NOT (N-1)
      const double a = a0 + s * (a1 - a0);

      const double x = center.x() + p_.radius_m * std::cos(a);
      const double y = center.y() + p_.radius_m * std::sin(a);
      const Eigen::Vector3d cam_pos(x, y, z_height_m);

      const double dx = center.x() - x;
      const double dy = center.y() - y;

      double yaw = 0.0;
      if (std::abs(dx) > 1e-9 || std::abs(dy) > 1e-9) {
        yaw = std::atan2(dy, dx);
      }

      const Eigen::Matrix3d R =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.linear() = R;
      T.translation() = cam_pos;
      out.push_back(T);
    }

    return out;
  }

private:
  RingParams p_;
};

struct EllipseParams {
  double a_m = 2.0; // semi-axis along +X in the target frame
  double b_m = 2.0; // semi-axis along +Y in the target frame
  int num_points = 4;

  double center_x = 0.0;
  double center_y = 0.0;
};

class EllipseGenerator {
public:
  explicit EllipseGenerator(EllipseParams p) : p_(std::move(p)) {
    p_.num_points = std::max(1, p_.num_points);
    // Clamp to non-degenerate axes.
    p_.a_m = std::max(0.0, p_.a_m);
    p_.b_m = std::max(0.0, p_.b_m);
  }

  // Returns candidate base poses as T_target_base (pose expressed in target
  // frame). Candidate yaw always faces the center (inward looking).
  std::vector<Eigen::Isometry3d> generate(double z_height_m) const {
    std::vector<Eigen::Isometry3d> out;
    out.reserve(static_cast<size_t>(p_.num_points));

    const Eigen::Vector3d center(p_.center_x, p_.center_y, 0.0);
    const int N = p_.num_points;

    // Sample phi in [0, 2pi) as phi_i = 2*pi*i/N to avoid duplicate endpoint.
    for (int i = 0; i < N; ++i) {
      const double s =
          (N == 1) ? 0.0 : (static_cast<double>(i) / static_cast<double>(N));
      const double phi = 2.0 * M_PI * s;

      const double x = center.x() + p_.a_m * std::cos(phi);
      const double y = center.y() + p_.b_m * std::sin(phi);
      const Eigen::Vector3d pos(x, y, z_height_m);

      const double dx = center.x() - x;
      const double dy = center.y() - y;

      double yaw = 0.0;
      if (std::abs(dx) > 1e-9 || std::abs(dy) > 1e-9) {
        yaw = std::atan2(dy, dx);
      }

      const Eigen::Matrix3d Rz =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.linear() = Rz;
      T.translation() = pos;
      out.push_back(T);
    }
    return out;
  }

private:
  EllipseParams p_;
};

} // namespace mua_nbv_planner
