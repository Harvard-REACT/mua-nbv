#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

// CGAL for ellipsoid fitting
#include <CGAL/Cartesian_d.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Approximate_min_ellipsoid_d.h>
#include <CGAL/Approximate_min_ellipsoid_d_traits_d.h>

namespace mua_nbv_planner {

// check if all three values are finite
inline bool finite3(double x, double y, double z) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

// Structure for ellipsoid parameters
struct EllipsoidParam {
  std::string type;      // "frontier" or "occupied"
  Eigen::Matrix4d pose;  // target->ellipsoid (rotation + translation)
  Eigen::Vector3d radii; // semi-axes lengths
};

// Convert roll-pitch-yaw to rotation matrix
inline Eigen::Matrix3d rpyToR(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll),  sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw),   sy = std::sin(yaw);
  Eigen::Matrix3d R;
  R << cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
       sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
       -sp,   cp*sr,            cp*cr;
  return R;
}  

// Convert PointCloud2 to vector of Eigen::Vector3d
std::vector<Eigen::Vector3d> cloudToVec(const sensor_msgs::msg::PointCloud2& msg) {
  const size_t npts = (size_t)msg.width * (size_t)msg.height;
  std::vector<Eigen::Vector3d> out;
  out.reserve(npts);

  sensor_msgs::PointCloud2ConstIterator<float> it_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(msg, "z");

  for (size_t i=0; i<npts; ++i, ++it_x, ++it_y, ++it_z) {
    const double x = (double)(*it_x);
    const double y = (double)(*it_y);
    const double z = (double)(*it_z);
    if (!finite3(x,y,z)) continue;
    out.emplace_back(x,y,z);
  }
  return out;
}

// Create dual quadric matrix for ellipsoid, then project to dual conic.
inline Eigen::Matrix4d createEllipsoidDualMatrix(const EllipsoidParam& param) {
  Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
  const Eigen::Vector3d radii2 = param.radii.array().square();
  if ((radii2.array() <= 1e-12).any()) return Eigen::Matrix4d::Zero();

  const Eigen::Vector3d inv = radii2.array().inverse();
  Q.block<3,3>(0,0) = inv.asDiagonal();
  Q(3,3) = -1.0;

  const double det = Q.determinant();
  if (std::abs(det) < 1e-18) return Eigen::Matrix4d::Zero();

  // dual in ellipsoid frame
  Eigen::Matrix4d Qdual0 = Q.inverse();

  // transform to target frame
  return param.pose * Qdual0 * param.pose.transpose();
}
  
  // Project dual quadric to conic
inline Eigen::Matrix3d projectDualQuadricToConic(
  const Eigen::Matrix<double,3,4>& P, const Eigen::Matrix4d& Qdual)
{
  Eigen::Matrix3d Cdual = P * Qdual * P.transpose();
  const double det = Cdual.determinant();
  if (std::abs(det) < 1e-18) return Eigen::Matrix3d::Zero();
  // conic matrix (primal)
  return Cdual.inverse();
}
 
// Convert conic to OpenCV ellipse using eigendecomposition
inline bool conicToEllipse(const Eigen::Matrix3d& C_in, cv::Point2d& center_px,
                           cv::Size2d& axes_px, double& angle_deg)
{
  // Symmetrize for numeric stability
  Eigen::Matrix3d C = 0.5 * (C_in + C_in.transpose());

  // Expand: A u^2 + B u v + C v^2 + D u + E v + F = 0
  const double A = C(0,0);
  const double B = 2.0 * C(0,1);
  const double Cc = C(1,1);
  const double D = 2.0 * C(0,2);
  const double E = 2.0 * C(1,2);
  const double F = C(2,2);
  Eigen::Matrix2d Q;
  Q << A, B/2.0,
       B/2.0, Cc;

  const double detQ = Q.determinant();
  if (std::abs(detQ) < 1e-18) return false;

  Eigen::Vector2d d(D, E);

  // center = -0.5 Q^{-1} d
  Eigen::Vector2d c = -0.5 * Q.inverse() * d;

  // F' after completing square
  const double Fp = F - 0.25 * d.transpose() * Q.inverse() * d;

  // Eigen-decompose Q (principal directions)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Q);
  if (es.info() != Eigen::Success) return false;

  Eigen::Vector2d eval = es.eigenvalues();
  Eigen::Matrix2d evec = es.eigenvectors(); // columns are eigenvectors

  // Semi-axis lengths: sqrt(-Fp / lambda)
  const double l0 = eval(0);
  const double l1 = eval(1);

  if (!std::isfinite(Fp) || !std::isfinite(l0) || !std::isfinite(l1)) return false;

  const double a2 = -Fp / l0;
  const double b2 = -Fp / l1;
  if (!(a2 > 1e-12) || !(b2 > 1e-12)) return false;

  double a = std::sqrt(a2);
  double b = std::sqrt(b2);

  // Pick major axis as larger radius
  Eigen::Vector2d major_dir = evec.col(0);
  if (b > a) {
    std::swap(a, b);
    major_dir = evec.col(1);
  }

  const double ang = std::atan2(major_dir.y(), major_dir.x());
  angle_deg = ang * 180.0 / M_PI;

  center_px = cv::Point2d(c.x(), c.y());
  axes_px = cv::Size2d(a, b); // OpenCV expects semi-axes
  return true;
}

// GMM-based clustering of 3D points
std::vector<std::vector<Eigen::Vector3d>> gmmClustering(
    const std::vector<Eigen::Vector3d>& voxels,
    int min_gmm_cluster_num, int max_gmm_cluster_num,
    int gmm_max_iters, double gmm_term_eps,
    bool gmm_use_bic, int gmm_max_points, size_t min_cluster_pts)
{
  // GMM-based clustering of 3D points
  std::vector<std::vector<Eigen::Vector3d>> clustered;
  const int N0 = static_cast<int>(voxels.size());
  if (N0 <= 0) return clustered;
  
  // Subsample for EM speed (uniform stride)
  const int Ncap = std::max(1, gmm_max_points);
  const int stride = (N0 > Ncap) ? std::max(1, N0 / Ncap) : 1;
  
  std::vector<Eigen::Vector3d> pts;
  pts.reserve((N0 + stride - 1) / stride);
  for (int i = 0; i < N0; i += stride) pts.push_back(voxels[i]);
  
  const int N = static_cast<int>(pts.size());
  if (N < 4) {
    clustered.resize(1);
    clustered[0] = pts;
    return clustered;
  }
  
  cv::Mat samples(N, 3, CV_32FC1);
  for (int i = 0; i < N; ++i) {
    samples.at<float>(i, 0) = static_cast<float>(pts[i].x());
    samples.at<float>(i, 1) = static_cast<float>(pts[i].y());
    samples.at<float>(i, 2) = static_cast<float>(pts[i].z());
  }
  
  auto makeEM = [&](int K) -> cv::Ptr<cv::ml::EM> {
    cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
    em->setClustersNumber(K); 
    em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
    em->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                         std::max(1, gmm_max_iters), gmm_term_eps));
    return em;
  };
  
  int maxK = std::min(std::max(1, max_gmm_cluster_num), N);
  int minK = std::min(std::max(1, min_gmm_cluster_num), maxK);
  
  int chosenK = 1;
  cv::Ptr<cv::ml::EM> best_em;
  
  // choose K by BIC when enough samples
  if (gmm_use_bic && N >= 2 * maxK && minK >= 2) {
    double best_val = std::numeric_limits<double>::infinity();
  
    for (int K = minK; K <= maxK; ++K) {
      if (N < K) continue;
  
      auto em = makeEM(K);
      const bool ok = em->trainEM(samples);
      if (!ok) continue;
  
      double ll = 0.0;
      // Use true log-likelihood from predict2  
      for (int i = 0; i < N; ++i) {
        cv::Mat probs;
        cv::Vec2d r = em->predict2(samples.row(i), probs);
        ll += r[0];  // log-likelihood
      }
  
      // BIC-ish criterion
      const double val = 3.0 * std::log(static_cast<double>(N)) - 2.0 * ll;
  
      if (val < best_val) {
        best_val = val;
        chosenK = K;
        best_em = em;
      }
    }
  }
  
  // Fallback if BIC selection not used or failed
  if (!best_em) {
    if (N > 3) chosenK = std::min(maxK, std::max(2, N / 2)); 
    else chosenK = 1; 
    best_em = makeEM(chosenK);
    best_em->trainEM(samples);
  }
  
  clustered.assign(chosenK, {});
  
  // Assign each sample to most probable component (predict2 returns argmax component index)
  for (int i = 0; i < N; ++i) {
    cv::Mat probs;
    cv::Vec2d r = best_em->predict2(samples.row(i), probs);
    int k = static_cast<int>(r[1]);
    if (k < 0 || k >= chosenK) k = 0;
    clustered[k].push_back(pts[i]);
  }
  
  // Drop tiny clusters  
  std::vector<std::vector<Eigen::Vector3d>> filtered;
  filtered.reserve(clustered.size());
  for (auto& c : clustered) {
    if (c.size() >= min_cluster_pts) filtered.push_back(std::move(c)); 
  }
  
  return filtered; 
}

// Fit ellipsoids with CGAL from clustered points
std::vector<EllipsoidParam> fitEllipsoids(
    const std::vector<Eigen::Vector3d>& pts,
    const std::string& type, double cgal_eps,
    size_t max_points_per_cluster, size_t min_cluster_pts,
    int min_gmm_cluster_num, int max_gmm_cluster_num,
    int gmm_max_iters, double gmm_term_eps,
    bool gmm_use_bic, int gmm_max_points) 
{
  typedef CGAL::Cartesian_d<double>                              Kernel;
  typedef CGAL::MP_Float                                         ET;
  typedef CGAL::Approximate_min_ellipsoid_d_traits_d<Kernel, ET> Traits;
  typedef Traits::Point                                          Point;
  typedef std::vector<Point>                                     Point_list;
  typedef CGAL::Approximate_min_ellipsoid_d<Traits>              AME;

  Traits traits;
  const int d = 3;

  std::vector<EllipsoidParam> out;

  const auto clusters = gmmClustering(pts, min_gmm_cluster_num, max_gmm_cluster_num, gmm_max_iters,
                                      gmm_term_eps, gmm_use_bic, gmm_max_points, min_cluster_pts);
  out.reserve(clusters.size());

  for (const auto& cluster : clusters) {
    // Subsample cluster to keep CGAL fast
    const size_t n = cluster.size();
    const size_t stride = std::max<size_t>(1, (n + max_points_per_cluster - 1) / max_points_per_cluster);

    Point_list plist;
    plist.reserve(std::min(n, max_points_per_cluster));

    for (size_t i=0; i<n; i+=stride) {
      const auto& p = cluster[i];
      std::vector<double> v = {p.x(), p.y(), p.z()};
      plist.push_back(Point(d, v.begin(), v.end()));
    }

    if (plist.size() < 4) continue;

    // Rank check: skip near-degenerate (coplanar/collinear) clusters that
    // cause CGAL's exact-arithmetic MEE to converge extremely slowly.
    {
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const auto& pt : cluster) mean += pt;
      mean /= static_cast<double>(cluster.size());
      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      const size_t ns = std::min(n, max_points_per_cluster);
      for (size_t ii = 0; ii < n; ii += stride) {
        Eigen::Vector3d diff = cluster[ii] - mean;
        cov += diff * diff.transpose();
      }
      cov /= static_cast<double>(ns);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
      Eigen::Vector3d ev = eig.eigenvalues();
      double max_ev = ev.maxCoeff();
      if (max_ev < 1e-12 || ev.minCoeff() / max_ev < 1e-4) continue;
    }

    AME mel(cgal_eps, plist.begin(), plist.end(), traits);
    if (!(mel.is_full_dimensional() && d == 3)) continue;

    auto radii = mel.axes_lengths_begin();
    auto centroid = mel.center_cartesian_begin();
    auto dir0 = mel.axis_direction_cartesian_begin(0);
    auto dir1 = mel.axis_direction_cartesian_begin(1);
    auto dir2 = mel.axis_direction_cartesian_begin(2);

    EllipsoidParam e;
    e.type = type;
    e.pose = Eigen::Matrix4d::Identity();
    e.pose.block<3,1>(0,3) = Eigen::Vector3d(centroid[0], centroid[1], centroid[2]);
    e.pose.block<3,3>(0,0) =
    (Eigen::Matrix3d() << dir0[0], dir1[0], dir2[0],
                          dir0[1], dir1[1], dir2[1],
                          dir0[2], dir1[2], dir2[2]).finished();
    e.radii = Eigen::Vector3d(radii[0], radii[1], radii[2]);

    // Basic sanity: radii finite and not absurdly tiny
    if (!e.radii.allFinite()) continue;
    if ((e.radii.array() < 1e-4).any()) continue;

    out.push_back(e);
  }

  return out;
}

} // namespace mua_nbv_planner