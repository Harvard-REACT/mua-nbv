# mua_nbv_planner

C++ Next-Best-View planner for MUA-NBV. Provides three service-driven ROS 2
nodes that form the core planning pipeline.

## Nodes

### `voxel_map_node`

Maintains a semantic OctoMap of the target object. Inserts depth-cloud points
as **occupied**, raycasts to carve **free** and **unseen** space, and extracts
a **frontier** (voxels at the boundary of known and unknown regions).

- **Input**: `sensor_msgs/PointCloud2` (target-frame depth cloud)
- **Output**: `PointCloud2` for occupied and frontier voxels
- **Service**: `std_srvs/Trigger` — update voxels from the latest buffered cloud

### `ring_generator_node`

Generates candidate viewpoints on an ellipse (or ring) centered on the
predicted target position.  The ellipse axes are derived from the prediction
covariance, aligned with the target heading.

- **Input**: `Float32MultiArray` prediction/estimation state (22 floats)
- **Output**: `PoseArray` candidates in both base-link and optical frames
- **Service**: `std_srvs/Trigger` — generate candidates for the current step

### `score_node`

Scores each candidate viewpoint by projecting fitted ellipsoids (from GMM
clustering of frontier/occupied voxels via CGAL) into a virtual camera view
and measuring the frontier-vs-occupied pixel ratio.

Supports deterministic scoring and Monte Carlo scoring over the prediction
state distribution.

- **Input**: candidates, frontier/occupied clouds, prediction state
- **Output**: scores (`Float32MultiArray`), best candidate (`PoseStamped`)
- **Service**: `std_srvs/Trigger` — score candidates for the current step

## Pipeline

An external experiment coordinator (from `simulation_bringup` or
`testbed_bringup`) calls the three services sequentially:

```
voxelize → generate candidates → score → move pursuer to best
```

All nodes synchronize on a shared `builtin_interfaces/Time` step token.

## Configuration

YAML configs in `config/`:

| File | Mode |
|------|------|
| `planner.yaml` | Dynamic target with prediction + Monte Carlo scoring |
| `planner_no_pred_est.yaml` | Estimation-only baseline (no prediction) |
| `planner_static.yaml` | Static target (TF-based) |

## Dependencies

Eigen3, OpenCV, CGAL, OctoMap, OpenMP, plus standard ROS 2 packages
(rclcpp, tf2, sensor_msgs, etc.).
