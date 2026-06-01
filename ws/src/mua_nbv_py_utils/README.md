# mua_nbv_py_utils

Pure-Python utility library for the MUA-NBV pipeline. **No ROS 2 dependencies.**

## Modules

### `transforms`
Quaternion and rotation helpers used throughout the pipeline:
- `quat_to_rot_np(qx, qy, qz, qw)` — quaternion to 3x3 rotation matrix
- `yaw_from_quat(qx, qy, qz, qw)` — extract planar yaw angle
- `quat_from_yaw(yaw)` — yaw angle to z-axis quaternion `(x, y, z, w)`
- `quat_from_rpy(roll, pitch, yaw)` — Euler angles to quaternion
- `quat_mul(q1, q2)` — Hamilton product
- `quat_conj(q)` — conjugate
- `quat_rotate(q, v)` — rotate a 3-vector by a quaternion
- `planarize(position, quat, *, z=0.0, yaw_offset=0.0)` — zero out z, flatten to pure yaw
- `rotate_z(yaw, v)` — rotate a 3-vector about +Z by yaw radians
- `yaw_var_from_vel(vx, vy, Sigma4, *, min_speed)` — heading variance from velocity covariance
- `angle_wrap_pi(angle)` — wrap angle to [-pi, pi]

### `pcd_io`
Point-cloud file I/O and voxelization:
- `read_pcd_xyz(path)` — read XYZ points from ASCII or binary PCD files
- `write_pcd_xyz_ascii(path, points)` — write XYZ ASCII PCD
- `voxel_ids(points, resolution)` — quantize points to integer voxel IDs
- `stamp_from_path(path)` — extract timestamp from `cloud_<stamp>.pcd` filenames
- `sorted_clouds(directory)` — list PCD files sorted by timestamp

## Usage (standalone, no ROS)

```python
from mua_nbv_py_utils.transforms import quat_to_rot_np
from mua_nbv_py_utils.pcd_io import read_pcd_xyz

R = quat_to_rot_np(0, 0, 0.707, 0.707)
pts = read_pcd_xyz("cloud_123.456.pcd")
```
