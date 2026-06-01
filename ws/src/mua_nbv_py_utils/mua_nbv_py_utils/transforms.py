"""
Quaternion, rotation matrix, and planarization helpers.

All quaternions use (x, y, z, w) convention.
All rotation matrices are 3x3 numpy arrays (float64).
"""

import math
from typing import Tuple

import numpy as np

Quat = Tuple[float, float, float, float]
Vec3 = Tuple[float, float, float]


def quat_from_yaw(yaw: float) -> Quat:
    """Pure yaw (z-axis) rotation to quaternion (x, y, z, w)."""
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quat(q: Quat) -> float:
    """Extract planar yaw from quaternion (x, y, z, w)."""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    """Euler angles (ZYX intrinsic) to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def quat_to_rot_np(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) to 3x3 rotation matrix (float64)."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def yaw_from_rot(R: np.ndarray) -> float:
    """Planar yaw from a 3x3 rotation matrix."""
    return float(math.atan2(float(R[1, 0]), float(R[0, 0])))


def quat_mul(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_conj(q: Quat) -> Quat:
    """Conjugate (inverse for unit quaternions)."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate a 3-vector by a unit quaternion: q * v * q_conj."""
    vx, vy, vz = v
    qv: Quat = (vx, vy, vz, 0.0)
    r = quat_mul(quat_mul(q, qv), quat_conj(q))
    return (r[0], r[1], r[2])


def rot_apply(R: np.ndarray, v: Vec3) -> Vec3:
    """Apply a 3x3 rotation matrix to a 3-vector (returns tuple)."""
    out = R @ np.array(v, dtype=np.float64)
    return (float(out[0]), float(out[1]), float(out[2]))


def rotate_z(yaw: float, v: Vec3) -> Vec3:
    """Rotate a 3-vector about +Z by *yaw* radians."""
    x, y, z = v
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y, z)


def angle_wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def planarize(
    t: Vec3,
    q: Quat,
    *,
    flatten_z: bool = True,
    z0: float = 0.0,
    flatten_roll_pitch: bool = True,
    yaw_offset_rad: float = 0.0,
) -> Tuple[Vec3, Quat]:
    """
    Flatten a 3D pose to planar: optionally zero z and collapse
    orientation to pure yaw (with optional yaw offset).
    """
    tx, ty, tz = t
    if flatten_z:
        tz = float(z0)
    if flatten_roll_pitch:
        yaw = yaw_from_quat(q) + float(yaw_offset_rad)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        q = quat_from_yaw(yaw)
    return (tx, ty, tz), q


def yaw_var_from_vel(vx: float, vy: float, Cov_v: np.ndarray) -> float:
    """
    Approximate yaw variance from velocity covariance via first-order
    propagation of yaw = atan2(vy, vx).
    """
    vx, vy = float(vx), float(vy)
    s2 = vx * vx + vy * vy
    if s2 < 1e-8:
        return 1.0
    J = np.array([-vy / s2, vx / s2], dtype=float).reshape(1, 2)
    Cov_v = np.asarray(Cov_v, dtype=float).reshape(2, 2)
    var = float((J @ Cov_v @ J.T).item())
    return float(max(1e-6, var))
