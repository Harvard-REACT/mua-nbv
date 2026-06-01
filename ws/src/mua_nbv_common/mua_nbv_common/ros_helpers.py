"""
Shared ROS 2 helper utilities for MUA-NBV Python nodes.

Provides stamp formatting, Trigger service call wrappers,
and prediction-state message packing used by both the simulation
and testbed bringup packages.
"""

from __future__ import annotations

from array import array
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger


# ── Stamp helpers ────────────────────────────────────────────────────

def time_tuple(t: TimeMsg) -> Tuple[int, int]:
    """Convert a ``builtin_interfaces/Time`` to an ``(sec, nanosec)`` tuple."""
    return (int(t.sec), int(t.nanosec))


def stamp_str(t: Optional[TimeMsg]) -> str:
    """Format a ``TimeMsg`` as ``"sec.nanosec"`` (9 digits, zero-padded)."""
    if t is None:
        return "None"
    return f"{int(t.sec)}.{int(t.nanosec):09d}"


def stamp_tuple_str(tup: Optional[Tuple[int, int]]) -> str:
    """Format an ``(sec, nanosec)`` tuple as ``"sec.nanosec"``."""
    if tup is None:
        return "None"
    return f"{int(tup[0])}.{int(tup[1]):09d}"


def make_stamp(sec: int, nanosec: int) -> TimeMsg:
    """Build a ``builtin_interfaces/Time`` from integers."""
    t = TimeMsg()
    t.sec = int(sec)
    t.nanosec = int(nanosec)
    return t


# ── Trigger service helper ───────────────────────────────────────────

def call_trigger(
    node: Node,
    client,
    name: str,
    timeout_sec: float,
    *,
    require_success: bool = True,
) -> Trigger.Response:
    """Call a ``std_srvs/Trigger`` service and optionally assert success.

    Parameters
    ----------
    node : rclpy.node.Node
        The calling node (needed by ``spin_until_future_complete``).
    client :
        A ``Trigger`` service client created with ``node.create_client``.
    name : str
        Human-readable label for error messages.
    timeout_sec : float
        Maximum seconds to wait for the service response.
    require_success : bool
        If *True* (default) and the response has ``success=False``,
        a ``RuntimeError`` is raised.

    Returns
    -------
    Trigger.Response
    """
    req = Trigger.Request()
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout_sec)
    if not fut.done():
        raise RuntimeError(f"{name} timed out after {timeout_sec:.1f}s")
    res = fut.result()
    if res is None:
        raise RuntimeError(f"{name} returned None")
    if require_success and not res.success:
        raise RuntimeError(f"{name} failed: {res.message}")
    return res


# ── Prediction-state Float32MultiArray packing ───────────────────────

_STATE_MSG_LEN = 22  # 2 stamp + 4 state + 16 covariance


def pack_state_msg(
    stamp: TimeMsg,
    mu4: np.ndarray,
    Sigma4: np.ndarray,
) -> Float32MultiArray:
    """Pack ``[sec, nsec, x, y, vx, vy, Sigma4_row_major]`` into a
    ``Float32MultiArray`` (22 floats), matching the format consumed by
    the C++ planner nodes.

    Parameters
    ----------
    stamp : TimeMsg
        Pipeline token stamp.
    mu4 : (4,) array
        State mean ``[x, y, vx, vy]``.
    Sigma4 : (4, 4) array
        State covariance.
    """
    mu4 = np.asarray(mu4, dtype=np.float64).ravel()
    Sigma4 = np.asarray(Sigma4, dtype=np.float64).reshape(4, 4)
    msg = Float32MultiArray()
    msg.data = array(
        "f",
        [
            float(stamp.sec),
            float(stamp.nanosec),
            float(mu4[0]),
            float(mu4[1]),
            float(mu4[2]),
            float(mu4[3]),
        ],
    )
    msg.data.extend(
        array("f", [float(Sigma4[r, c]) for r in range(4) for c in range(4)])
    )
    return msg


def unpack_state_msg(
    msg: Float32MultiArray,
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_state_msg`.

    Returns
    -------
    stamp_tuple : (sec, nanosec)
    mu4 : (4,) ndarray
    Sigma4 : (4, 4) ndarray
    """
    d = list(msg.data)
    if len(d) < _STATE_MSG_LEN:
        raise ValueError(
            f"Expected {_STATE_MSG_LEN} floats, got {len(d)}"
        )
    stamp_tuple = (int(d[0]), int(d[1]))
    mu4 = np.array(d[2:6], dtype=np.float64)
    Sigma4 = np.array(d[6:22], dtype=np.float64).reshape(4, 4)
    return stamp_tuple, mu4, Sigma4
