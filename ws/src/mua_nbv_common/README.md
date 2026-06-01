# mua_nbv_common

Shared ROS 2 Python utilities for the MUA-NBV pipeline. Used by both
`simulation_bringup` and `testbed_bringup` nodes.

## Module: `ros_helpers`

### Stamp helpers

```python
from mua_nbv_common.ros_helpers import time_tuple, stamp_str, stamp_tuple_str, make_stamp

t = make_stamp(sec=10, nanosec=500_000_000)
time_tuple(t)        # (10, 500000000)
stamp_str(t)         # "10.500000000"
stamp_tuple_str((10, 500_000_000))  # "10.500000000"
```

### Trigger service wrapper

```python
from mua_nbv_common.ros_helpers import call_trigger

res = call_trigger(node, client, "predict", timeout_sec=5.0)
```

### Prediction-state message packing

Both the simulation and testbed trajectory predictors publish a
`Float32MultiArray` with 22 floats:
`[sec, nsec, x, y, vx, vy, Sigma4_row_major(16)]`.

```python
from mua_nbv_common.ros_helpers import pack_state_msg, unpack_state_msg

msg = pack_state_msg(stamp, mu4, Sigma4)
(sec, nsec), mu4, Sigma4 = unpack_state_msg(msg)
```

## Dependencies

- `rclpy`, `std_srvs`, `std_msgs`, `builtin_interfaces` (ROS 2)
- `numpy`
