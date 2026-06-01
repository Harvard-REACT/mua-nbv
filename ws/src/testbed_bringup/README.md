# testbed_bringup

Real-world/testbed runtime package for the MUA-NBV core pipeline.

## Nodes

- `static_tf_broadcaster`
- `vrpn_tf_bridge`
- `target_stepper`
- `trajectory_predictor`
- `cloud_capturer`
- `pursuer_teleporter`
- `pursuer_mover`
- `trajectory_follower`
- `coordinated_trajectory_follower`
- `experiment_coordinator`
- `closest_candidate_coordinator`
- `rgb_capturer`

## Launch

```bash
ros2 launch testbed_bringup testbed.launch.py \
  testbed_config:=src/testbed_bringup/config/testbed_dynamic.yaml \
  planner_config:=src/mua_nbv_planner/config/planner.yaml
```

`testbed_dynamic.yaml` is configured for `measurement_source: "waypoint"` in the camera-ready core configuration.
