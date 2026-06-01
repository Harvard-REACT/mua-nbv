# simulation_bringup

Simulation runtime package for the MUA-NBV core pipeline.

## Nodes

- `sim_pose_tf_bridge`
- `sim_static_tf`
- `target_stepper`
- `trajectory_predictor`
- `cloud_capturer`
- `pursuer_spawner`
- `experiment_coordinator`

## Launch

```bash
ros2 launch simulation_bringup simulation.launch.py \
  target_mode:=dynamic \
  sim_config:=src/simulation_bringup/config/simulation.yaml \
  planner_config:=src/mua_nbv_planner/config/planner.yaml
```

Static baseline:

```bash
ros2 launch simulation_bringup simulation.launch.py \
  target_mode:=static \
  sim_config_static:=src/simulation_bringup/config/simulation_static.yaml \
  planner_config_static:=src/mua_nbv_planner/config/planner_static.yaml
```
