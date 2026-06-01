import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, LogInfo
from launch.conditions import LaunchConfigurationEquals, IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    sim_pkg_share = get_package_share_directory("simulation_bringup")
    planner_pkg_share = get_package_share_directory("mua_nbv_planner")

    # Paths
    world_path_default = os.path.join(sim_pkg_share, "worlds", "sim.world")
    bridge_yaml = os.path.join(sim_pkg_share, "config", "bridge.yaml")
    simulation_yaml_default = os.path.join(sim_pkg_share, "config", "simulation.yaml")
    simulation_static_yaml_default = os.path.join(sim_pkg_share, "config", "simulation_static.yaml")
    planner_yaml_default = os.path.join(planner_pkg_share, "config", "planner.yaml")
    planner_static_yaml_default = os.path.join(planner_pkg_share, "config", "planner_static.yaml")

    pursuer_sdf = os.path.join(sim_pkg_share, "models", "pursuer", "model.sdf")
    target_sdf = os.path.join(sim_pkg_share, "models", "target", "bunny.sdf")

    # Arguments
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=world_path_default,
        description="Full path to an SDF world file",
    )
    
    gz_world_name_arg = DeclareLaunchArgument(
        "gz_world_name",
        default_value="sim_world",
        description="Gazebo world name",
    )

    gz_args_arg = DeclareLaunchArgument(
        "gz_args",
        default_value="-r -v 1",
        description="Arguments passed to `gz sim` via ros_gz_sim (e.g. '-r -v 1 -s' for headless/server-only).",
    )

    sim_config_arg = DeclareLaunchArgument(
        "sim_config",
        default_value=simulation_yaml_default,
        description="Path to simulation.yaml (dynamic mode defaults).",
    )

    sim_config_static_arg = DeclareLaunchArgument(
        "sim_config_static",
        default_value=simulation_static_yaml_default,
        description="Path to simulation_static.yaml (static mode defaults).",
    )

    planner_config_arg = DeclareLaunchArgument(
        "planner_config",
        default_value=planner_yaml_default,
        description="Path to planner.yaml (dynamic mode defaults).",
    )

    planner_config_static_arg = DeclareLaunchArgument(
        "planner_config_static",
        default_value=planner_static_yaml_default,
        description="Path to planner_static.yaml (static mode defaults).",
    )

    target_mode_arg = DeclareLaunchArgument(
        "target_mode",
        default_value="dynamic",
        description="Target mode: 'dynamic' (use stepper+predictor) or 'static' (baseline; no stepper+predictor).",
    )
    pipeline_mode_arg = DeclareLaunchArgument(
        "pipeline_mode",
        default_value="full",
        description="Pipeline mode for dynamic runs: 'full' (NBV planning) or 'predict_only' (stepper+predictor only).",
    )
    log_target_mode = LogInfo(msg=["[simulation.launch.py] target_mode=", LaunchConfiguration("target_mode")])
    log_pipeline_mode = LogInfo(msg=["[simulation.launch.py] pipeline_mode=", LaunchConfiguration("pipeline_mode")])
    log_configs = LogInfo(
        msg=[
            "[simulation.launch.py] sim_config=", LaunchConfiguration("sim_config"),
            " sim_config_static=", LaunchConfiguration("sim_config_static"),
            " planner_config=", LaunchConfiguration("planner_config"),
            " planner_config_static=", LaunchConfiguration("planner_config_static"),
        ]
    )
    log_gz = LogInfo(msg=["[simulation.launch.py] gz_args=", LaunchConfiguration("gz_args"), " world=", LaunchConfiguration("world")])

    dynamic_full = IfCondition(
        PythonExpression(
            [
                '"',
                LaunchConfiguration("target_mode"),
                '" == "dynamic" and "',
                LaunchConfiguration("pipeline_mode"),
                '" == "full"',
            ]
        )
    )

    # 1. Gazebo Sim
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": [LaunchConfiguration("gz_args"), " ", LaunchConfiguration("world")]}.items(),
    )

    # 2. Bridges
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        parameters=[{"config_file": bridge_yaml, "use_sim_time": True}],
    )

    set_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_set_pose_bridge",
        output="screen",
        arguments=[["/world/", LaunchConfiguration("gz_world_name"),
                    "/set_pose@ros_gz_interfaces/srv/SetEntityPose"
        ]],
        parameters=[{"use_sim_time": True}],
    )

    # 3. TF Structure
    # Static: world -> sim_world
    world_to_sim_world = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_sim_world_static_tf",
        arguments=["0","0","0","0","0","0","world","sim_world"],
        parameters=[{"use_sim_time": True}],
    )


    # Static: Base -> Camera (Required for Ring Generator)
    sim_static_tf_dynamic = Node(
        package="simulation_bringup",
        executable="sim_static_tf",
        name="sim_static_tf",
        output="screen",
        parameters=[LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "dynamic"),
    )
    sim_static_tf_static = Node(
        package="simulation_bringup",
        executable="sim_static_tf",
        name="sim_static_tf",
        output="screen",
        parameters=[LaunchConfiguration("sim_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    # Dynamic: sim_world -> pursuer/base_link (Ground Truth from GZ)
    sim_pose_tf_bridge_dynamic = Node(
        package="simulation_bringup",
        executable="sim_pose_tf_bridge",
        name="sim_pose_tf_bridge",
        output="screen",
        parameters=[LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "dynamic"),
    )
    sim_pose_tf_bridge_static = Node(
        package="simulation_bringup",
        executable="sim_pose_tf_bridge",
        name="sim_pose_tf_bridge",
        output="screen",
        parameters=[LaunchConfiguration("sim_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    # 4. Simulation Logic Nodes
    spawn_launch = os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_spawn_model.launch.py")

    spawn_pursuer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_launch),
        launch_arguments={"world": LaunchConfiguration("gz_world_name"),
                          "file": pursuer_sdf, "entity_name": "pursuer", "x": "-0.50", "y": "0.0", "z": "0.01",
        }.items(),
    )

    spawn_target = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_launch),
        launch_arguments={"world": LaunchConfiguration("gz_world_name"),
                          "file": target_sdf, "entity_name": "target",
                          "x": "0.0", "y": "0.0", "z": "0.0",
        }.items(),
    )

    target_stepper = Node(
        package="simulation_bringup",
        executable="target_stepper",
        name="target_stepper",
        output="screen",
        parameters=[LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "dynamic"),
    )

    cloud_capturer_dynamic = Node(
        package="simulation_bringup",
        executable="cloud_capturer",
        name="cloud_capturer",
        output="screen",
        parameters=[LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=dynamic_full,
    )
    cloud_capturer_static = Node(
        package="simulation_bringup",
        executable="cloud_capturer",
        name="cloud_capturer",
        output="screen",
        parameters=[LaunchConfiguration("sim_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    trajectory_predictor = Node(
        package="simulation_bringup",
        executable="trajectory_predictor",
        name="trajectory_predictor",
        output="screen",
        parameters=[LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "dynamic"),
    )

    # 5. Planner Nodes (MUA_NBV_PLANNER)
    # Map Node: Accumulates cloud -> Voxel Grid
    voxel_map_node_dynamic = Node(
        package="mua_nbv_planner",
        executable="voxel_map_node",
        name="voxel_map_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_config"), {"use_sim_time": True}],
        condition=dynamic_full,
    )
    voxel_map_node_static = Node(
        package="mua_nbv_planner",
        executable="voxel_map_node",
        name="voxel_map_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    # Ring Generator: Creates candidate poses around predicted target
    ring_generator_node_dynamic = Node(
        package="mua_nbv_planner",
        executable="ring_generator_node",
        name="ring_generator_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_config"), {"use_sim_time": True}],
        condition=dynamic_full,
    )
    ring_generator_node_static = Node(
        package="mua_nbv_planner",
        executable="ring_generator_node",
        name="ring_generator_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    # Score Node: Evaluates candidates
    # Load planner_config first, then sim_config so per-run overrides (e.g. mc_seed) take effect.
    score_node_dynamic = Node(
        package="mua_nbv_planner",
        executable="score_node",
        name="score_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_config"), LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=dynamic_full,
    )
    score_node_static = Node(
        package="mua_nbv_planner",
        executable="score_node",
        name="score_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_config_static"), LaunchConfiguration("sim_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    pursuer_spawner_dynamic = Node(
        package="simulation_bringup",
        executable="pursuer_spawner",
        name="pursuer_spawner",
        output="screen",
        parameters=[LaunchConfiguration("sim_config"), {"use_sim_time": True}],
        condition=dynamic_full,
    )
    pursuer_spawner_static = Node(
        package="simulation_bringup",
        executable="pursuer_spawner",
        name="pursuer_spawner",
        output="screen",
        parameters=[LaunchConfiguration("sim_config_static"), {"use_sim_time": True}],
        condition=LaunchConfigurationEquals("target_mode", "static"),
    )

    # 6. Timed Execution
    spawn_both = TimerAction(period=3.0, actions=[spawn_pursuer, spawn_target])
    
    # Delay logic nodes to ensure TF tree is roughly established
    logic_nodes = TimerAction(period=5.0, actions=[
        target_stepper,
        trajectory_predictor,
        cloud_capturer_dynamic,
        cloud_capturer_static,
        voxel_map_node_dynamic,
        voxel_map_node_static,
        ring_generator_node_dynamic,
        ring_generator_node_static,
        score_node_dynamic,
        score_node_static,
        pursuer_spawner_dynamic,
        pursuer_spawner_static,
    ])

    return LaunchDescription([
        # Arguments
        world_arg,
        gz_world_name_arg,
        gz_args_arg,
        sim_config_arg,
        sim_config_static_arg,
        planner_config_arg,
        planner_config_static_arg,
        target_mode_arg,
        pipeline_mode_arg,
        log_target_mode,
        log_pipeline_mode,
        log_configs,
        log_gz,
        # Gazebo Sim
        gz_sim_launch,
        # Bridges
        bridge,
        set_pose_bridge,
        # TF Structure
        world_to_sim_world,
        sim_static_tf_dynamic,
        sim_static_tf_static,
        sim_pose_tf_bridge_dynamic,
        sim_pose_tf_bridge_static,
        # Simulation Logic Nodes
        spawn_both,
        logic_nodes
    ])
