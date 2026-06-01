from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch.conditions import IfCondition, LaunchConfigurationEquals


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare("testbed_bringup"),
        "config",
        "testbed_static.yaml",
    ])
    
    Plannar_params = PathJoinSubstitution([
        FindPackageShare("testbed_bringup"),
        "config",
        "planner_testbed_static.yaml",
    ])

    params_file = DeclareLaunchArgument(
        "params_file",
        default_value=params,
        description="Path to testbed_bringup YAML params file",
    )

    planner_params_file = DeclareLaunchArgument(
        "planner_params_file",
        default_value=Plannar_params,
        description="Path to mua_nbv_planner params YAML (testbed static defaults).",
    )

    launch_planner = DeclareLaunchArgument(
        "launch_planner",
        default_value="false",
        description="Launch mua_nbv_planner voxel_map_node + ring_generator_node + score_node",
    )

    target_mode = DeclareLaunchArgument(
        "target_mode",
        default_value="static",
        description="Target mode: 'static' (no predictor) or 'dynamic' (launch predictor).",
    )

    # Staged startup (helps avoid TF race conditions on hardware)
    tf_start_delay_sec = DeclareLaunchArgument(
        "tf_start_delay_sec",
        default_value="0.0",
        description="Delay before starting TF publishers (sec).",
    )
    bringup_start_delay_sec = DeclareLaunchArgument(
        "bringup_start_delay_sec",
        default_value="1.0",
        description="Delay after TF before starting bringup nodes (sec).",
    )
    planner_start_delay_sec = DeclareLaunchArgument(
        "planner_start_delay_sec",
        default_value="2.0",
        description="Delay after TF before starting planner nodes (sec).",
    )

    # Repo nodes
    vrpn = Node(
        package="testbed_bringup",
        executable="vrpn_tf_bridge",
        name="vrpn_tf_bridge",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    static_tf = Node(
        package="testbed_bringup",
        executable="testbed_static_tf",
        name="testbed_static_tf",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    cloud_capturer = Node(
        package="testbed_bringup",
        executable="cloud_capturer",
        name="cloud_capturer",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    target_stepper = Node(
        package="testbed_bringup",
        executable="target_stepper",
        name="target_stepper",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        condition=LaunchConfigurationEquals("target_mode", "dynamic"),
    )

    pursuer_mover = Node(
        package="testbed_bringup",
        executable="pursuer_mover",
        name="pursuer_mover",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    trajectory_predictor = Node(
        package="testbed_bringup",
        executable="trajectory_predictor",
        name="trajectory_predictor",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        condition=LaunchConfigurationEquals("target_mode", "dynamic"),
    )

    voxel_map = Node(
        package="mua_nbv_planner",
        executable="voxel_map_node",
        name="voxel_map_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_params_file")],
        condition=IfCondition(LaunchConfiguration("launch_planner")),
    )

    ring_generator = Node(
        package="mua_nbv_planner",
        executable="ring_generator_node",
        name="ring_generator_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_params_file")],
        condition=IfCondition(LaunchConfiguration("launch_planner")),
    )

    score_node = Node(
        package="mua_nbv_planner",
        executable="score_node",
        name="score_node",
        output="screen",
        parameters=[LaunchConfiguration("planner_params_file")],
        condition=IfCondition(LaunchConfiguration("launch_planner")),
    )

    tf_group = TimerAction(
        period=LaunchConfiguration("tf_start_delay_sec"),
        actions=[vrpn, static_tf],
    )

    bringup_group = TimerAction(
        period=LaunchConfiguration("bringup_start_delay_sec"),
        actions=[cloud_capturer, target_stepper, trajectory_predictor, pursuer_mover],
    )

    planner_group = TimerAction(
        period=LaunchConfiguration("planner_start_delay_sec"),
        actions=[voxel_map, ring_generator, score_node],
    )

    return LaunchDescription([
        params_file,
        planner_params_file,
        launch_planner,
        target_mode,
        tf_start_delay_sec,
        bringup_start_delay_sec,
        planner_start_delay_sec,

        tf_group,
        bringup_group,
        planner_group,
    ])
