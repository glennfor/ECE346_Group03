from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ece_346_share = FindPackageShare('racecar_ece346')
    racecar_interface_share = FindPackageShare('racecar_interface')

    default_params_file = PathJoinSubstitution([
        ece_346_share,
        'config',
        'lab5_eval.yaml',
    ])

    param_file = LaunchConfiguration('param_file')

    declare_params = DeclareLaunchArgument(
        'param_file',
        default_value=default_params_file,
        description='YAML file for BC eval parameters',
    )

    # Visualization (RViz + routing + viz node)
    visualization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([racecar_interface_share, 'launch', 'visualization_launch.py'])
        ),
        launch_arguments={'param_file': param_file}.items()
    )

    bc_eval_node = Node(
        package='racecar_ece346',
        executable='bc_eval_node.py',
        name='bc_eval_node',
        output='screen',
        parameters=[param_file],
    )

    return LaunchDescription([
        declare_params,
        visualization_launch,
        bc_eval_node,
    ])
