from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        Node(
                package='system_nodes',
                executable='hydraulic_press_sensor',
                name='hydraulic_press_sensor_node'
            ),

        Node(
                package='system_nodes',
                executable='process_pump_sensor',
                name='process_pump_sensor_node'
            ),

        Node(
                package='system_nodes',
                executable='job_scheduler',
                name='job_scheduler_node'
            )
    ])
