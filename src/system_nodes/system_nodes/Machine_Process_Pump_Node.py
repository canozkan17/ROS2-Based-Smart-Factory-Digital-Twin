#!/usr/bin/env python3

"""
ROS2 Node for generating synthetic sensor data for Process Pump Machine.

Subscribes to Job_Orders topic for data generation. 
Subscribes to Control_CMD topic for corrective/preventative actions.

Publishes generated data to Sensors topic.
Publishes status to Completed Topic. 
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Machine_Process_Pump_Sensor_Node(Node):
    """ROS2 Node for generating synthetic sensor data for Process Pump Machine."""

    def __init__(self):
        """
        Initialize the Process Pump Sensor node, set up subscriptions and publishers.
        """
        super().__init__('Machine_Process_Pump_Sensor_Node')
        
        # Subscription to Job Orders
        self.subscription_job_order = self.create_subscription(String, 'Job_Orders', self.listener_job_orders_callback, 10)
    
        # Subscription to Control CMD
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD', self.listener_callback_control_cmd, 10)
        

        # Publisher for Sensors data
        self.publisher_sensors = self.create_publisher(String, "Sensors", 10)

        # Publisher for Completed status
        self.publisher_completed = self.create_publisher(String, "Completed", 10)
        
        self.get_logger().info("Process Pump Sensor node ready!")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Control_CMD' topic.")

    
    # Callback functions
    #--------------------
    # Callback for Job Orders
    def listener_job_orders_callback(self, msg: String):
        """
        Callback function for Job Orders subscription.
        Generates synthetic sensor data based on job orders.
        """
        self.get_logger().info(f"Received Job Order: {msg.data}")
        
        # Simulate sensor data generation
        sensor_data = String()
        sensor_data.data = f"Synthetic Sensor Data for Job Order: {msg.data}"
        
        self.publisher_sensors.publish(sensor_data)
        self.get_logger().info(f"Published Sensor Data: {sensor_data.data}")

    #--------------------
    # Callback for Control CMD
    def listener_callback_control_cmd(self, msg: String):
        """
        Callback function for Control CMD subscription.
        Processes control commands for corrective/preventative actions.
        """
        self.get_logger().info(f"Received Control Command: {msg.data}")
        



def main(args=None):
    """
    Main entry point for the machine_hydraulic_press_sensor_node.
    """
    rclpy.init(args=args)
    node = Machine_Process_Pump_Sensor_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()