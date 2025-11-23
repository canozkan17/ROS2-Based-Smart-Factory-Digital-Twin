#!/usr/bin/env python3

"""
Listenes to Job Orders, and Maintenance Queue topics.
Publishes to Sensors and Completed topics.

Handles tasks for the Hydraulic Press machine.
"""

from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import json
import time
import os

class Machine_Hydraulic_Press_Node(Node):
    def __init__(self):
        super().__init__('Machine_Hydraulic_Press_Node')
        
        # Subscription to Job Orders
        self.subscription_job_orders = self.create_subscription(String, 'Job_Orders', self.listener_job_orders_callback, 10)        
        # Subscription to Maintenance Queue
        self.subscription_maintenance_queue = self.create_subscription(String, 'Maintenance_Queue', self.listener_maintenance_queue_callback, 10)
                
        # Publisher for Sensors data
        self.publisher_sensors = self.create_publisher(String, "Sensors", 10)
        # Publisher for Completed status
        self.publisher_completed = self.create_publisher(String, "Completed", 10)
        
        # Set-up logs
        self.get_logger().info("Machine Hydraulic Press node ready!")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Maintenance_Queue' topic.")
        self.get_logger().info("Listening on 'Completed' topic.")

        # Variable set-up
        self.job_queue = []


    def listener_job_orders_callback(self, msg: String):
        """
        Callback function for Job Orders subscription.
        """
        try: 
            job_input_data = json.loads(msg.data)  # JSON string to dict
            for job in job_input_data:
                if job['machine'] == 'hydraulic_press' and job['depending_on'] is None:
                    self.get_logger().info(
                                            f"\nReceived task: '{job['job_ID']}, to produce {job['produce_amount']}"
                                        )
                    self.job_queue.append(job)
        except json.JSONDecodeError as e:
                self.get_logger().error(f"User Input JSON parse error: {e}")

    # TODO: Implement actual processing logic here
    def listener_maintenance_queue_callback(self, msg: String):
        """
        Callback function for Maintenance Queue subscription.
        """
        self.get_logger().info(f"Received Maintenance Queue message: {msg.data}")

    





def main(args=None):
    """
    Main entry point for the Machine Hydraulic Press node.
    """
    rclpy.init(args=args)
    node = Machine_Hydraulic_Press_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()