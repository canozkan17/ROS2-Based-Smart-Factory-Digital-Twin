#!/usr/bin/env python3

"""
ROS2 Node for generating synthetic sensor data for Process Pump Machine.


Subscribes to User_Input topic for job order input.
Subscribes to Maintenance_Queue topic for rescheduling. 
Subscribes to Completed topic for next job order pushing.

Gets the job orders from user input or JSON file.

Publishes selected job order to Job_Orders Topic.
Publishes status to Production_Log Topic. 
"""

from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import json
import time
import os

class Job_Scheduler_Node(Node):
    """ROS2 Node for generating job orders to the production system."""

    def __init__(self):
        """
        Initialize the Job Scheduler node, set up subscriptions and publishers.
        Get user input for job scheduling.
        """
        super().__init__('Job_Scheduler_Node')
        
        # Subscription to User Input
        self.subscription_user_input = self.create_subscription(String, 'User_Input', self.listener_user_input_callback, 10)
        
        # Subscription to Maintenance Queue
        self.subscription_maintenance_queue = self.create_subscription(String, 'Maintenance_Queue', self.listener_maintenance_queue_callback, 10)
    
        # Subscription to Completed
        self.subscription_completed = self.create_subscription(String, 'Completed', self.listener_completed_callback, 10)
        
        # Publisher for Job Orders data
        self.publisher_job_orders = self.create_publisher(String, "Job_Orders", 10)

        # Publisher for Production Log status
        self.publisher_production_log = self.create_publisher(String, "Production_Log", 10)
        
        # Set-up logs
        self.get_logger().info("Job Scheduler node ready!")
        self.get_logger().info("Listening on 'User_Input' topic.")
        self.get_logger().info("Listening on 'Maintenance_Queue' topic.")
        self.get_logger().info("Listening on 'Completed' topic.")

        # Job setup
        self.job_finished = False
        self.job_number = 000
        self.target_job = None

    
    
    # Callback Functions Block                                                                        !! REMAINING TO DO !!
    #--------------------
    
    # Callback for User Input                                                            !! REMAINING TO DO !!
    def listener_user_input_callback(self, msg: String):
        """
        Callback function for User Input subscription.
        Processes user input for job scheduling.
        """
        try:
            user_input_data = json.loads(msg.data)  # JSON string to dict
            self.get_logger().info(f"Received User Input item: {user_input_data}")
        # TO DO : Process user_input_data as necessary
        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
        
        
    # Callback for maintenance queue                                                            !! REMAINING TO DO !!
    def listener_maintenance_queue_callback(self, msg: String):
        """
        Callback function for Maintenance Queue subscription.
        Re-organizes job_orders according to maintenance requirements.
        """
        self.get_logger().info(f"Received Maintenance Queue item: {msg.data}")
        
        # Simulate job order re-organization                                                    !! REMAINING TO DO !!
        job_order = String()
        job_order.data = f"Job Order: {msg.data}"
        
        self.publisher_job_orders.publish(job_order)
        self.get_logger().info(f"Published Job Order: {job_order.data}")
    # Callback for Completed                                                                    !! REMAINING TO DO !!
    def listener_completed_callback(self, msg: String):
        """
        Callback function for Completed  subscription.
        Processes control commands for corrective/preventative actions.
        """
        self.get_logger().info(f"Received Completed item: {msg.data}")
    
    #--------------------


    def publish_job_orders(self):
        """
        Publish job order to the Job_Orders topic.
        """
        job_order_msg = String()

        # For Hydraulic Press Machine
        job_order_msg.data = json.dumps({
                                            "sender": self.get_name(),
                                            "data": True
                                        })
        
        self.publisher_job_orders.publish(job_order_msg)

        



def main(args=None):
    """
    Main entry point for the Job_Scheduler_Node.
    """
    rclpy.init(args=args)
    node = Job_Scheduler_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()