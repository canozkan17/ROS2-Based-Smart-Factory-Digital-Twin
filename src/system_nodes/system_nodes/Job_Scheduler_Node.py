#!/usr/bin/env python3

"""
ROS2 Node for generating synthetic sensor data for Process Pump Machine.


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
        self.get_logger().info("Listening on 'Maintenance_Queue' topic.")
        self.get_logger().info("Listening on 'Completed' topic.")

        # Job setup
        self.job_finished = False
        self.job_number = 000
        self.target_job = None

        self.job_names = []
        self.data = []
        self.example_json = '''{{
                                "job_name": "New Job Name",
                                "job_ID": "3-4 letter code",
                                "machine_ID": "hydraulic_press / process_pump",
                                "part_type": "metal_sheet / other ",
                                "material": "type of material (e.g., st37, aluminium)",
                                "part_thickness_mm": number,
                                "part_weight_kg":  number,
                                "pressure_level": "low / medium / high",
                                "process_type": "forming / bending / deep_bending / bending_and_piercing",
                                "press_force_ton": number,
                                "produce_amount": null,
                                "cycles_per_part": number,
                                "total_cycle_count": null,
                                "cycle_duration_sec": number,
                                "temperature_C": number,
                                "priority": "high / medium / low",
                                "surface_quality_mm": number,
                                "tolerance_mm": number
                        }}'''
        
        self.get_job_order()

    def total_cycle_calculator(self, produce_amount, cycles_per_part):
            return produce_amount * cycles_per_part

    def job_selecting(self):
            global job_name_input, job_ID, produce_amount
            
            correct_selection = False

            while not correct_selection:    
                job_name_input = self.job_names[int(input("\nEnter the job name number from the list: "))-1]
                self.target_job = next((job for job in self.data if job["job_name"] == job_name_input), None)

                if self.target_job:
                    job_ID = self.target_job["job_ID"]
                
                    # Get user input for production amount
                    produce_amount = int(input("Enter the production amount:             "))
                    
                    self.target_job["produce_amount"] = produce_amount
                    self.target_job["total_cycle_count"] = self.total_cycle_calculator(produce_amount, self.target_job["cycles_per_part"])
                    print(f"{self.target_job} is ")
                    correct_selection = True

                else:
                    print(f"{job_name_input} not found in job orders.")

    def job_entering(self):
            global job_name_input, job_ID, produce_amount

            while True: 
                print("\nEnter the job order as JSON like below: ")
                
                print(self.example_json)
                
                job_order_input = input("\nPaste the job order JSON here: ")
                
                try:
                    job_order_data = json.loads(job_order_input)
                    job_name_input = job_order_data["job_name"]
                    job_ID = job_order_data["job_ID"]
                    produce_amount = int(input("Enter the production amount:             "))
                    
                    job_order_data["produce_amount"] = produce_amount
                    job_order_data["total_cycle_count"] = self.total_cycle_calculator(produce_amount, job_order_data["cycles_per_part"])

                    # read the existing job orders from JSON file
                    file_path = os.path.join(os.path.dirname(__file__), "job_orders.json")
                    if os.path.exists(file_path):
                        self.data = self.read_json_file(file_path)
                        if self.job_entry_check(job_name_input):
                            print(f"Job name '{job_name_input}' already exists as a job order. Please enter a unique job name.")
                            continue
                    else:
                        print("Job orders JSON not found. A new file will be created.")
                        self.data = []

                    # add the new job order to the list
                    self.data.append(job_order_data)

                    # write to the JSON file
                    self.write_json_file(file_path, self.data)            
                    break
                    
                except Exception as e:
                    print("Invalid JSON input:", e)
                    e = input("press 'e' to exit, press any other key to try again: ")
                    if e.lower() == 'e':
                        exit(1)
            
    def job_entry_check(self, job_name_input):
            for job in self.data:
                if job['job_name'] == job_name_input:
                    return True
            return False
        
    def get_job_order(self):
            try:
                # read the existing job orders from JSON file
                file_path = os.path.join(os.path.dirname(__file__), "job_orders.json")
                self.data = self.read_json_file(file_path)

                # extract job names from the data
                for job in self.data:
                    self.job_names.append(job['job_name'])

                # Get user input for existing job or new job
                print("\n == Welcome to the Job Scheduler! ==\n")
                print(f"Available Operations:")
                for i, job in enumerate(self.job_names, 1):
                    print(f"{i}. {job}")
                user_input = input("\nDoes the job already exist in above list? (yes/no): ").strip().lower()

                if user_input[0] == 'y':
                    self.job_selecting()
                else:
                    self.job_entering()

            except FileNotFoundError:
                print("Job orders JSON not found.")
                self.job_entering()
                exit(1)
            except json.JSONDecodeError as e:
                print("JSON format error:", e)
                exit(1)

    def read_json_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data    
    
    def write_json_file(self, file_path, data):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    
    
    # Callback Functions Block                                                                        !! REMAINING TO DO !!
    #--------------------
    
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