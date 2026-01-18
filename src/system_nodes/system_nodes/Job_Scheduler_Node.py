#!/usr/bin/env python3

"""
ROS2 Node for generating job orders to the production system.


Subscribes to User_Input topic for job order input.
Subscribes to Maintenance_Queue topic for rescheduling. 
Subscribes to Completed topic for next job order pushing.

Gets the job orders from user input or JSON file.

Publishes selected job order to Job_Orders Topic.
Publishes status to Production_Log Topic. 
"""

from std_msgs.msg import String
from rclpy.node import Node
from typing import Dict
from typing import Any
import rclpy
import json
import math


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
        # Subscription to Maintenance Queue from Controller_Node
        self.subscription_maintenance_queue = self.create_subscription(String, 'Maintenance_Queue', self.listener_maintenance_queue_callback, 10)
        # Subscription to Completed
        self.subscription_completed = self.create_subscription(String, 'Completed/hydraulic_press', self.listener_completed_callback, 10)

        # Subscription to Maintenance Feedback
        self.subscription_maintenance_feedback = {
                                                    self.create_subscription(String, 'Maintenance_Feedback/process_pump', self.listener_maintenance_feedback_callback, 10),
                                                    self.create_subscription(String, 'Maintenance_Feedback/hydraulic_press', self.listener_maintenance_feedback_callback, 10)
                                                }




        # Publisher for Job Orders data
        self.publisher_job_orders = self.create_publisher(String, "Job_Orders", 10)
        # Publisher for Production Log status
        self.publisher_production_log = self.create_publisher(String, "Production_Log", 10)
        # Publisher for Node status (for GUI)
        self.publisher_node_status = self.create_publisher(String, "Node_Status", 10)
        self._node_status = 'READY'
        # Timer to periodically broadcast node status so GUI can pick it up even if it starts late
        try:
            self.node_status_timer = self.create_timer(2.0, self._publish_node_status)
        except Exception:
            self.node_status_timer = None

        # Job setup
        self.job_order = []
        self.target_job = {}
        self.job_id_list = []
        self.running_operations = {} # key: machine_id, value: task
        self.pending_operations = []
        self.completed_operations = set()
        self.completed_jobs = set()
        self.maintenance_queue_details = {}
    
        # Control flags
        self.process_finished = False
        self.user_input_received = False
        self.maintenance_required = False
        self.maintenance_completed = False
        
        # Set-up logs
        self.get_logger().info("Job Scheduler node ready!")
        self.get_logger().info("Listening on 'User_Input' topic.")
        self.get_logger().info("Listening on 'Maintenance_Queue' topic.")
        self.get_logger().info("Listening on 'Completed' topic.")
        # initial announcement
        try:
            msg = String()
            msg.data = json.dumps({"node": self.get_name(), "status": self._node_status})
            self.publisher_node_status.publish(msg)
        except Exception:
            pass

    def _publish_node_status(self):
        try:
            msg = String()
            msg.data = json.dumps({"node": self.get_name(), "status": self._node_status})
            self.publisher_node_status.publish(msg)
        except Exception:
            pass

    def generate_job_ID(self, user_input_data) -> Dict[str, Any]:
        """
        Generate a unique job ID.
        """
        # Initialize job_ID with 000 suffix
        if user_input_data['job_ID'] not in self.job_id_list:
            self.job_id_list.append(user_input_data['job_ID']) 
            user_input_data['job_ID'] = (f"{user_input_data['job_ID']}000")  
        else: # If job_ID already exists, increment the suffix
            existing_jobs = [job for job in self.job_order if job['job_ID'].startswith(user_input_data['job_ID'])]
            suffix_numbers = [int(job['job_ID'][-3:]) for job in existing_jobs]
            new_suffix = max(suffix_numbers) + 1
            user_input_data['job_ID'] = (f"{user_input_data['job_ID']}{str(new_suffix).zfill(3)}")

        return user_input_data
    
    def select_machine_for_process(self, process_name):
        """"
        Select machine based on the process name.
        """
        
        machine_process_map = {
                                "bending": "hydraulic_press",
                                "forming": "hydraulic_press",
                                "drilling": "cnc",
                                "grooving": "cnc",
                                "pocketing": "cnc",
                                "assembling": "robot_arm",
                                "quality_control": "robot_arm"
                            }
        
        if process_name in machine_process_map:
            return machine_process_map[process_name]
        else:
            return "unknown_machine"
        
    def task_time_calculator(self, produce_amount, process_name, material) -> int:
        """
        Calculate total time required for a process based on produce amount and material.
        """
        
        process_time_map_in_seconds = {
                            "bending": {
                                            "DC01_ZE": 30,
                                            "Stainless_304": 45,
                                            "Aluminium_6082_T6": 25
                                        },
                            "forming": {
                                            "DC01_ZE": 40,
                                            "Stainless_304": 60,
                                            "Aluminium_6082_T6": 35
                                        },
                            "drilling": {
                                            "DC01_ZE": 50,
                                            "Stainless_304": 70,
                                            "Aluminium_6082_T6": 40
                                        },
                            "grooving": {
                                            "DC01_ZE": 45,
                                            "Stainless_304": 65,
                                            "Aluminium_6082_T6": 35
                                        },
                            "pocketing": {
                                            "DC01_ZE": 60,
                                            "Stainless_304": 80,
                                            "Aluminium_6082_T6": 50
                                        },
                            "assembling": {
                                            "DC01_ZE": 90,
                                            "Stainless_304": 90,
                                            "Aluminium_6082_T6": 90
                                        },
                            "quality_control": {
                                                "DC01_ZE": 60,
                                                "Stainless_304": 60,
                                                "Aluminium_6082_T6": 60
                                            }
                        }
        if process_name in process_time_map_in_seconds and material in process_time_map_in_seconds[process_name]:
            time_per_part = process_time_map_in_seconds[process_name][material]
            task_time_min = math.ceil((produce_amount * time_per_part) / 60)  # Convert to minutes
        else:
            self.get_logger().error(f"Unknown process '{process_name}' or material '{material}' for time calculation. Falling back to default time of 45.")
            task_time_min = 45 * produce_amount
            
        return task_time_min
        
    def add_new_job(self, user_input_data: Dict[str, Any]):
        # Generate unique job ID
        user_input_data = self.generate_job_ID(user_input_data)
        # Extract first process and remaining processes
        for i in range(len(user_input_data['process_order'])):  
            firs_process = user_input_data['process_order'][i]
            
            # Define task
            task = {
                        "task_ID": user_input_data['job_ID'] + "_" + firs_process + f"_{str(i+1).zfill(2)}",
                        "job_ID": user_input_data['job_ID'],
                        "priority": user_input_data['priority'],
                        "process": firs_process,
                        "machine": self.select_machine_for_process(firs_process),
                        "depending_on": None if i == 0 else self.pending_operations[-1]['task_ID'],  # None for first process, else previous process task_ID
                        "task_time_min": self.task_time_calculator(user_input_data['produce_amount'], firs_process, user_input_data['material']),
                        "part_weight_kg": user_input_data['part_weight_kg'],
                        "part_thickness_mm": user_input_data['part_thickness_mm'],
                        "part_width_mm": user_input_data['part_width_mm'],
                        "surface_quality_mm": user_input_data["surface_quality_mm"],
                        "tolerance_mm": user_input_data["tolerance_mm"],
                        "produce_amount": user_input_data["produce_amount"],
                        "material": user_input_data["material"],
                        "status": "PENDING",
                        "mode": user_input_data.get("mode", "FAST")
                    }
            # Add job to job order list
            self.pending_operations.append(task)
        self.job_order.append(user_input_data)

    def is_machine_available(self, machine_id): #!! REMAINING TODO !!
        """
        Check if the specified machine is available.
        """
        if machine_id in self.running_operations:
            return False
        if machine_id in self.maintenance_queue_details:
            return False
        return True
    
    def listener_user_input_callback(self, msg: String):
        """
        Callback function for User Input subscription.
        Processes user input for job scheduling.
        """
        try:
            user_input_data = json.loads(msg.data)  # JSON string to dict
            self.get_logger().info(
                                    f"\nReceived User Input item: '{user_input_data['job_name']}'"
                                    f"\nProduction Amount: {user_input_data['produce_amount']} "
                                    )
            
            self.user_input_received = True     
            self.add_new_job(user_input_data)       
            self.schedule_conducter()
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    # Callback for maintenance queue from Controller_Node
    def listener_maintenance_queue_callback(self, msg: String):
        """
        Callback function for Maintenance Queue subscription.
        Re-organizes job_orders according to maintenance requirements.
        """
        data = json.loads(msg.data)
        machine = data["machine"]

        self.get_logger().info(
            f"[TEMP:DEBUG] MAINTENANCE_QUEUE_RECEIVED: machine={machine}, "
            f"recovery_time_min={data.get('recovery_time_min', 0)}, "
            f"remaining_min={data.get('remaining_min', 0)}"
        )

        self.maintenance_queue_details[machine] = data
        self.get_logger().info(f"{machine} added to maintenance queue")
        self.maintenance_required = True
        self.maintenance_completed = False
        self.schedule_conducter()
    
    # Callback for maintenance feedback from Machines
    def listener_maintenance_feedback_callback(self, msg: String):
        """
        Callback function for Maintenance Feedback subscription.
        Processes maintenance completion feedback from machines.
        """
        feedback_data = json.loads(msg.data)
        machine = feedback_data["machine"]
        status = feedback_data["status"]

        self.get_logger().info(
            f"[TEMP:DEBUG] MAINTENANCE_FEEDBACK_RECEIVED: machine={machine}, status={status}"
        )

        if status == "READY" and machine in self.maintenance_queue_details:
            # remove the machine from maintenance queue details
            del self.maintenance_queue_details[machine]
            self.get_logger().info(f"[TEMP:DEBUG] MACHINE_READY: {machine} removed from maintenance queue, resuming operations")
            self.get_logger().info(f"{machine} maintenance completed and removed from maintenance queue")

            if not self.maintenance_queue_details:
                self.maintenance_required = False
                self.maintenance_completed = True
            
            self.schedule_conducter()

    # Callback for Completed
    def listener_completed_callback(self, msg: String):
        """
        Callback function for Completed  subscription.
        Processes control commands for corrective/preventative actions.
        """

        # listen for process_finished signal & completed task data 
        completed_data = json.loads(msg.data)
        if completed_data['status'] == 'COMPLETED':
            self.get_logger().info(f"Received Completed item: {json.dumps(completed_data, indent=2)}")
            self.completed_task = completed_data
            self.process_finished = True
            self.schedule_conducter()
    
    def publish_job_orders(self, ready_tasks=None):
        """
        Publish next order (target job) to the Job_Orders topic.
        Removes the published job from the job_order list.
        """
        job_order_msg = String()


        # Publish the ready task
        job_order_msg.data = json.dumps(ready_tasks if ready_tasks else [])
        self.publisher_job_orders.publish(job_order_msg)
        self.get_logger().info(f"Published Job Orders:\n{json.dumps(json.loads(job_order_msg.data), indent=2)}")
 
    def schedule_conducter(self):
        """
        Control job_order and process_order lists.
        Conduct publishing of job orders based on completion status.
        """
        # Handling user input received logic
        if self.user_input_received:
            ready_tasks = self.load_balancer()
            if ready_tasks:
                self.publish_job_orders(ready_tasks)
                # remove the published task from pending operations and add to running operations
                for task in ready_tasks:
                    self.pending_operations.remove(task)
                    self.running_operations[task['machine']] = task
            self.user_input_received = False
        

        # Handling maintenance required logic
        if self.maintenance_required:
            # update running operations with maintenance info
            for machine, info in list(self.maintenance_queue_details.items()):
                if machine in self.running_operations:
                    task = self.running_operations[machine]
                    task["status"] = "MAINTENANCE"
                    task["task_time_min"] = info["remaining_min"]
                    # move the task back to pending operations
                    self.pending_operations.append(task)
                    del self.running_operations[machine] # remove from running operations
            
            # reset flag
            self.maintenance_required = False
            ready_tasks = self.load_balancer()
            
            if ready_tasks:
                self.publish_job_orders(ready_tasks)
                # remove the published task from pending operations and add to running operations
                for task in ready_tasks:
                    self.pending_operations.remove(task)
                    self.running_operations[task['machine']] = task
        
        # Handling maintenance completed logic
        if self.maintenance_completed:
            # remove machine from maintenance queue details
            self.maintenance_completed = False
            ready_tasks = self.load_balancer()
            if ready_tasks:
                self.publish_job_orders(ready_tasks)
                # remove the published task from pending operations and add to running operations
                for task in ready_tasks:
                    self.pending_operations.remove(task)
                    self.running_operations[task['machine']] = task
        
        # following the COMPLETION of process
        if self.process_finished:
            completed_task_ID = self.completed_task['task_ID']
            for pending_op in self.pending_operations[:]:
                if pending_op['depending_on'] == completed_task_ID:
                    pending_op['depending_on'] = None  # Mark dependency as resolved 
            # add the completed task to completed operations
            self.completed_operations.add(self.completed_task['task_ID'])
            self.process_finished = False # Reset process finished flag 

            # remove the completed task from running operations
            machine_id = self.completed_task['machine']
            if self.completed_task['task_ID'] == self.running_operations[machine_id]['task_ID']:
                del self.running_operations[machine_id]
            
            ready_tasks = self.load_balancer()           # Rebalance jobs according to the new/remaining tasks
            if ready_tasks:
                self.publish_job_orders(ready_tasks)     # publish new ready tasks
                # remove the published task from pending operations and add to running operations
                for task in ready_tasks:
                    self.pending_operations.remove(task)
                    self.running_operations[task['machine']] = task
            
            # check for job completion    
            for order in self.job_order[:]:  
                job_tasks = [t for t in self.pending_operations + list(self.running_operations.values()) 
                            if t['job_ID'] == order['job_ID']]
                if not job_tasks:  # no task = job completed
                    self.job_order.remove(order)
                    self.get_logger().info(f"Job {order['job_ID']} fully COMPLETED and removed.")

    def load_balancer(self):
        """
        Load balancer to manage job scheduling.
        Sorts task orders by priority, total cycle count, and arrival time (FIFO).
        """
        priority_map = {"high": 0, "medium": 1, "low": 2}
        ready_tasks = [t for t in self.pending_operations if t['depending_on'] is None 
                       and self.is_machine_available(t['machine'])
                       ]
        
        ready_tasks.sort(
                            key=lambda t: (
                                            priority_map.get(t['priority'], 3), # Sort by primarily priority (high to low)
                                            t['task_time_min'],                     # Sort by task_time_min (lower is higher priority)(SJF)
                                            int(t['job_ID'][-3:])               # Sort by arrival time (FIFO based on job_ID suffix)
                                          )
                        )
        return ready_tasks
        
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