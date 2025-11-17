#!/usr/bin/env python3

"""
ROS2 Node for overviewing the whole system.

Pompts user for job_order selection or creation. 
Reads from selected or creates and writes the entered job_order from src/system_nodes/system_nodes/job_orders.json.

Publishes to User_Input Topic for Job_Scheduler_Node.

Subscribes to below topics;
- Sensors topic
- Completed topic
- Job_Orders topic
- Control_CMD topic
- Maintenance_Queue topic

Currently prints them to the console.
Later evolves to streamlit dashboard for better GUI.
"""
import os
import json
import rclpy
import launch
from rclpy.node import Node
from std_msgs.msg import String
import launch.launch_description_sources

class temp_GUI_Node(Node):
    """ROS2 Node for overviewing the whole system."""

    def __init__(self):
        """
        Initialize the temp GUI node, set up subscriptions and publishers.
        """
        super().__init__('temp_GUI_Node')
        
        # Subscription to Sensors
        self.subscription_sensors = self.create_subscription(String, 'Sensors', self.listener_callback_sensors, 10)

        # Subscription to Completed
        self.subscription_completed = self.create_subscription(String, 'Completed', self.listener_callback_completed, 10)
        
        # Subscription to Job Orders
        self.subscription_job_order = self.create_subscription(String, 'Job_Orders', self.listener_callback_job_orders, 10)
    
        # Subscription to Control CMD
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD', self.listener_callback_control_cmd, 10)
    
        # Subscription to Maintenance Queue
        self.subscription_maintenance_queue = self.create_subscription(String, 'Maintenance_Queue', self.listener_callback_maintenance_queue, 10)

        # Publisher for Sensors data
        self.publisher_user_input = self.create_publisher(String, "User_Input", 10)

        
        
        self.get_logger().info("Process temp_GUI_Node ready!")
        self.get_logger().info("Listening on 'Sensors' topic.")
        self.get_logger().info("Listening on 'Completed' topic.")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Control_CMD' topic.")
        self.get_logger().info("Listening on 'Maintenance_Queue' topic.")

        self.job_names = []
        self.data = []
        self.target_job = None
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
        
        print("\n == Welcome to the Job Scheduler! ==\n")
        self.order_collect_main()

    def total_cycle_calculator(self, produce_amount, cycles_per_part):
            return produce_amount * cycles_per_part

    def job_selecting(self):
            
            
            correct_selection = False

            while not correct_selection:    

                a = int(input("\nEnter the job name number from the list: "))
                if a > len(self.job_names):
                    print("Invalid selection. Please try again.")
                    continue

                job_name_input = self.job_names[a-1]


                self.target_job = next((job for job in self.data if job["job_name"] == job_name_input), None)

                if self.target_job:
                    job_ID = self.target_job["job_ID"]
                
                    # Get user input for production amount
                    produce_amount = int(input("Enter the production amount:             "))
                    
                    self.target_job["produce_amount"] = produce_amount
                    self.target_job["total_cycle_count"] = self.total_cycle_calculator(produce_amount, self.target_job["cycles_per_part"])

                    print(f"\n Selected Job Order: {self.target_job['job_name']}")
                    print("\n Published order")
                    
                    print(f"\n{json.dumps(self.target_job, indent=4)}")

                    correct_selection = True

                else:
                    print(f"{job_name_input} not found in job orders.")

    def job_entering(self):

            while True: 
                print("\nEnter the job order as JSON like below: ")
                
                print(self.example_json)
                
                job_order_input = input("\nPaste the job order JSON here: ")
                
                try:
                    job_order_data = json.loads(job_order_input)
                    job_name_input = job_order_data["job_name"]
                    job_ID = job_order_data["job_ID"]

                    # check if the entered job already exists
                    if self.job_entry_check(job_name_input):
                        print(f"\nERROR: Job name '{job_name_input}' already exists as a job order. Please enter a unique job name.")
                        input("\nPress Enter to try again...")
                        continue


                    produce_amount = int(input("Enter the production amount:             "))
                    
                    job_order_data["produce_amount"] = produce_amount
                    job_order_data["total_cycle_count"] = self.total_cycle_calculator(produce_amount, job_order_data["cycles_per_part"])

                    # first read the existing job orders from JSON file
                    file_path = os.path.join(os.path.dirname(__file__), "job_orders.json")
                    if os.path.exists(file_path):
                        self.data = self.read_json_file(file_path)
                    else:
                        print("Job orders JSON not found. A new file will be created.")
                        self.data = []

                    # add the new job order to the list
                    self.data.append(job_order_data)

                    # then write to the JSON file
                    self.write_json_file(file_path, self.data)            
                    break
                    
                except Exception as e:
                    print("\nERROR: Invalid JSON input:", e)
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
                self.job_names = []
                for job in self.data:
                    self.job_names.append(job['job_name'])

                # Get user input for existing job or new job
                print(f"Available Operations:")
                for i, job in enumerate(self.job_names, 1):
                    print(f"{i}. {job}")

                while True:
                    user_input = input("\nDoes the desired job already exist in above list? (yes/no): ").strip().lower()
                    if user_input[0] == 'y':
                        self.job_selecting()
                        break
                    elif user_input[0] == 'n':
                        self.job_entering()
                        break
                    else:
                        print("\nInvalid input. Please enter 'yes' or 'no'.")
                        continue

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
    
    # Callback functions
    #--------------------
    # Callback for Sensors # !! TO DO !! -> EXPAND #
    def listener_callback_sensors(self, msg: String):
        """
        Callback function for Sensors subscription.
        """
        self.get_logger().info(f"Sensor Data sent to Sensors: {msg.data} From Machine_X") 

    # Callback for Completed # !! TO DO !! -> EXPAND #
    def listener_callback_completed(self, msg: String):
        """
        Callback function for Completed subscription.
        """
        self.get_logger().info(f"Scheduler Received Completed: {msg.data} From Machine_X") 
    
    # Callback for Job Orders # !! TO DO !! -> EXPAND #
    def listener_callback_job_orders(self, msg: String):
        """
        Callback function for Job Orders subscription.
        """
        
        self.get_logger().info(f"Job_Scheduler Received Job Order: {msg.data}")
        
    # Callback for Control CMD # !! TO DO !! -> EXPAND #
    def listener_callback_control_cmd(self, msg: String):
        """
        Callback function for Control CMD subscription.
        """
        self.get_logger().info(f"Machine_X Received Control Command: {msg.data}")
    
    # Callback for Maintenance Queue # !! TO DO !! -> EXPAND #
    def listener_callback_maintenance_queue(self, msg: String):
        """
        Callback function for Maintenance Queue subscription.
        """
        self.get_logger().info(f"Machine_X Received Maintenance Queue: {msg.data}")     
    #--------------------

    def publish_user_input(self):
        """
        Publish user input to the User_Input topic.
        """
        user_input_msg = String()

        if self.target_job is not None:
            user_input_msg.data = json.dumps(self.target_job, indent=4)
            self.publisher_user_input.publish(user_input_msg)
        else:
            msg = String()
            msg.data = " No valid job order to publish."
            self.publisher_user_input.publish(msg)
            
    def order_collect_main(self):
        while rclpy.ok():
            self.get_job_order()        # Get job order from user input or JSON file.
            self.publish_user_input()   # Publish the selected or created job order to User_Input topic.
            input("\nPress Enter to continue to next job order or Ctrl+C to exit...")

def main(args=None):
    """
    Main entry point for the machine_hydraulic_press_sensor_node.
    """
    rclpy.init(args=args)
    node = temp_GUI_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()