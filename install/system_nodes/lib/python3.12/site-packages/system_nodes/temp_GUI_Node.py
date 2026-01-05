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
from rclpy.node import Node
from std_msgs.msg import String


class temp_GUI_Node(Node):
    """ROS2 Node for overviewing the whole system."""

    def __init__(self):
        """
        Initialize the temp GUI node, set up subscriptions and publishers.
        """
        super().__init__('temp_GUI_Node')
        
        # Subscription to Sensors
        self.subscription_sensors = self.create_subscription(String, 'Sensors/hydraulic_press', self.listener_callback_sensors, 10)
        self.subscription_sensors = self.create_subscription(String, 'Sensors/process_pump', self.listener_callback_sensors, 10)

        # Subscription to Completed
        self.subscription_completed = self.create_subscription(String, 'Completed/hydraulic_press', self.listener_callback_completed, 10)
        
        # Subscription to Job Orders
        self.subscription_job_order = self.create_subscription(String, 'Job_Orders', self.listener_callback_job_orders, 10)
    
        # Subscription to Control CMD
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD/process_pump', self.listener_callback_control_cmd, 10)
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD/hydraulic_press', self.listener_callback_control_cmd, 10)
    
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
                                    "part_type": "metal_sheet / other ",
                                    "material": "type of material (e.g., st37, aluminium)",
                                    "part_thickness_mm": number,
                                    "part_width_mm": number,
                                    "part_weight_kg":  number,
                                    "process_order": ["bending", "forming", "drilling", "grooving", "pocketing", "assembling", "quality_control"],
                                    "surface_quality_mm": number,
                                    "tolerance_mm": number
                            }}'''
        
        print("\n == Welcome to the Job Scheduler! ==\n")
        self.order_collect_main()



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

                    # Get user to verify or enter job priority for load balancing
                    priority = input(f"-->'{self.target_job['job_name']}' order's current priority is '{self.target_job['priority']}'.\n Enter new priority or press Enter to keep it: ").strip().lower()
                    if priority:
                        self.target_job["priority"] = priority
                    
                    self.target_job["produce_amount"] = produce_amount
                
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
                    
                    # check if the entered job is correctly formed
                    self.job_entry_check(job_order_data)                    

                    # Get user input for production amount
                    produce_amount = int(input("Enter the production amount:             "))
                    job_order_data["produce_amount"] = produce_amount

                    # Get user input for job priority for load balancing
                    priority = input("Enter the job priority (low / medium / high): ").strip().lower()
                    job_order_data["priority"] = priority

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
            
    def job_entry_check(self, job_order_data):
            while True:
                for job in self.data:
                    # checking for duplicate job name
                    if job['job_name'] == job_order_data['job_name']:
                        a = input(f"\nERROR: Job name '{job_order_data['job_name']}' already exists as a job order. Please enter a unique job name:  ")
                        if a and all(a != job['job_name'] for job in self.data):
                            job['job_name'] = a
                        else:
                            print("please enter a valid job name.")
                            continue

                    # or duplicate job ID
                    if job['job_ID'] == job_order_data['job_ID'] or job_order_data['job_ID'] == None or job_order_data['job_ID'] == "":
                        a = input(f"\nERROR: Job ID '{job_order_data['job_ID']}'. Please enter a unique Job ID:  ")
                        if a and all(a != job['job_ID'] for job in self.data):
                            job['job_ID'] = a
                        else:
                            print("please enter a valid Job ID.")
                            continue    

                # cheking for missing material
                if job_order_data['material'] == None or job_order_data['material'] == "":
                    a = input(f"\nERROR: Material is missing for the '{job_order_data['job_name']}'. Please enter a valid material:  ")
                    if a != None and a != "":
                        job_order_data['material'] = a
                    else:
                        print("please enter a valid material.")
                        continue

                # cheking for missing part thickness
                if job_order_data['part_thickness_mm'] == None or job_order_data['part_thickness_mm'] == "":
                    a = input(f"\nERROR: Part thickness is missing for the '{job_order_data['job_name']}'. Please enter a valid part thickness in mm:  ")
                    if a != None and a != "" and a.isdigit():
                        job_order_data['part_thickness_mm'] = a
                    else:
                        print("please enter a valid part thickness in mm.")
                        continue
                # Cheking for missing part width
                if job_order_data['part_width_mm'] == None or job_order_data['part_width_mm'] == "":
                    a = input(f"\nERROR: Part width is missing for the '{job_order_data['job_name']}'. Please enter a valid part width in mm:  ")
                    if a != None and a != "" and a.isdigit():
                        job_order_data['part_width_mm'] = a
                    else:
                        print("please enter a valid part width in mm.")
                        continue
                # cheking for missing part weight
                if job_order_data['part_weight_kg'] == None or job_order_data['part_weight_kg'] == "":
                    a = input(f"\nERROR: Part weight is missing for the '{job_order_data['job_name']}'. Please enter a valid part weight in kg:  ")
                    if a != None and a != "" and a.isdigit():
                        job_order_data['part_weight_kg'] = a
                    else:
                        print("please enter a valid part weight in kg.")
                        continue
                # cheking for missing process type
                if job_order_data['process_order'] == None or job_order_data['process_order'] == "":
                    a = input(f"\nERROR: Process type is missing for the '{job_order_data['job_name']}'. Please enter at least one process type in brackets [], separated by commas, with quotes e.g., [\"bending\", \"drilling\"]:  ")
                    if a != None and a != "":
                        job_order_data['process_order'] = a
                    else:
                        print("please enter at least one process type.")
                        continue
                # cheking for missing surface quality and tolerance
                if job_order_data['surface_quality_mm'] == None or job_order_data['tolerance_mm'] == None:
                    a = input("\nNote: No surface quality and tolerance detected, these will be set to 0 if not critical for the job. Please enter surface quality in and tolerance in mm with comma separated values, or press Enter to set them to 0: ")
                    if a == None or a == "":
                        job_order_data['surface_quality_mm'] = 0
                        job_order_data['tolerance_mm'] = 0
                    else:
                        try:
                            surface_quality, tolerance = map(float, a.split(','))
                            job_order_data['surface_quality_mm'] = surface_quality
                            job_order_data['tolerance_mm'] = tolerance
                        except ValueError:
                            print("Invalid input. Please enter two numeric values separated by a comma.")
                            continue
                # final check complete
                if job_order_data['job_name'] != None and job_order_data['job_name'] != "" and job_order_data['job_ID'] != None and job_order_data['job_ID'] != "" and job_order_data['material'] != None and job_order_data['material'] != "" and job_order_data['part_thickness_mm'] != None and job_order_data['part_thickness_mm'] != "" and job_order_data['part_weight_kg'] != None and job_order_data['part_weight_kg'] != "" and job_order_data['process_order'] != None and job_order_data['process_order'] != "":
                    return
            
        
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
    Main entry point for the temporary GUI node.
    """
    rclpy.init(args=args)
    node = temp_GUI_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()