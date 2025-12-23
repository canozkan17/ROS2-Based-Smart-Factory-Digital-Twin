#!/usr/bin/env python3

"""
ROS2 Node for generating control commands (control_CMD) to the machines according to the RUL predictions from the predictor node.

Subscribes to Predictions topic for RUL prediction input.
Subscribes to Job_Scheduler topic for job orders to see the remaining cycle to finish job. 

Publishes to Control_CMD topic for control commands to the machines.
Publishes to Maintenance_Queue topic for maintenance scheduling for Job_Scheduler Node.  

"""
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import json


class Controller_Node(Node):

    def __init__(self):
        """
        Initialize the Controller node, set up subscriptions and publishers.
        """
        super().__init__('Controller_Node')
        
        # Subscription to machine_hydraulic_press_node  
        #self.subscription_hydraulic_press_node = self.create_subscription(String, 'Predictions/hydraulic_press', self.listener_hydraulic_press_callback, 10)        
        
        # Subscription to machine_process_pump_node  
        self.subscription_process_pump_node = self.create_subscription(String, 'Predictions/process_pump', self.listener_process_pump_predictions_callback, 10)    
        
        # Subscription to Job_Scheduler node  
        self.subscription_job_scheduler_node = self.create_subscription(String, 'Job_Scheduler', self.listener_job_scheduler_callback, 10)    

        # Publishers for predictions
        self.control_cmd_publishers = {
                                        "hydraulic_press": self.create_publisher(String, "control_CMD/hydraulic_press", 10),
                                        "process_pump": self.create_publisher(String, "control_CMD/process_pump", 10)
                                    }
        # Machine Status Control
        # NORMAL_OPERATION: 1 - default state
        # SLOW_DOWN: 2 - slow down machine operation to extend RUL
        # SHUTDOWN: 3 - shut down machine for maintenance
        self.machine_status = {
                                    "hydraulic_press": 1,
                                    "process_pump": 1
                                }
        
        # Process Pump Variables
        self.process_pump_total_cycles = 0
        self.process_pump_current_cycles = 0
        self.process_pump_last_recieved_rul = 0
        self.process_pump_CRITICAL = 20             # critical RUL threshold to send maintenance
        self.process_pump_WARNING  = 60             # warning RUL threshold to slow down machine
        self.process_pump_SAFE_BUFFER = 10          # safe buffer to avoid false alarms
        
        # Set-up logs
        self.get_logger().info("Controller node ready!")
        self.get_logger().info("Listening on 'Predictions/ ' topic. For 2 machine sensors")
        
    # PROCESS PUMP MACHINE METHOD
    def listener_process_pump_predictions_callback(self, msg: String):
        """
        Callback function for Process_Pump Predictions.
        """
        try: 
            prediction = json.loads(msg.data)
            self.get_logger().info(f"Received Process_Pump Predictions message")

            if prediction.get('machine') == 'process_pump' and prediction.get('rul') is not None: # predictions are machine specific (main or helper)

                self.process_pump_last_recieved_rul = prediction['rul']
                self.process_pump_current_cycles    = prediction['cycle']

                self.compute_control_CMD(
                                            rul         = self.process_pump_last_recieved_rul, 
                                            cycle       = self.process_pump_current_cycles, 
                                            total_cycle = self.process_pump_total_cycles, 
                                            machine     ='process_pump'
                                        )
                    

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
    
    def listener_job_scheduler_callback(self, msg: String):
        """
        Callback function for job scheduler to get the total cycle count.
        Expland if statement for each new main machine - add the helper as they will have the same total cycle count. 
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Job_Scheduler message")
            for task in received_data:
                if task['machine'] == 'hydraulic_press' and task['depending_on'] is None: # because there is no seperate job order for the helper machine

                    self.process_pump_total_cycles = task['task_time']

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    def compute_control_CMD(self, rul:float, cycle:int, total_cycle:int ,machine=None):
        """
        Compute control command based on RUL and current cycle.
        """
        command = "NORMAL_OPERATION"

        if machine == "process_pump":

            if rul <= self.process_pump_CRITICAL:
                command = "SHUTDOWN"

            else:

                remaining_cycles = total_cycle - cycle
                required_time = remaining_cycles + self.process_pump_SAFE_BUFFER

                # Job feasibility check
                if rul <= self.process_pump_WARNING:
                    if rul > required_time:
                        command = "SLOW_DOWN"
                    else:
                        command = "SHUTDOWN"
                
                # Safe Zone
                else:
                    command = "NORMAL_OPERATION"



        self.publish_control_CMD(rul, cycle, command, machine)

            

    def publish_control_CMD(self, rul:float, cycle:int, command:str, machine=None):
        """
        Controls if the machine command has been changed from Normal_Operation to another state.
        Publish control command message to the appropriate topic.
        """
        if not machine or machine not in self.machine_status:
            self.get_logger().error(f"Invalid machine type for control command publishing: {machine}")
            return
        
        # Check if the state is escalating or not
        if self.machine_status[machine] <= 1:
            current_state = self.machine_status[machine]
            new_state = 1 if command == "NORMAL_OPERATION" else (2 if command == "SLOW_DOWN" else 3)

        # Only publish if the machine state is escalating
            if new_state > current_state:
                self.machine_status[machine] = new_state
                control_msg = String()
                control_msg.data = json.dumps({
                                                "machine": machine,
                                                "cycle": cycle,
                                                "rul": rul,
                                                "command": command
                                            })
                self.control_cmd_publishers[machine].publish(control_msg)
                self.get_logger().info(f"Published Control Command for {machine} at cycle {cycle}: RUL={rul}, Command={command}")

def main(args=None):
    """
    Main entry point for the Predictor_Node.
    """
    rclpy.init(args=args)
    node = Controller_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

