#!/usr/bin/env python3

"""
ROS2 Node for generating control commands (control_CMD) to the machines according to the RUL predictions from the predictor node.

Subscribes to Predictions topic for RUL prediction input.
Subscribes to Job_Scheduler topic for job orders to see the remaining cycle to finish job. 

Publishes to Control_CMD topic for control commands to the machines.
Publishes to Maintenance_Queue topic for maintenance scheduling for Job_Scheduler Node.  

NOTE: In the system, job orders are the sole trigger for machine activation. After maintenance, READY is evaluated only by the scheduler; for the controller, the arrival of a job order is the invariant that guarantees the machine is operational.
"""
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import json
import random
import math


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
        self.subscription_job_scheduler_node = self.create_subscription(String, 'Job_Orders', self.listener_job_scheduler_callback, 10)    

        # Publishers for predictions
        self.control_cmd_publishers = {
                                        "hydraulic_press": self.create_publisher(String, "control_CMD/hydraulic_press", 10),
                                        "process_pump": self.create_publisher(String, "control_CMD/process_pump", 10)
                                    }
        # Publisher for maintenance queue
        self.maintenance_publisher = self.create_publisher(String, "Maintenance_Queue", 10)

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
        self.process_pump_current_cycle = 0
        self.process_pump_last_received_rul = 0
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

            if prediction.get('machine') == 'process_pump':

                # payload 
                cycle = prediction.get('cycle')
                rul_payload = prediction.get('rul')

                if isinstance(rul_payload, dict):
                    rul_min = rul_payload.get('rul_min')
                    rul_unit = (rul_payload.get('unit') or '').lower()
                else:
                    rul_min = None
                    rul_unit = ''

                if rul_min is None or cycle is None:
                    self.get_logger().warning("Incomplete prediction payload")
                    return

                # UNIT NORMALIZATION
                # system wide 1 cycle = 1 minute
                rul_value = float(rul_min)
                if rul_unit in {"hour", "hours", "hr", "hrs"}:
                    rul_value *= 60.0

                rul_cycles = int(rul_value)

                # Keep previous value for spike checks inside compute_control_CMD
                prev_rul = int(self.process_pump_last_received_rul)
                self.process_pump_current_cycle    = cycle

                self.compute_control_CMD(
                                            rul         = rul_cycles,
                                            prev_rul    = prev_rul,
                                            cycle       = self.process_pump_current_cycle,
                                            total_cycle = self.process_pump_total_cycles,
                                            machine     = 'process_pump'
                                        )

                # Update last received after decision
                self.process_pump_last_received_rul = rul_cycles

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    
    def listener_job_scheduler_callback(self, msg: String):
        """
        Callback function for job scheduler to get the total cycle count.
        Expland if statement for each new main machine - add the helper as they will have the same total cycle count. 
        
        - Controller state reset is implicitly handled via Job_Orders.
        - In this system, machines do not generate sensor data unless a job_order is active.
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Job_Scheduler message")

            # Any new job order implies (re)activation -> reset controller state
            if isinstance(received_data, list) and received_data:
                self.machine_status['hydraulic_press'] = 1
                self.machine_status['process_pump'] = 1

            for task in received_data:
                # process_pump is a helper machine triggered by the main hydraulic_press job
                if task.get('machine') == 'hydraulic_press' and task.get('depending_on') is None:
                    try:
                        self.process_pump_total_cycles = task.get('task_time_min', 0)
                        
                        # resetting machine specific variables
                        self.process_pump_current_cycle = 0
                        self.process_pump_last_received_rul = 0

                    except (ValueError, TypeError):
                        self.process_pump_total_cycles = 0

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    def compute_control_CMD(self, rul: float, prev_rul: float, cycle: int, total_cycle: int, machine=None):
        """
        Compute control command based on RUL and current cycle.
        """
        command = "NORMAL_OPERATION"

        if machine == "process_pump":

            # Guard: don't proceed if total_cycle is unknown
            if total_cycle <= 0:
                self.get_logger().warning("Total cycle unknown, skipping control decision")
                return

            remaining_min = max(0, total_cycle - cycle)

            if abs(rul - prev_rul) > 2000:
                self.get_logger().warning("RUL spike detected, ignoring")
                return

            # HARD CRITICAL ZONE
            if rul <= self.process_pump_CRITICAL:
                command = "SHUTDOWN"

            # WARNING ZONE
            elif rul <= self.process_pump_WARNING:

                # Required buffer to safely finish the job
                required_time = remaining_min + self.process_pump_SAFE_BUFFER

                if rul >= required_time:
                    command = "SLOW_DOWN"
                else:
                    command = "SHUTDOWN"

            # SAFE ZONE
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
        
        
        current_state = self.machine_status[machine]
        new_state = 1 if command == "NORMAL_OPERATION" else (2 if command == "SLOW_DOWN" else 3)

        # Check if the state is escalating or not
        # Only publish if the machine state is escalating
        if new_state > current_state:

            self.machine_status[machine] = new_state
            recovery_time_min, remaining_min = self.compute_maintenance_schedule(machine)

            control_msg = String()
            control_msg.data = json.dumps({
                                            "machine": machine,
                                            "cycle": cycle,
                                            "rul": rul,
                                            "command": command,
                                            "recovery_time_min": recovery_time_min
                                        })
            self.control_cmd_publishers[machine].publish(control_msg)                       # informs the machine
            self.publish_maintenance_schedule(machine, recovery_time_min, remaining_min) # informs the job scheduler
            self.get_logger().info(f"Published Control Command for {machine} at cycle {cycle}: RUL={rul}, Command={command}")
            # trigger the maintenance scheduling

    def compute_maintenance_schedule(self, machine=None):
        """
        Compute maintenance schedule based on machine type.
        """
        recovery_time_min = 0
        remaining_min = 0

        # additional control layer just in case.
        if self.machine_status[machine] > 1:  # type: ignore
            if machine == "process_pump":
                remaining_min = self.process_pump_total_cycles - self.process_pump_current_cycle
                if self.machine_status[machine] == 2:           # SLOW_DOWN
                    recovery_time_hrs = random.randint(1, 3)    # Random recovery time between 1 to 3 hours
                    recovery_time_min = recovery_time_hrs * 60  # Convert to minutes aka cycles in the system
                
                elif self.machine_status[machine] == 3:         # SHUTDOWN
                    recovery_time_hrs = random.randint(4, 12)   # Random recovery time between 4 to 12 hours
                    recovery_time_min = recovery_time_hrs * 60  # Convert to minutes aka cycles in the system
            
            elif machine == "hydraulic_press":
                if self.machine_status[machine] == 2:
                    recovery_time_hrs = random.randint(2, 6)
                    recovery_time_min = recovery_time_hrs * 60
                
                elif self.machine_status[machine] == 3:
                    recovery_time_hrs = random.randint(8, 24)
                    recovery_time_min = recovery_time_hrs * 60
        
        return recovery_time_min, remaining_min

                
    def publish_maintenance_schedule(self, machine:str, recovery_time_min:int, remaining_min:int=0):
        """
        Publish maintenance schedule to Maintenance_Queue topic.
        """

        maintenance_msg = String()
        maintenance_msg.data = json.dumps({
                                            "machine": machine,
                                            "recovery_time_min": recovery_time_min,
                                            "remaining_min": remaining_min
                                        })
        self.maintenance_publisher.publish(maintenance_msg)
        self.get_logger().info(f"Published Maintenance Schedule for {machine}: Recovery Time={recovery_time_min} minutes/cycles")
                
                
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

