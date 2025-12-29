#!/usr/bin/env python3

"""
Listenes to Job Orders, and Maintenance Queue topics.
Publishes to Sensors and Completed topics.

Handles tasks for the Hydraulic Press machine.
"""

from std_msgs.msg import String
from rclpy.node import Node
import numpy as np
import threading
import random
import rclpy
import json
import math
import time


class Machine_Hydraulic_Press_Node(Node):
    def __init__(self):
        super().__init__('Machine_Hydraulic_Press_Node')
        
        # Subscription to Job Orders
        self.subscription_job_orders = self.create_subscription(String, 'Job_Orders', self.listener_job_orders_callback, 10)      
        # Subscription to Control_CMD
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD/hydraulic_press', self.listener_control_cmd_callback, 10)
        self.subscription_control_cmd_2 = self.create_subscription(String, 'Control_CMD/process_pump', self.listener_control_cmd_callback, 10) # for helper machine
              
        # Publisher for Sensors data
        self.publisher_sensors = self.create_publisher(String, "Sensors/hydraulic_press", 10)
        # Publisher for Completed status
        self.publisher_completed = self.create_publisher(String, "Completed/hydraulic_press", 10)
        # Publisher for maintenance feedback 
        self.publisher_maintenance_feedback = self.create_publisher(String, "Maintenance_Feedback/hydraulic_press", 10)
            
        
        # Variable set-up
        self.current_task = {}
        self.total_ran_cycles = 0
        self.total_production = 0
        self.total_time_producing = 0.0

        # Simulation and production control flags
        # Set "REALTIME" for 60s per cycle, "FAST" for max speed
        self.simulation_mode = "FAST"      # or "REALTIME" TODO: Make this configurable in gui later  
        self.production_timer = None       # Holds the rclpy.Timer object
        self.cycles_to_run = 0
        self.cycles_done = 0

        self.in_maintenance = False
        self.control_cmd = "NORMAL_OPERATION"

        # Defaults
        self.SEED = 42
        self.degredation_factor = 1.0

        rng = np.random.default_rng(self.SEED)
        
        # total_rul (MUST be first to sync RNG state with training)
        self.total_lifetime_minutes = int(rng.integers(120_000, 720_000))  # matches training
        self.max_lifetime = self.total_lifetime_minutes / 60.0  # convert to hours
        self.get_logger().info(f"Hydraulic_Press initialized with max lifetime: {self.max_lifetime:.1f} hours ({self.total_lifetime_minutes} min)")

        # HYDRAULIC PRESS SENSOR PARAMETERS
        # Hydraulic pressure (bar) - seal wear dominant
        self.base_pressure = rng.uniform(180.0, 220.0)
        self.pressure_drop_rate = rng.uniform(0.00005, 0.00015)
        self.pressure_noise = rng.uniform(0.5, 2.0)

        # Oil temperature (°C) - oil degradation
        self.base_oil_temp = rng.uniform(45.0, 65.0)
        self.oil_temp_rate = rng.uniform(0.0003, 0.001)
        self.oil_temp_noise = rng.uniform(0.1, 0.6)

        # Oil contamination index (dimensionless)
        self.base_contamination = rng.uniform(0.5, 2.0)
        self.contamination_growth = rng.uniform(0.00002, 0.00008)
        self.contamination_noise = rng.uniform(0.01, 0.05)

        # Ram position deviation (mm) - misalignment
        self.base_ram_dev = rng.uniform(0.01, 0.05)
        self.ram_dev_growth = rng.uniform(0.00001, 0.00005)
        self.ram_dev_noise = rng.uniform(0.001, 0.005)

        # Press force / tonnage (tons)
        self.base_force = rng.uniform(80.0, 120.0)
        self.force_loss_rate = rng.uniform(0.00003, 0.0001)
        self.force_noise = rng.uniform(0.3, 1.5)

        # Frame / ram vibration (mm/s)
        self.base_vibration = rng.uniform(0.1, 0.5)
        self.failure_vibration = rng.uniform(3.0, 8.0)
        self.vibration_noise = rng.uniform(0.02, 0.1)

        # Hydraulic flow rate (L/min)
        self.base_flow = rng.uniform(90.0, 130.0)
        self.flow_loss_rate = rng.uniform(0.00004, 0.00012)
        self.flow_noise = rng.uniform(0.3, 1.2)

        # Motor current (A)
        self.base_current = rng.uniform(30.0, 55.0)
        self.current_growth_rate = rng.uniform(0.00005, 0.0002)
        self.current_noise = rng.uniform(0.1, 0.6)
        
        # Store RNG for noise generation during simulation
        self.RNG = rng

        # Set-up logs
        self.get_logger().info("Machine Hydraulic Press node ready!")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Control_CMD' topic.")



    def listener_job_orders_callback(self, msg: String):
        """
        Callback function for Job Orders subscription.
        """

        try: 
            received_data = json.loads(msg.data)  # JSON string to dict
            for task in received_data:
                if task['machine'] == 'hydraulic_press' and task['depending_on'] is None:
                    
                    self.current_task = task
                    
                    # Calculate cycles needed based on task_time (from Job_Scheduler)
                    # Each cycle simulates 60 seconds - training dataset compatibility
                    try:
                        task_time_seconds = float(self.current_task['task_time'])
                        # math.ceil to ensure to run enough cycles
                        self.cycles_to_run = int(math.ceil(task_time_seconds / 60.0))
                        self.cycles_done = 0
                        
                        if self.cycles_to_run <= 0:
                            self.get_logger().warn("Task time is zero or negative. Skipping task.")
                            self.current_task = {}
                            return

                    except (KeyError, ValueError, TypeError) as e:
                        self.get_logger().error(f"Invalid or missing 'task_time' in job. Cannot start. Error: {e}")
                        self.current_task = {}
                        return

                    self.get_logger().info(
                                                f"\nReceived task: '{json.dumps(task['job_ID'], indent=2)}'. "
                                                f"Total time: {task_time_seconds}s. "
                                                f"Calculated cycles: {self.cycles_to_run}"
                                            )

                    # Start production loop based on simulation mode
                    if self.simulation_mode == "REALTIME":
                        self.get_logger().info(f"Starting REALTIME simulation ({self.cycles_to_run} cycles)...")
                        # Create a non-blocking timer that calls the function every 60s
                        self.production_timer = self.create_timer(60.0, self.run_production_cycle)
                    
                    elif self.simulation_mode == "FAST":
                        self.get_logger().info(f"Starting FAST simulation ({self.cycles_to_run} cycles) in new thread...")
                        # Run the simulation in a separate thread to not block the ROS node
                        sim_thread = threading.Thread(target=self.run_fast_simulation, daemon=True)
                        sim_thread.start()
                    break 
                    
        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    def listener_control_cmd_callback(self, msg: String):
        """
        Callback function for Control CMD subscription.
        Processes control commands for corrective/preventative actions.
        """
        control_msg = json.loads(msg.data)
        self.get_logger().info(f"Received Control Command")

        # Extract control command and recovery time from the message
        command = control_msg.get("command", "NORMAL_OPERATION")
        self.maintenance_cycles_remaining = control_msg.get("recovery_time_min", 0)

        if command == "SHUTDOWN":
            self.get_logger().info(f"Received SHUTDOWN command. Stopping current task.")
            self.control_cmd = "SHUTDOWN"
            self.in_maintenance = True
            if self.production_timer:
                self.production_timer.cancel()
                self.production_timer = None

        elif command == "SLOW_DOWN":
            self.control_cmd = "SLOW_DOWN"
            self.get_logger().info(f"Received SLOW_DOWN command. Slowing down operation.")
            self.degredation_factor *= 0.9  # slowing down degradation by 10%
        
        
        elif command == "NORMAL_OPERATION":
            self.get_logger().info(f"Received NORMAL_OPERATION command. Continuing operation.")
        else:
            self.get_logger().warning(f"Unknown command received: {command}")

    def maintenance_timer(self):
        """
        Handles maintenance cycles when in maintenance mode.
        In REALTIME mode, sets a timer for maintenance duration.
        In FAST mode, decrements maintenance cycles immediately - causing friction of waiting for maintenance to finish thus simulating.
        """
        if not self.in_maintenance:
            return

        if self.simulation_mode == "REALTIME":
            self.get_logger().info(
                f"REALTIME maintenance started for {self.maintenance_cycles_remaining} cycles"
            )

            self.maintenance_realtime_timer = self.create_timer(
                                                                    self.maintenance_cycles_remaining * 60.0,
                                                                    self.finish_maintenance,
                                                                    callback_group=None
                                                                )

        elif self.simulation_mode == "FAST":
            while self.maintenance_cycles_remaining > 0:
                self.maintenance_cycles_remaining -= 1
                time.sleep(0.01)  # small sleep to simulate time passage
            self.finish_maintenance()
    
    def finish_maintenance(self):
        self.degredation_factor = 1.0
        self.in_maintenance = False
        self.maintenance_cycles_remaining = 0

        self.cycles_done = 0
        self.control_cmd = "NORMAL_OPERATION"
        self.current_task = {}

        # MAINTENANCE_FEEDBACK
        msg = String()
        msg.data = json.dumps({
                                    "machine": "process_pump",
                                    "status": "READY"
                                })
        
        self.publisher_maintenance_feedback.publish(msg) 

        self.get_logger().info(f"Maintenance completed. Machine READY. Degradation reset. Cycles done reset. Machine total ran cycles: {self.total_ran_cycles} minutes.")


    def generate_cycle(self, total_rul: int, rng: np.random.Generator):

        t = float(self.cycles_done)
        current_rul = total_rul - t - 1
        fraction = t / total_rul

        # CRITICAL REGION BOOST (RUL <= 600)
        critical_mask = current_rul <= 600
        critical_boost = np.where(
                                    critical_mask,
                                    1.0 + (600 - current_rul) / 600 * 0.6,
                                    1.0
                                )

        # SENSOR SIGNAL GENERATION
        hydraulic_pressure = (
                                self.base_pressure
                                - self.pressure_drop_rate * t * critical_boost * self.degredation_factor
                                + rng.normal(0.0, self.pressure_noise * (1.0 + fraction), size=total_rul)
                            )

        oil_temperature = (
                            self.base_oil_temp
                            + self.oil_temp_rate * t * critical_boost * self.degredation_factor
                            + rng.normal(0.0, self.oil_temp_noise, size=total_rul)
                        )

        oil_contamination = (
                                self.base_contamination
                                + self.contamination_growth * t * critical_boost * self.degredation_factor
                                + rng.normal(0.0, self.contamination_noise * (1.0 + fraction), size=total_rul)
                            )

        ram_position_deviation = (
                                    self.base_ram_dev
                                    + self.ram_dev_growth * t * critical_boost * self.degredation_factor
                                    + rng.normal(0.0, self.ram_dev_noise, size=total_rul)
                                )

        press_force = (
                        self.base_force
                        - self.force_loss_rate * t * critical_boost * self.degredation_factor
                        + rng.normal(0.0, self.force_noise * (1.0 + 0.5 * fraction), size=total_rul)
                    )

        vibration = (
                        self.base_vibration
                        + (self.failure_vibration - self.base_vibration) * (fraction ** 2) * critical_boost * self.degredation_factor
                        + rng.normal(0.0, self.vibration_noise * (1.0 + fraction), size=total_rul)
                    )

        flow_rate = (
                        self.base_flow
                        - self.flow_loss_rate * t * critical_boost * self.degredation_factor
                        + rng.normal(0.0, self.flow_noise * (1.0 + fraction), size=total_rul)
                    )

        motor_current = (
                            self.base_current
                            + self.current_growth_rate * t * critical_boost * self.degredation_factor
                            + rng.normal(0.0, self.current_noise * (1.0 + fraction), size=total_rul)
                        )

        cycle_output = {
                            "hydraulic_pressure": float(hydraulic_pressure),
                            "oil_temperature": float(oil_temperature),
                            "oil_contamination": float(oil_contamination),
                            "ram_position_deviation": float(ram_position_deviation),
                            "press_force": float(press_force),
                            "vibration": float(vibration),
                            "flow_rate": float(flow_rate),
                            "motor_current": float(motor_current),
                        }

        self.total_ran_cycles += 1
        return cycle_output

    def publish_completed_status(self):
        """
        Publishes completed status to Completed topic.
        """
        completed_msg = String()
        completed_msg.data = json.dumps(self.current_task)
        self.publisher_completed.publish(completed_msg)
        self.get_logger().info(f"Published completed status for job {self.current_task['job_ID']}.")
        self.current_task = {} # Clear task after completion

    def run_production_cycle(self):
        """
        Callback for the rclpy.Timer (REALTIME mode).
        Runs one production cycle every 60 seconds.
        """
        elapsed_minutes = float(self.cycles_done)
        elapsed_hours = elapsed_minutes / 60.0
        current_rul = max(0.0, self.max_lifetime - elapsed_hours)

        if current_rul <= 0:
            self.get_logger().error(f"Machine failure! RUL=0, stopping task {self.current_task.get('job_ID', 'unknown')}.")
            self.current_task['status'] = 'FAILED_DUE_TO_DEGRADATION'
            #TODO: Publish failure status in due time at this line. 
            if self.production_timer:
                self.production_timer.cancel()
                self.production_timer = None
            return

        # Check if task is completed
        if self.cycles_done >= self.cycles_to_run:
            self.get_logger().info(f"Task {self.current_task.get('job_ID')} completed in REALTIME mode.")
            if self.production_timer:
                self.production_timer.cancel()
                self.production_timer = None 
                # Publish completed status
                self.current_task['status'] = 'COMPLETED'
                self.publish_completed_status()
        
            if self.in_maintenance:
                self.maintenance_timer()
            return
        
        if self.control_cmd == "SHUTDOWN":
            self.get_logger().info(f"REALTIME simulation stopped due to SHUTDOWN command at cycle {self.cycles_done}.")
            if self.production_timer:
                self.production_timer.cancel()
                self.production_timer = None
                self.maintenance_timer()
            return

        # If not completed, run one cycle
        self.get_logger().info(f"Running cycle {self.cycles_done + 1}/{self.cycles_to_run} (Realtime)...")
        self.publish_generated_data()
        self.cycles_done += 1

    def run_fast_simulation(self):
        """
        Runs all production cycles as fast as possible in a separate thread (FAST mode).
            
        """
        self.get_logger().info(f"FAST simulation started for {self.cycles_to_run} cycles.")
        
        try:
            for _ in range(self.cycles_to_run):
                # In FAST mode, just loop and publish
                elapsed_minutes = float(self.cycles_done)
                elapsed_hours = elapsed_minutes / 60.0
                current_rul = max(0.0, self.max_lifetime - elapsed_hours)

                if current_rul <= 0:
                    self.get_logger().error(f"Machine failure during FAST sim! RUL=0 at cycle {self.cycles_done}.")
                    self.current_task['status'] = 'FAILED_DUE_TO_DEGRADATION'
                    #TODO: Publish failure status in due time at this line. 
                    break
                
                if self.control_cmd == "SHUTDOWN":
                    self.get_logger().info(f"FAST simulation stopped due to SHUTDOWN command at cycle {self.cycles_done}. remaining cycles: {self.cycles_to_run - self.cycles_done}.")
                    break

                self.publish_generated_data()
                self.cycles_done += 1
            
            if self.in_maintenance:
                if self.control_cmd == "SLOW_DOWN":
                    self.get_logger().info(f"FAST simulation stopped due to SLOW_DOWN command")
                self.maintenance_timer()
            
        except Exception as e:
            self.get_logger().error(f"Error during FAST simulation: {e}")
        
        finally:
            self.get_logger().info(f"Task {self.current_task.get('job_ID')} completed in FAST mode.")
            
            # Publish completed status
            self.current_task['status'] = 'COMPLETED'
            self.publish_completed_status()
            
    def publish_generated_data(self):
        """
        Publishes generated sensor data to Sensors topic.
        """
        # Cycle info in minutes
        cycle = self.cycles_done
        
        # Elapsed time in hours
        elapsed_hours = float(cycle) / 60.0
        
        # Ground Truth RUL
        gt_rul_hours = max(0.0, self.max_lifetime - elapsed_hours)
        
        sensor_data = self.generate_cycle(
                                            total_rul=self.total_lifetime_minutes,
                                            rng=self.RNG
                                        )
        
                
        sensor_data['cycle'] = cycle
        sensor_data['elapsed_hours'] = round(elapsed_hours, 6)
        sensor_data['elapsed_minutes'] = float(cycle)


        msg = String()
        msg.data = json.dumps(sensor_data)
        self.publisher_sensors.publish(msg)
            
        self.get_logger().info(
                                f"Published sensor data for cycle: {cycle} (in minutes) "
                            )
        self.get_logger().info(f"GroundTruth RUL={gt_rul_hours:.1f} hours ({(gt_rul_hours*60):.1f} min) after cycle {self.cycles_done}/{self.cycles_to_run}.")
            
            


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