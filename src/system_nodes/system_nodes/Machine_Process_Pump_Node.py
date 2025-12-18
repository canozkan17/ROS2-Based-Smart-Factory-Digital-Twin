#!/usr/bin/env python3

"""
ROS2 Node for generating synthetic sensor data for Process Pump Machine triggered by machine_hydraulic_press_sensor_node's Job Orders.

Subscribes to Job_Orders topic for data generation - listens to hydraulic_press signal. 
Subscribes to Control_CMD topic for corrective/preventative actions - listens to hydraulic_press signal.

Publishes generated data to Sensors topic.
DOES NOT publish status to Completed Topic - as this is a helper machine. 
"""

from std_msgs.msg import String
from rclpy.node import Node
from typing import Dict
import numpy as np
import threading
import logging
import random
import rclpy
import json
import math
import os

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
        self.publisher_sensors = self.create_publisher(String, "Sensors/process_pump", 10)
        
        # Loading Patterns for Sensor Data Generation
        # REMOVED PATTERNS
        base_dir = os.path.dirname(os.path.abspath(__file__))
        deg_path = os.path.normpath(os.path.join(base_dir, "../../..", "pump_sensor_rul_hrs_data", "degradation_patterns.json"))
        noise_path = os.path.normpath(os.path.join(base_dir, "../../..", "pump_sensor_rul_hrs_data", "noise_characteristics.json"))
        
        with open(deg_path, 'r') as f:
            self.degradation_patterns = json.load(f)
        with open(noise_path, 'r') as f:
            self.noise_chars = json.load(f)
            
        self.get_logger().info("Real degradation and noise patterns loaded.")

        # Variable set-up
        self.total_ran_cycles = 0 # real cycle count in minutes
        self.current_task = {}

        # THIS MIGHT BE MINUTES
        # random lifetime (500-800 hrs, based on training dataset)
        self.max_lifetime = random.randint(500, 800)  # in hours
        self.get_logger().info(f"Process_Pump initialized with max lifetime: {self.max_lifetime} hours (ground truth, hidden)")

        # Simulation and production control flags
        # Set "REALTIME" for 60s per cycle, "FAST" for max speed
        self.simulation_mode = "FAST"      # or "REALTIME" TODO: Make this configurable in gui later  
        self.production_timer = None       # Holds the rclpy.Timer object
        self.cycles_to_run = 0
        self.cycles_done = 0

        # Defaults
        np.random.seed(42)
        random.seed(42) 

        self.get_logger().info("Process Pump Sensor node ready!")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Control_CMD' topic.")

    
    # Callback functions
    #--------------------
    # Callback for Job Orders
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

    #--------------------
    # Callback for Control CMD
    def listener_callback_control_cmd(self, msg: String):
        """
        Callback function for Control CMD subscription.
        Processes control commands for corrective/preventative actions.
        """
        self.get_logger().info(f"Received Control Command: {json.dumps(msg.data, indent=2)}")#TODO: Implement control cmd in due time.&& check sent data format.

    def generate_sensor_data(self):
        """
        Generates one timestep of realistic synthetic sensor data for the pump.
        """
        sensor_data = {}

        # Calculate elapsed time in hours
        elapsed_minutes = float(self.total_ran_cycles)
        elapsed_hours = elapsed_minutes / 60.0

        t = min(1.0, elapsed_hours / float(self.max_lifetime))  # normalized time [0,1]

        for sensor_name, pattern in self.degradation_patterns.items():
            
            healthy_mean = pattern['healthy_mean']
            eol_mean = pattern.get('eol_mean', healthy_mean)

            trend = pattern.get('trend', 'increasing')
            
            rate_per_hour = float(pattern.get('degradation_rate_per_hour', 0.0))

            # noise characteristics
            noise_char = self.noise_chars.get(sensor_name, {})
            healthy_noise_std = float(noise_char.get('healthy_noise_std', max(0.01 * abs(healthy_mean), 0.001)))
            noise_ratio = noise_char.get('noise_increase_ratio', 1.0)
            noise_ratio = float(noise_ratio)

            delta = eol_mean - healthy_mean
            delta_sign = 0.0
            if abs(delta) > 0.000001:
                delta_sign = np.sign(delta)
                # if conflict: force to mean based direction
                if (trend == 'increasing' and delta_sign < 0) or (trend == 'decreasing' and delta_sign > 0):
                    trend = 'increasing' if delta_sign > 0 else 'decreasing'

            # training-derived rate_per_hour
            if rate_per_hour != 0.0:
                direction_sign = 1.0 if trend == 'increasing' else -1.0
                base_value = healthy_mean + (direction_sign * abs(rate_per_hour) * elapsed_hours)
                # to avoid overshooting beyond EOL mean
                if trend == 'increasing':
                    base_value = float(np.clip(base_value, min(healthy_mean, eol_mean), max(healthy_mean, eol_mean)))
                else:
                    base_value = float(np.clip(base_value, min(eol_mean, healthy_mean), max(eol_mean, healthy_mean)))
            else: # fallback to linear interpolation healthy->EOL over assigned lifetime
                delta = eol_mean - healthy_mean
                effective_rate_per_hour = delta / float(self.max_lifetime)
                base_value = healthy_mean + (effective_rate_per_hour * elapsed_hours)


            # Noise mimics real behaviour
            current_noise_std = healthy_noise_std * (1.0 + (noise_ratio - 1.0) * t)
            noise = np.random.normal(0, max(current_noise_std / 5.0, 0.0001))

            value = base_value + noise
            sensor_data[sensor_name] = round(float(value),6)

        self.total_ran_cycles += 1
        return sensor_data
    
    def run_production_cycle(self):
        """
        Callback for the rclpy.Timer (REALTIME mode).
        Runs one production cycle every 60 seconds.
        """
        elapsed_minutes = float(self.total_ran_cycles)
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
                elapsed_minutes = float(self.total_ran_cycles)
                elapsed_hours = elapsed_minutes / 60.0
                current_rul = max(0.0, self.max_lifetime - elapsed_hours)

                if current_rul <= 0:
                    self.get_logger().error(f"Machine failure during FAST sim! RUL=0 at cycle {self.total_ran_cycles}.")
                    self.current_task['status'] = 'FAILED_DUE_TO_DEGRADATION'
                    #TODO: Publish failure status in due time at this line. 
                    break
                
                self.publish_generated_data()
                self.cycles_done += 1
                        
        except Exception as e:
            self.get_logger().error(f"Error during FAST simulation: {e}")
        
        finally:
            self.get_logger().info(f"Task {self.current_task.get('job_ID')} completed in FAST mode.")
    
    def publish_generated_data(self):
        """
        Publishes generated sensor data to Sensors topic.
        Ensures elapsed_hours and age_ratio match training format.
        """
        sensor_data = self.generate_sensor_data()
        
        # Cycle info in minutes
        cycle = self.total_ran_cycles  
        
        # Elapsed time in hours
        elapsed_hours = float(cycle) / 60.0
        
        # age_ratio as in training: elapsed_hours / max_lifetime
        # max_lifetime is already in hours
        age_ratio = min(1.0, elapsed_hours / float(self.max_lifetime))
        
        sensor_data['cycle'] = cycle
        sensor_data['elapsed_hours'] = round(elapsed_hours, 6)
        sensor_data['elapsed_minutes'] = float(cycle)
        sensor_data['age_ratio'] = round(age_ratio, 6)
        
        msg = String()
        msg.data = json.dumps(sensor_data)
        self.publisher_sensors.publish(msg)
        
        self.get_logger().info(
                                f"Published sensor data for cycle {cycle} (minute), "
                                f"elapsed={elapsed_hours:.2f}h, age_ratio={age_ratio:.4f}."
                            )
        
        # Ground Truth RUL
        gt_rul_hours = max(0.0, self.max_lifetime - elapsed_hours)
        self.get_logger().info(f"GroundTruth RUL={gt_rul_hours:.1f} hours after cycle {self.total_ran_cycles - 1}.")


def main(args=None):
    """
    Main entry point for the machine_process_pump_sensor_node.
    """
    rclpy.init(args=args)
    node = Machine_Process_Pump_Sensor_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()