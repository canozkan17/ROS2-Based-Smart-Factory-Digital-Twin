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
        
        self.SEED = 42
        self.RNG = np.random.default_rng(seed=self.SEED)
            
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

        # Sensor Parameters
        rng = np.random.default_rng(42)
        self.base_vib = rng.uniform(0.02, 0.3)
        self.failure_vib = rng.uniform(2.0, 6.0)
        self.vib_noise_scale = rng.uniform(0.01, 0.1)

        self.base_temp = rng.uniform(40.0, 60.0)
        self.k_temp = rng.uniform(0.005, 0.02)
        self.temp_noise_scale = rng.uniform(0.05, 0.3)

        self.base_pressure = rng.uniform(7.6, 8.4)
        self.k_pressure = rng.uniform(0.0005, 0.003)
        self.pressure_noise_scale = rng.uniform(0.01, 0.08)

        self.coupling_factor = rng.uniform(0.4, 0.9)
        self.vib_motor_noise = rng.uniform(0.005, 0.05)


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
                    self.cycles_to_run = task['task_time']
                    
                    if self.cycles_to_run <= 0:
                        self.get_logger().warn("Task time is zero or negative. Skipping task.")
                        self.current_task = {}
                        self.get_logger().error(f"Invalid or missing 'task_time' in job. Cannot start.")
                        return

                    self.get_logger().info(
                                                f"\nReceived task: '{json.dumps(task['job_ID'], indent=2)}'. "
                                                f"Calculated cycles: {self.cycles_to_run}"
                                            )

                    # Start production loop based on simulation mode
                    if self.simulation_mode == "REALTIME":
                        self.get_logger().info(f"Starting REALTIME simulation ({self.cycles_to_run} cycles)...")
                        # Create a non-blocking timer that calls the function every minute
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

    def generate_cycle(self, total_rul: int, rng: np.random.Generator):
        """
        Generate synthetic sensor data for a single pump cycle until failure.
        """

        t = float(self.total_ran_cycles)
        current_rul = total_rul - t - 1
        fraction = t / total_rul

        # Critical region (last 100 minutes)
        critical_mask = current_rul <= 100
        critical_boost = np.where(
                                    critical_mask,
                                    1.0 + (100 - current_rul) / 100 * 0.5,
                                    1.0
                                )
        vibration = (
                        self.base_vib
                        + (self.failure_vib - self.base_vib) * (fraction ** 2) * critical_boost
                        + rng.normal(0.0, self.vib_noise_scale * (1.0 + fraction), size=1)
                    )[0]
        temp_motor = (
                        self.base_temp + self.k_temp * t * critical_boost
                        + rng.normal(0.0, self.temp_noise_scale, size=1)
                    )[0]
        pressure = (
                        self.base_pressure - self.k_pressure * t * critical_boost
                        + rng.normal(0.0, self.pressure_noise_scale * (1.0 + 0.5 * fraction), size=1)
                    )[0]
        vib_motor = (
                        vibration * self.coupling_factor
                        + rng.normal(0.0, self.vib_motor_noise * (1.0 + fraction), size=1)
                    )[0]
        cycle_output = {
                            "vibration": float(vibration),
                            "temp_motor": float(temp_motor),
                            "pressure": float(pressure),
                            "vib_motor": float(vib_motor),
                        }
        self.total_ran_cycles += 1
        return cycle_output
    
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
        """

        # Cycle info in minutes
        cycle = self.total_ran_cycles  
        
        # Elapsed time in hours
        elapsed_hours = float(cycle) / 60.0
        
        # Ground Truth RUL
        gt_rul_hours = max(0.0, self.max_lifetime - elapsed_hours)
        gt_rul_minutes = max(0.0, (self.max_lifetime * 60) - cycle)
        
        sensor_data = self.generate_cycle(total_rul= int(gt_rul_minutes), rng=self.RNG)
        
                
        sensor_data['cycle'] = cycle
        sensor_data['elapsed_hours'] = round(elapsed_hours, 6)
        sensor_data['elapsed_minutes'] = float(cycle)
        
        # REMOVE AFTER DONE
        print(sensor_data)
        # =======================

        msg = String()
        msg.data = json.dumps(sensor_data)
        self.publisher_sensors.publish(msg)
        
        self.get_logger().info(
                                f"Published sensor data for cycle: {cycle} (in minutes) "
                            )
        self.get_logger().info(f"GroundTruth RUL={gt_rul_hours:.1f} hours after cycle {self.total_ran_cycles}.")


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

