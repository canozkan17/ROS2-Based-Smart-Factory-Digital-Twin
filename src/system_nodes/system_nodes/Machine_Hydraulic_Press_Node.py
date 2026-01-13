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

        # Simulation and production control flags
        # Set "REALTIME" for 60s per cycle, "FAST" for max speed
        self.simulation_mode = "FAST"      # or "REALTIME" TODO: Make this configurable in gui later  
        self.production_timer = None       # Holds the rclpy.Timer object
        self.cycles_to_run = 0
        self.cycles_done = 0

        self._state_lock = threading.Lock()
        self.in_maintenance = False
        self.control_cmd = "NORMAL_OPERATION"

        # Defaults
        self.SEED = 42
        self.degredation_factor = 1.0
        
        # Initialize RNG with fixed seed for reproducibility
        rng = np.random.default_rng(self.SEED)
        self.RNG = rng
        
        # total_rul for this machine lifecycle
        self.total_lifetime_minutes = int(rng.integers(120_000, 720_000))
        self.max_lifetime = self.total_lifetime_minutes / 60.0  # convert to hours

        # HYDRAULIC PRESS SENSOR PARAMETERS
        # MUST MATCH generate_hydraulic_data.py EXACTLY for training consistency!

        # Degradation profile (matches generate_hydraulic_data.py)
        self.rul_for_near_term = 5000.0
        self.rul_for_critical = 600.0
        
        # Hydraulic pressure (bar) - seal wear dominant
        self.base_pressure = rng.uniform(180.0, 220.0)
        self.pressure_drop_rate = rng.uniform(0.00005, 0.00015)
        self.pressure_noise = rng.uniform(0.3, 1.0)

        # Oil temperature (°C) - oil degradation
        self.base_oil_temp = rng.uniform(45.0, 65.0)
        self.oil_temp_rate = rng.uniform(0.0003, 0.0010)
        self.oil_temp_noise = rng.uniform(0.08, 0.3)

        # Oil contamination index (dimensionless)
        self.base_contamination = rng.uniform(0.5, 2.0)
        self.contamination_growth = rng.uniform(0.00003, 0.00010)
        self.contamination_noise = rng.uniform(0.008, 0.025)

        # Ram position deviation (mm) - misalignment
        self.base_ram_dev = rng.uniform(0.01, 0.05)
        self.ram_dev_growth = rng.uniform(0.00001, 0.00006)
        self.ram_dev_noise = rng.uniform(0.0008, 0.002)

        # Press force / tonnage (tons)
        self.base_force = rng.uniform(80.0, 120.0)
        self.force_loss_rate = rng.uniform(0.00003, 0.00012)
        self.force_noise = rng.uniform(0.2, 0.8)

        # Frame / ram vibration (mm/s)
        self.base_vibration = rng.uniform(0.1, 0.5)
        self.failure_vibration = rng.uniform(6.0, 12.0)
        self.vibration_noise = rng.uniform(0.015, 0.05)

        # Hydraulic flow rate (L/min)
        self.base_flow = rng.uniform(90.0, 130.0)
        self.flow_loss_rate = rng.uniform(0.00004, 0.00014)
        self.flow_noise = rng.uniform(0.2, 0.6)

        # Motor current (A)
        self.base_current = rng.uniform(30.0, 55.0)
        self.current_growth_rate = rng.uniform(0.00006, 0.00020)
        self.current_noise = rng.uniform(0.08, 0.3)

        # Set-up logs
        self.get_logger().info("Machine Hydraulic Press node ready!")
        self.get_logger().info(f"Machine lifecycle: max lifetime = {self.max_lifetime:.1f} hours ({self.total_lifetime_minutes} min)")
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
                    self.cycles_to_run = task['task_time_min']  # in minutes
                    
                    if self.cycles_to_run <= 0:
                        self.get_logger().warn("Task time is zero or negative. Skipping task.")
                        self.current_task = {}
                        self.get_logger().error(f"Invalid or missing 'task_time_min' in job. Cannot start.")
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


    def listener_control_cmd_callback(self, msg: String):
        """
        Callback function for Control CMD subscription.
        Processes control commands for corrective/preventative actions.
        """
        control_msg = json.loads(msg.data)
        self.get_logger().info(f"Received Control Command")

        # Extract control command and recovery time from the message
        command = control_msg.get("command", "NORMAL_OPERATION")
        machine = control_msg.get("machine", "hydraulic_press")

        with self._state_lock:
            self.maintenance_cycles_remaining = control_msg.get("recovery_time_min", 0)

            if command == "SHUTDOWN":
                self.get_logger().info(f"Received SHUTDOWN command for machine {machine}. Stopping current task.")
                self.control_cmd = "SHUTDOWN"
                self.in_maintenance = True
                
            elif command == "SLOW_DOWN":
                self.control_cmd = "SLOW_DOWN"
                self.get_logger().info(f"Received SLOW_DOWN command for machine {machine}. Slowing down operation.")
                self.degredation_factor *= 0.9  # slowing down degradation by 10%
            
            
            elif command == "NORMAL_OPERATION":
                self.get_logger().info(f"Received NORMAL_OPERATION command. Continuing operation.")
            else:
                self.get_logger().warning(f"Unknown command received: {command}")
        
        if self.production_timer:
            self.production_timer.cancel()
            self.production_timer = None

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
            self.get_logger().info(
                                        f"FAST maintenance started for {self.maintenance_cycles_remaining} cycles"
                                    )
            while self.maintenance_cycles_remaining > 0:
                self.maintenance_cycles_remaining -= 1
                time.sleep(0.01)  # small sleep to simulate time passage
            self.finish_maintenance()
    
    def finish_maintenance(self):
        """
        Complete maintenance - reset cycles_done to simulate "like-new" state.
        Sensor parameters stay the same (they don't "wear out" - degradation is based on cycles_done/total_rul fraction).
        """
        old_cycles = self.cycles_done
        
        self.degredation_factor = 1.0
        self.in_maintenance = False
        self.maintenance_cycles_remaining = 0

        self.cycles_done = 0
        self.control_cmd = "NORMAL_OPERATION"
        self.current_task = {}

        # MAINTENANCE_FEEDBACK
        msg = String()
        msg.data = json.dumps({
                                    "machine": "hydraulic_press",
                                    "status": "READY"
                                })
        
        self.publisher_maintenance_feedback.publish(msg) 

        self.get_logger().info(f"[TEMP:DEBUG] MAINTENANCE_COMPLETE: cycles_done {old_cycles} -> 0, total_lifetime_minutes={self.total_lifetime_minutes}")
        self.get_logger().info(f"Maintenance completed. Machine READY. Degradation reset. Cycles done reset. Machine total ran cycles: {self.total_ran_cycles} minutes.")


    def generate_cycle(self, total_rul: int, rng: np.random.Generator):
        """
        Generate sensor data for current cycle.
        MUST MATCH generate_hydraulic_data.py EXACTLY for training consistency!
        """
        t = float(self.cycles_done)
        current_rul = total_rul - t - 1
        fraction = t / total_rul if total_rul > 0 else 0.0

        # Degradation calculations
        base_degradation = fraction

        relative_near = max((self.rul_for_near_term - current_rul) / self.rul_for_near_term, 0.0)
        near_term_factor = 1.0 + (relative_near ** 1.5) * 2.0

        relative_critical = max((self.rul_for_critical - current_rul) / self.rul_for_critical, 0.0)
        critical_boost = 1.0 + (relative_critical ** 2) * 5.0

        # Combined degradation factor (multiplicative)
        degradation_factor = base_degradation * near_term_factor * critical_boost

        # Noise multiplier (heteroscedastic)
        noise_multiplier = 1.0 + degradation_factor

        # Physical limits for realistic sensor values
        max_oil_temperature = 120.0
        min_hydraulic_pressure = 50.0
        max_oil_contamination = 50.0
        max_vibration = 15.0
        min_press_force = 20.0
        min_flow_rate = 20.0
        max_motor_current = 100.0

        hydraulic_pressure = (
                                self.base_pressure
                                - self.pressure_drop_rate * t * near_term_factor * critical_boost
                                + rng.normal(0.0, self.pressure_noise) * noise_multiplier
                            )
        hydraulic_pressure = max(hydraulic_pressure, min_hydraulic_pressure)

        oil_temperature = (
                            self.base_oil_temp
                            + self.oil_temp_rate * t * near_term_factor * critical_boost
                            + rng.normal(0.0, self.oil_temp_noise) * (1.0 + 0.5 * degradation_factor)
                        )
        oil_temperature = min(oil_temperature, max_oil_temperature)

        oil_contamination = (
                                self.base_contamination
                                + self.contamination_growth * t * near_term_factor * critical_boost
                                + rng.normal(0.0, self.contamination_noise) * noise_multiplier
                            )
        oil_contamination = min(oil_contamination, max_oil_contamination)

        ram_position_deviation = (
                                    self.base_ram_dev
                                    + self.ram_dev_growth * t * near_term_factor * critical_boost
                                    + rng.normal(0.0, self.ram_dev_noise) * noise_multiplier
                                )

        press_force = (
                            self.base_force
                            - self.force_loss_rate * t * near_term_factor * critical_boost
                            + rng.normal(0.0, self.force_noise) * (1.0 + 0.3 * degradation_factor)
                        )
        press_force = max(press_force, min_press_force)

        vibration_profile = (self.failure_vibration - self.base_vibration) * (
                                                                                0.1 * fraction +
                                                                                0.3 * (relative_near ** 2) +
                                                                                0.6 * (relative_critical ** 1.5)
                                                                            )
        vibration = (
                        self.base_vibration
                        + vibration_profile
                        + rng.normal(0.0, self.vibration_noise) * noise_multiplier
                    )
        vibration = min(vibration, max_vibration)

        flow_rate = (
                        self.base_flow
                        - self.flow_loss_rate * t * near_term_factor * critical_boost
                        + rng.normal(0.0, self.flow_noise) * noise_multiplier
                    )
        flow_rate = max(flow_rate, min_flow_rate)

        motor_current = (
                            self.base_current
                            + self.current_growth_rate * t * near_term_factor * critical_boost
                            + rng.normal(0.0, self.current_noise) * (1.0 + 0.5 * degradation_factor)
                        )
        motor_current = min(motor_current, max_motor_current)

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
                
                with self._state_lock:
                    if self.in_maintenance:
                    
                        if self.control_cmd == "SHUTDOWN":
                            self.get_logger().info(f"FAST simulation stopped due to SHUTDOWN command at cycle {self.cycles_done}. remaining cycles: {self.cycles_to_run - self.cycles_done}.")
                            break
                        
                        elif self.control_cmd == "SLOW_DOWN":
                            self.get_logger().info(f"FAST simulation slowed due to SLOW_DOWN command")

                self.publish_generated_data()
                self.cycles_done += 1
            
            if self.in_maintenance:
                self.maintenance_timer()
            
        except Exception as e:
            self.get_logger().error(f"Error during FAST simulation: {e}")
        
        finally:

            if self.cycles_done == self.cycles_to_run:
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
        
        # Calculate fraction for debug
        fraction = float(cycle) / self.total_lifetime_minutes if self.total_lifetime_minutes > 0 else 0
        current_rul_min = self.total_lifetime_minutes - cycle
        
        # Debug log for key sensor values
        self.get_logger().info(
            f"[TEMP:DEBUG] SENSOR cycle={cycle}, fraction={fraction:.6f}, current_rul={current_rul_min}min, "
            f"vibration={sensor_data['vibration']:.4f}, pressure={sensor_data['hydraulic_pressure']:.2f}, "
            f"oil_temp={sensor_data['oil_temperature']:.2f}, contamination={sensor_data['oil_contamination']:.4f}"
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