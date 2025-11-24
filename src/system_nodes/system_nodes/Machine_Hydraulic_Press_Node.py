#!/usr/bin/env python3

"""
Listenes to Job Orders, and Maintenance Queue topics.
Publishes to Sensors and Completed topics.

Handles tasks for the Hydraulic Press machine.
"""

from std_msgs.msg import String
from rclpy.node import Node
import numpy as _np
import threading
import random
import rclpy
import json
import time
import math
import os


class Machine_Hydraulic_Press_Node(Node):
    def __init__(self):
        super().__init__('Machine_Hydraulic_Press_Node')
        
        # Subscription to Job Orders
        self.subscription_job_orders = self.create_subscription(String, 'Job_Orders', self.listener_job_orders_callback, 10)      
        # Subscription to Control_CMD
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD', self.listener_control_cmd_callback, 10)
              
        # Publisher for Sensors data
        self.publisher_sensors = self.create_publisher(String, "Sensors", 10)
        # Publisher for Completed status
        self.publisher_completed = self.create_publisher(String, "Completed", 10)
        
        # Variable set-up
        self.current_task = {}
        self.cycle_index = 0
        self.total_production = 0
        self.total_time_producing = 0.0

        # Simulation and production control flags
        # Set "REALTIME" for 60s per cycle, "FAST" for max speed
        self.simulation_mode = "FAST"      # or "REALTIME" TODO: Make this configurable in gui later  
        self.production_timer = None       # Holds the rclpy.Timer object
        self.cycles_to_run = 0
        self.cycles_done = 0

        # Defaults and influence maps
        self.material_factor = {
                                "DC01_ZE": 1.00,
                                "Stainless_304": 1.20,
                                "Aluminium_6082_T6": 0.90,
                               }
        self.process_factors = {
                                "bending": 1.00,
                                "forming": 1.10,
                                "drilling": 0.70,
                                "grooving": 0.75,
                                "pocketing": 0.80,
                                "assembling": 0.50,
                                "quality_control": 0.30,
                               }
        self.PRESS_MAP = {
                            "bending": {
                                        "DC01_ZE": {
                                                    "speed_mm_s": (80, 120),
                                                    "dwell_s": (0.0, 0.2),
                                                    "baseline_tonnage_band_t": (20, 35)  # 2 mm, 1 m width, V=16 mm
                                                    },
                                        "Stainless_304": {
                                                    "speed_mm_s": (60, 90),
                                                    "dwell_s": (0.1, 0.3),
                                                    "baseline_tonnage_band_t": (28, 48)
                                                    },
                                    "Aluminium_6082_T6": {
                                                    "speed_mm_s": (100, 140),
                                                    "dwell_s": (0.0, 0.2),
                                                    "baseline_tonnage_band_t": (12, 22)
                                                    },
                                        },
                            "forming": {
                                        "DC01_ZE": {
                                                    "speed_mm_s": (40, 80),
                                                    "dwell_s": (0.3, 0.8),
                                                    "baseline_tonnage_band_t": (25, 40)
                                                    },
                                        "Stainless_304": {
                                                    "speed_mm_s": (30, 60),
                                                    "dwell_s": (0.5, 1.0),
                                                    "baseline_tonnage_band_t": (35, 55)
                                                    },
                                    "Aluminium_6082_T6": {
                                                    "speed_mm_s": (60, 100),
                                                    "dwell_s": (0.2, 0.6),
                                                    "baseline_tonnage_band_t": (15, 25)
                                                    },
                                        }
                            }



        # Set-up logs
        self.get_logger().info("Machine Hydraulic Press node ready!")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Control_CMD' topic.")
        self.get_logger().info("Listening on 'Completed' topic.")



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
                                                f"\nReceived task: '{task['job_ID']}'. "
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

    # TODO: Implement actual processing logic here in due time
    def listener_control_cmd_callback(self, msg: String):
        """
        Callback function for Control_CMD subscription.
        """
        self.get_logger().info(f"Received Control_CMD message: {msg.data}")

    def calculate_tonnage(self):
        """
        Calculates estimated tonnage based on current task's material, process, thickness, and width.
        Scaling is done relative to the 2 mm / 1000 mm reference in PRESS_MAP
        """
        
        material = self.current_task['material']
        process = self.current_task['process']
        thickness_mm = float(self.current_task['part_thickness_mm'])
        width_mm = float(self.current_task['part_width_mm'])

        if thickness_mm <= 0:
            self.get_logger().warn("Invalid thickness value.")
            return 0.0

        # Get baseline tonnage band from PRESS_MAP
        press_proc = self.PRESS_MAP.get(process)
        
        if not press_proc:
            self.get_logger().error(f"Process '{process}' not found in PRESS_MAP.")
            return 30.0 # default average value
        
            
        mat_entry = press_proc.get(material)
        if mat_entry:
            base_min, base_max = mat_entry["baseline_tonnage_band_t"]
            base_avg = (base_min + base_max) / 2.0
        else:
            self.get_logger().warn(f"Material '{material}' not found for process '{process}'. Using default tonnage.")
            base_avg = 30.0  # default average value

        # Effect of thickness based on process type (approximation)
        if process == "bending":
            thickness_exp = 1.0
        elif process == "forming":
            thickness_exp = 1.3
        else:
            thickness_exp = 1.0

        material_fac = self.material_factor.get(material, 1.0)
        process_fac = self.process_factors.get(process, 1.0)

        # 2 mm reference for thickness scaling
        thickness_scale = (thickness_mm / 2.0) ** thickness_exp
        # 1000 mm reference for width scaling
        width_scale = width_mm / 1000.0

        tonnage = base_avg * thickness_scale * width_scale * material_fac * process_fac
        tonnage = max(0.1, tonnage)

        return tonnage
    
    def calulate_production_rate(self):
        
        self.total_production += self.current_task['produce_amount']
        self.total_time_producing += self.current_task['task_time']
        production_rate_per_hour = (self.total_production / self.total_time_producing) * 3600.0
        return production_rate_per_hour

    def _get_degradation_state(self):
        """Return dataset-like health parameters for the current cycle."""
        idx = self.cycle_index
        state = {
                    "health": 1.0,
                    "cooler_eff": 100,
                    "valve_cond": 100,
                    "pump_leak": 0.0,
                    "acc_pressure": 130,
                    "internal_leak": 0.0
                }

        if idx < 1800:
            pass
        elif idx < 2000:
            frac = (idx - 1800) / 200
            state["health"] = 1.0 - 0.3 * frac
            state["cooler_eff"] = 100 - 60 * frac
        elif idx < 2100:
            frac = (idx - 2000) / 100
            state["health"] = 0.7 - 0.4 * frac
            state["cooler_eff"] = 40
            state["valve_cond"] = 100 - 27 * frac
            state["pump_leak"] = 0.02 * (idx - 2000)
        elif idx < 2150:
            frac = (idx - 2100) / 50
            state["health"] = 0.3 - 0.2 * frac
            state["cooler_eff"] = 40
            state["valve_cond"] = 73
            state["pump_leak"] = 2 + 0.02 * (idx - 2100)
            state["acc_pressure"] = 130 - 40 * frac
            state["internal_leak"] = min(1.0, 0.02 * (idx - 2100))
        else:
            state["health"] = max(0.05, 0.1 - 0.05 * (idx - 2150) / 100)
            state["cooler_eff"] = 40
            state["valve_cond"] = 73
            state["pump_leak"] = 3.0
            state["acc_pressure"] = 90
            state["internal_leak"] = 1.0
        return state

    def _flags_from_state(self, state):
        """Translate degradation state into UCI profile flags."""
        cooler = 100 if state["cooler_eff"] >= 90 else 40 if state["cooler_eff"] >= 40 else 3
        valve = 100 if state["valve_cond"] >= 90 else 73
        pump = 0
        if state["pump_leak"] >= 3:
            pump = 3
        elif state["pump_leak"] >= 2:
            pump = 2
        elif state["pump_leak"] >= 1:
            pump = 1
        accumulator = 130 if state["acc_pressure"] >= 120 else 115 if state["acc_pressure"] >= 100 else 90
        internal_leak = 0 if state["internal_leak"] < 0.5 else 1
        return {
                    "cooler": cooler,
                    "valve": valve,
                    "pump": pump,
                    "accumulator": accumulator,
                    "stable": 1,
                    "internal_leakage": internal_leak
                }

    def generate_sensor_data(self):
        """Create UCI-compatible raw signals for one 60 s cycle."""
        lengths = {"100hz": 6000, "10hz": 600, "1hz": 60}
        state = self._get_degradation_state()
        tonnage = self.calculate_tonnage()
        load_factor = min(2.5, tonnage / 40.0)
        t100 = _np.linspace(0, 60, lengths["100hz"])
        t10 = _np.linspace(0, 60, lengths["10hz"])
        t1 = _np.linspace(0, 60, lengths["1hz"])
        pulse = _np.exp(-((t100 - 30) ** 2) / (2 * 3 ** 2))

        # Pressures PS1-PS6
        ps_nominals = [160, 155, 158, 152, 150, 148]
        ps_data = {}
        for idx, base in enumerate(ps_nominals, start=1):
            signal = base + 70 * load_factor * pulse
            signal *= state["health"]
            signal += _np.random.normal(0, 2.5, lengths["100hz"])
            if state["internal_leak"] > 0.5:
                signal *= 0.85
            ps_data[f"PS{idx}"] = _np.clip(signal, 0, 350).tolist()

        # EPS1
        eps_base = 3000 + 2000 * load_factor
        eps_signal = eps_base * (1 + 0.35 * pulse) * state["health"]
        if state["pump_leak"] > 1.5:
            eps_signal *= 1.1
        eps_signal += _np.random.normal(0, 40, lengths["100hz"])
        EPS1 = _np.clip(eps_signal, 500, 12000).tolist()

        # FS1, FS2
        FS1 = (11 + 2.2 * load_factor + _np.random.normal(0, 0.25, lengths["10hz"]))
        FS2 = (9 + 1.7 * load_factor + _np.random.normal(0, 0.25, lengths["10hz"]))
        if state["internal_leak"] > 0.5:
            FS1 *= 0.8
            FS2 *= 0.8

        # Temperatures TS1-TS4
        base_temp = 38 + 12 * (1 - state["cooler_eff"] / 100) + 6 * load_factor
        offsets = [-1.5, 0.5, 1.5, -0.5]
        TS = {}
        for i in range(4):
            TS[f"TS{i+1}"] = (
                                 base_temp + offsets[i] + _np.random.normal(0, 0.6, lengths["1hz"])
                            ).tolist()

        # VS1
        VS1 = (
                    0.5 + 0.7 * load_factor + 0.4 * (1 - state["health"])
                    + _np.random.normal(0, 0.05, lengths["1hz"])
               ).tolist()

        # CE, CP, SE
        CE = (
                    100 - 0.45 * (base_temp - 38) - 12 * (1 - state["health"])
                    + _np.random.normal(0, 1.5, lengths["1hz"])
            ).tolist()
        
        CP = (
                2.0 + 0.4 * state["pump_leak"] + _np.random.normal(0, 0.08, lengths["1hz"])
            ).tolist()
        
        SE = (
                90 - 18 * (1 - state["health"]) + _np.random.normal(0, 2.5, lengths["1hz"])
            ).tolist()

        flags = self._flags_from_state(state)
        self.cycle_index += 1

        return {
                    "cycle": self.cycle_index - 1,
                    "load_factor": round(float(load_factor), 3),
                    "tonnage_est": round(float(tonnage), 2),
                    "PS1": ps_data["PS1"],
                    "PS2": ps_data["PS2"],
                    "PS3": ps_data["PS3"],
                    "PS4": ps_data["PS4"],
                    "PS5": ps_data["PS5"],
                    "PS6": ps_data["PS6"],
                    "EPS1": EPS1,
                    "FS1": FS1.tolist(),
                    "FS2": FS2.tolist(),
                    "TS1": TS["TS1"],
                    "TS2": TS["TS2"],
                    "TS3": TS["TS3"],
                    "TS4": TS["TS4"],
                    "VS1": VS1,
                    "CE": CE,
                    "CP": CP,
                    "SE": SE,
                    "profile": flags
                }
    
    def publish_generated_data(self):
        """
        Publishes generated sensor data to Sensors topic.
        """
        sensor_data = self.generate_sensor_data()
        sensor_msg = String()
        sensor_msg.data = json.dumps(sensor_data)
        self.publisher_sensors.publish(sensor_msg)
        self.get_logger().info(f"Published sensor data for cycle {self.cycle_index - 1}.")

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
        # Check if task is completed
        if self.cycles_done >= self.cycles_to_run:
            self.get_logger().info(f"Task {self.current_task.get('job_ID')} completed in REALTIME mode.")
            if self.production_timer:
                self.production_timer.cancel()
                self.production_timer = None
            
            # Publish completed status
            self.current_task['status'] = 'COMPLETED'
            self.publish_completed_status()
            
            
            return

        # If not completed, run one cycle
        self.get_logger().info(f"Running cycle {self.cycles_done + 1}/{self.cycles_to_run} (Realtime)...")
        self.publish_generated_data()                                                                        # This publishes data and increments self.cycle_index
        self.cycles_done += 1

    def run_fast_simulation(self):
        """
        Runs all production cycles as fast as possible in a separate thread (FAST mode).
        """
        self.get_logger().info(f"FAST simulation started for {self.cycles_to_run} cycles.")
        
        try:
            for i in range(self.cycles_to_run):
                # In FAST mode, just loop and publish
                self.publish_generated_data()
                self.cycles_done += 1
                
                # Log progress periodically to avoid spamming the console
                if (self.cycles_done % 50 == 0) or (self.cycles_done == self.cycles_to_run):
                     self.get_logger().info(f"FAST sim progress: {self.cycles_done}/{self.cycles_to_run} cycles...")
        
        except Exception as e:
            self.get_logger().error(f"Error during FAST simulation: {e}")
        
        finally:
            self.get_logger().info(f"Task {self.current_task.get('job_ID')} completed in FAST mode.")
            
            
            # Publish completed status
            self.current_task['status'] = 'COMPLETED'
            self.publish_completed_status()
            
            
            


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