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
        
        # Variable set-up
        self.total_ran_cycles = 0
        self.current_task = {}

        # random lifetime (700-837 cycles, based on training dataset)
        self.max_lifetime = random.randint(700, 837)  # in cycles
        self.get_logger().info(f"Process_Pump initialized with max lifetime: {self.max_lifetime} cycles (ground truth, hidden)")

        # Simulation and production control flags
        # Set "REALTIME" for 60s per cycle, "FAST" for max speed
        self.simulation_mode = "FAST"      # or "REALTIME" TODO: Make this configurable in gui later  
        self.production_timer = None       # Holds the rclpy.Timer object
        self.cycles_to_run = 0
        self.cycles_done = 0

        # Defaults
            # Statistics from original dataset (mean ± std)
            # Nominal values for sensors under normal operation (high RUL)
        self.nominal_means = {
                                'sensor_00': 2.349,   'sensor_01': 47.017,  'sensor_02': 50.628,  'sensor_03': 43.338,
                                'sensor_04': 577.716, 'sensor_05': 74.266,  'sensor_06': 13.097,  'sensor_07': 15.623,
                                'sensor_08': 15.007,  'sensor_09': 14.660,  'sensor_10': 40.358,  'sensor_11': 39.448,
                                'sensor_12': 27.667,  'sensor_13': 4.731,   'sensor_14': 363.892, 'sensor_15': 0.0,
                                'sensor_16': 402.396, 'sensor_17': 408.303, 'sensor_18': 2.232,   'sensor_19': 567.890,
                                'sensor_20': 349.156, 'sensor_21': 770.537, 'sensor_22': 436.958, 'sensor_23': 870.379,
                                'sensor_24': 535.851, 'sensor_25': 625.486, 'sensor_26': 760.167, 'sensor_27': 476.982,
                                'sensor_28': 878.081, 'sensor_29': 589.398, 'sensor_30': 586.181, 'sensor_31': 853.608,
                                'sensor_32': 781.529, 'sensor_33': 479.250, 'sensor_34': 225.392, 'sensor_35': 391.479,
                                'sensor_36': 531.891, 'sensor_37': 74.765,  'sensor_38': 49.387,  'sensor_39': 37.236,
                                'sensor_40': 65.248,  'sensor_41': 35.697,  'sensor_42': 35.817,  'sensor_43': 43.272,
                                'sensor_44': 40.335,  'sensor_45': 41.135,  'sensor_46': 45.176,  'sensor_47': 43.246,
                                'sensor_48': 122.100, 'sensor_49': 52.637,  'sensor_50': 0.0,     'sensor_51': 202.055
                            }
        
        # Standard Variations (under normal conditions)
        self.nominal_stds = {
                                'sensor_00': 0.447,   'sensor_01': 3.424,   'sensor_02': 4.140,   'sensor_03': 2.585,
                                'sensor_04': 161.666, 'sensor_05': 19.456,  'sensor_06': 2.304,   'sensor_07': 2.447,
                                'sensor_08': 2.265,   'sensor_09': 2.345,   'sensor_10': 13.449,  'sensor_11': 13.827,
                                'sensor_12': 10.749,  'sensor_13': 5.899,   'sensor_14': 126.661, 'sensor_15': 0.0,
                                'sensor_16': 141.303, 'sensor_17': 145.749, 'sensor_18': 0.861,   'sensor_19': 222.869,
                                'sensor_20': 113.895, 'sensor_21': 253.464, 'sensor_22': 170.156, 'sensor_23': 315.528,
                                'sensor_24': 204.021, 'sensor_25': 247.086, 'sensor_26': 272.439, 'sensor_27': 154.473,
                                'sensor_28': 340.112, 'sensor_29': 256.824, 'sensor_30': 214.968, 'sensor_31': 319.827,
                                'sensor_32': 283.217, 'sensor_33': 171.013, 'sensor_34': 94.152,  'sensor_35': 144.879,
                                'sensor_36': 304.376, 'sensor_37': 31.008,  'sensor_38': 11.100,  'sensor_39': 16.428,
                                'sensor_40': 21.070,  'sensor_41': 8.177,   'sensor_42': 11.317,  'sensor_43': 11.741,
                                'sensor_44': 8.555,   'sensor_45': 10.310,  'sensor_46': 13.649,  'sensor_47': 9.088,
                                'sensor_48': 63.938,  'sensor_49': 14.243,  'sensor_50': 0.0,     'sensor_51': 119.694
                            }
        #  most critical sensors according to Feature importance - more degredation
        self.high_impact_sensors = [
                                        'sensor_22', 'sensor_23', 'sensor_32', 'sensor_29', 'sensor_30',
                                        'sensor_25', 'sensor_00', 'sensor_26', 'sensor_34', 'sensor_14', 
                                        'sensor_28'
                                    ]
        # Sensors that react strongly to load (pressure/vibration related)
        self.load_sensitive = [
                                'sensor_04', 'sensor_14', 'sensor_21', 'sensor_22',
                                'sensor_23', 'sensor_28', 'sensor_29', 'sensor_30'
                            ]
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

    def generate_sensor_data(self):
        """
        Generates one timestep of realistic synthetic sensor data for the pump.
        
        Returns:
            dict: Raw sensor readings (50 sensors), like real data
        """
        sensor_data = {}

        current_rul = max(0.0, self.max_lifetime - self.total_ran_cycles)
        degradation_ratio = max(0.0, min(1.0, 1.0 - (current_rul / self.max_lifetime)))

        tonnage = self.calculate_tonnage()
        load_factor = tonnage / 50.0  # 50 ton average reference

        # Noise increases as pump degrades
        noise_multiplier = 1.0 + degradation_ratio * 2.0  # up to 3x noise at end

        # Load effect from hydraulic press (higher tonnage = higher values in pressure/vibration sensors)
        load_factor = tonnage / 50.0  # 50 ton ortalama kabul ediyoruz

        for sensor_name, nominal_mean in self.nominal_means.items():
            base_std = self.nominal_stds[sensor_name]

            # Base value with normal noise
            value = np.random.normal(nominal_mean, base_std * 0.5)

            # Add load effect (more tonnage = higher readings)
            if sensor_name in self.load_sensitive:
                value += load_factor * 15.0  # pressure-related sensors rise with load

            # Critical sensors degrade more visibly
            if sensor_name in self.high_impact_sensors:
                # Drift: move away from nominal as RUL decreases
                drift_direction = 1.0 if sensor_name in ['sensor_22', 'sensor_23', 'sensor_29', 'sensor_30', 'sensor_32'] else -1.0
                drift_amount = drift_direction * degradation_ratio * 80.0  # max ±80 birim sapma
                value += drift_amount

                # Extra noise for high impact sensors
                extra_noise = np.random.normal(0, base_std * degradation_ratio * 2.0)
                value += extra_noise

            # General degradation: all sensors get noisier near failure
            noise = np.random.normal(0, base_std * noise_multiplier)
            value += noise

            # Final clipping to realistic bounds
            min_val = max(0.0, nominal_mean * 0.3)
            max_val = nominal_mean * 2.0
            value = np.clip(value, min_val, max_val)

            sensor_data[sensor_name] = round(float(value), 6)

        self.total_ran_cycles += 1

        if current_rul <= 0:
            self.get_logger().warn(f"Process_Pump has reached end of life at cycle {self.total_ran_cycles - 1}!")

        return sensor_data

    def run_production_cycle(self):
        """
        Callback for the rclpy.Timer (REALTIME mode).
        Runs one production cycle every 60 seconds.
        """
        current_rul = max(0.0, self.max_lifetime - self.total_ran_cycles)
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
            for i in range(self.cycles_to_run):
                # In FAST mode, just loop and publish
                current_rul = max(0.0, self.max_lifetime - self.total_ran_cycles)
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
        sensor_data = self.generate_sensor_data()

        sensor_data['cycle'] = self.total_ran_cycles - 1

        sensor_msg = String()
        sensor_msg.data = json.dumps(sensor_data)
        self.publisher_sensors.publish(sensor_msg)
        self.get_logger().info(f"Published sensor data for cycle {self.total_ran_cycles - 1}.")


def main(args=None):
    """
    Main entry point for the machine_hydraulic_press_sensor_node.
    """
    rclpy.init(args=args)
    node = Machine_Process_Pump_Sensor_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()