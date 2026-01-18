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
import numpy as np
import threading
import random
import rclpy
import json
import time


class Machine_Process_Pump_Sensor_Node(Node):
    """ROS2 Node for generating synthetic sensor data for Process Pump Machine."""

    def __init__(self):
        """
        Initialize the Process Pump Sensor node, set up subscriptions and publishers.
        """
        super().__init__('Machine_Process_Pump_Sensor_Node')
        
        # Subscription to Job Orders
        self.subscription_job_order = self.create_subscription(String, 'Job_Orders', self.listener_job_orders_callback, 100)
    
        # Subscription to Control CMD
        self.subscription_control_cmd = self.create_subscription(String, 'Control_CMD/process_pump', self.listener_callback_control_cmd, 100)
        self.subscription_control_cmd_2 = self.create_subscription(String, 'Control_CMD/hydraulic_press', self.listener_callback_control_cmd, 100) # listens to the main machine too.
        

        # Publisher for Sensors data
        self.publisher_sensors = self.create_publisher(String, "Sensors/process_pump", 100)
        
        # Publisher for maintenance feedback 
        self.publisher_maintenance_feedback = self.create_publisher(String, "Maintenance_Feedback/process_pump", 100)
        # Publisher for Production Log (ground-truth RUL)
        self.publisher_production_log = self.create_publisher(String, "Production_Log", 100)
            

        # Variable set-up
        self.total_ran_cycles = 0 # real cycle count in minutes
        self.current_task = {}

        # Simulation and production control flags
        # Set "REALTIME" for 60s per cycle, "FAST" for max speed
        self.simulation_mode = "FAST"      # default FAST, user decides in gui
        self.simulation_mode_lock = False
        self.production_timer = None       # Holds the rclpy.Timer object
        self.cycles_to_run = 0
        self.cycles_done = 0

        self.in_maintenance = False
        self.control_cmd = "NORMAL_OPERATION"

        # Defaults
        self.SEED = 42
        self.degredation_factor = 1.0

        # Training generator (generate_pump_data.py) uses this order:
        #   1. RNG.integers(30000, 48001) -> total_rul
        #   2. RNG.uniform(0.02, 0.3) -> base_vib
        #   3. RNG.uniform(2.0, 6.0) -> failure_vib
        rng = np.random.default_rng(self.SEED)
        
        # total_rul (MUST be first to sync RNG state with training)
        self.total_lifetime_minutes = int(rng.integers(30000, 48001))  # matches training
        self.max_lifetime = self.total_lifetime_minutes / 60.0  # convert to hours
        self.get_logger().info(f"Process_Pump initialized with max lifetime: {self.max_lifetime:.1f} hours ({self.total_lifetime_minutes} min)")

        # Sensor parameters (same order as generate_pump_data.py)
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
        
        # Store RNG for noise generation during simulation
        self.RNG = rng
        
        self.get_logger().info("Process Pump Sensor node ready!")
        self.get_logger().info("Listening on 'Job_Orders' topic.")
        self.get_logger().info("Listening on 'Control_CMD' topic.")
        # Publisher for Node status
        self.publisher_node_status = self.create_publisher(String, "Node_Status", 10)
        self._node_status = 'READY'
        try:
            self.node_status_timer = self.create_timer(2.0, self._publish_node_status)
        except Exception:
            self.node_status_timer = None
        try:
            msg = String()
            msg.data = json.dumps({"node": self.get_name(), "status": self._node_status})
            self.publisher_node_status.publish(msg)
        except Exception:
            pass

    def _publish_node_status(self):
        try:
            msg = String()
            msg.data = json.dumps({"node": self.get_name(), "status": self._node_status})
            self.publisher_node_status.publish(msg)
        except Exception:
            pass

    
    # Callback functions
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
                    # check if the mode has been entered - first time we accept mode, later we lock it
                    if not self.simulation_mode_lock:
                        self.simulation_mode = task.get("mode", "FAST")  # default to FAST if not specified
                        self.simulation_mode_lock = True
                        self.get_logger().info(f"Simulation mode set to: {self.simulation_mode}")

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

    # Callback for Control CMD
    def listener_callback_control_cmd(self, msg: String):
        """
        Callback function for Control CMD subscription.
        Processes control commands for corrective/preventative actions.
        """
        control_msg = json.loads(msg.data)
        self.get_logger().info(f"Received Control Command")

        # Extract control command and recovery time from the message
        command = control_msg.get("command", "NORMAL_OPERATION")
        machine = control_msg.get("machine", "process_pump")
        self.maintenance_cycles_remaining = control_msg.get("recovery_time_min", 0)

        if command == "SHUTDOWN":
            self.get_logger().info(f"Received SHUTDOWN command for machine {machine}. Stopping current task.")
            self.control_cmd = "SHUTDOWN"
            self.in_maintenance = True
            if self.production_timer:
                self.production_timer.cancel()
                self.production_timer = None

        elif command == "SLOW_DOWN":
            self.control_cmd = "SLOW_DOWN"
            self.get_logger().info(f"Received SLOW_DOWN command for machine {machine}. Slowing down operation.")
            self.degredation_factor *= 0.9  # slowing down degradation by 10%
        
        
        elif command == "NORMAL_OPERATION":
            self.get_logger().info(f"Received NORMAL_OPERATION command for machine {machine}. Continuing operation.")
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
            self.get_logger().info(
                                        f"FAST maintenance started for {self.maintenance_cycles_remaining} cycles"
                                    )

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
        """
        Generate synthetic sensor data for a single pump cycle until failure.
        """

        t = float(self.cycles_done)
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
                        + (self.failure_vib - self.base_vib) * (fraction ** 2) * critical_boost * self.degredation_factor
                        + rng.normal(0.0, self.vib_noise_scale * (1.0 + fraction), size=1)
                    )[0]
        temp_motor = (
                        self.base_temp + self.k_temp * t * critical_boost * self.degredation_factor
                        + rng.normal(0.0, self.temp_noise_scale, size=1)
                    )[0]
        pressure = (
                        self.base_pressure - self.k_pressure * t * critical_boost * self.degredation_factor
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
                # Small delay to avoid overwhelming subscribers in FAST mode
                time.sleep(0.02)
            
            if self.in_maintenance:
                if self.control_cmd == "SLOW_DOWN":
                    self.get_logger().info(f"FAST simulation stopped due to SLOW_DOWN command")
                self.maintenance_timer()
            
                        
        except Exception as e:
            self.get_logger().error(f"Error during FAST simulation: {e}")
        
        finally:
            self.get_logger().info(f"Task {self.current_task.get('job_ID')} completed in FAST mode.")
    
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

        # Ensure current_rul_min is available and publish to Production_Log
        try:
            current_rul_min = int(self.total_lifetime_minutes - cycle)
        except Exception:
            current_rul_min = None

        prod_msg = String()
        prod_msg.data = json.dumps({
                                    "machine": "process_pump",
                                    "cycle": cycle,
                                    "current_rul_min": current_rul_min,
                                    "total_lifetime_minutes": self.total_lifetime_minutes
                                })
        self.publisher_production_log.publish(prod_msg)
            
        self.get_logger().info(
                                f"Published sensor data for cycle: {cycle} (in minutes) "
                            )
        self.get_logger().info(f"GroundTruth RUL={gt_rul_hours:.1f} hours ({(gt_rul_hours*60):.1f} min) after cycle {self.cycles_done}/{self.cycles_to_run}.")


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

