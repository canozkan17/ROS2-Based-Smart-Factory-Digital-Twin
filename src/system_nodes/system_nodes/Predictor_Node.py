#!/usr/bin/env python3

"""
ROS2 Node for generating predictions according to the machine sensor data in the system.

Subscribes to Sensors topic for raw data input. 

1- Loads Models and featurelists into memory on initialization. 
2- Gets the raw data input from Sensors/machine as JSON file.
3- Preprocesses the raw data using training scaler and/or feature extraction.
4- Generates RUL predictions
5- Publishes RUL predictions to Predictions/machine Topic.

Publishes selected job order to Predictions Topic. 
"""
from system_nodes.prediction_handler import process_pump_prediction_handler as process_pump_handler
from system_nodes.prediction_handler import hydraulic_press_prediction_handler as hydraulic_press_handler
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from rclpy.node import Node
import numpy as np
import rclpy
import json
import os



class Predictor_Node(Node):

    def __init__(self):
        """
        Initialize the Predictor node, set up subscriptions and publishers.
        Get models and scalers from disk into memory via handlers at ~/prediction_handler.
        set up variables/constants and flags.
        """
        super().__init__('Predictor_Node')
        
        # Subscription to machine_hydraulic_press_node  TODO!!
        self.subscription_hydraulic_press_node = self.create_subscription(String, 'Sensors/hydraulic_press', self.listener_hydraulic_press_callback, 10)
        
        # Subscription to machine_process_pump_node
        self.subscription_process_pump_node = self.create_subscription(String, 'Sensors/process_pump', self.listener_process_pump_callback, 10)


        # Publishers for predictions
        self.prediction_publishers = {
                                        "hydraulic_press": self.create_publisher(String, "Predictions/hydraulic_press", 10),
                                        "process_pump": self.create_publisher(String, "Predictions/process_pump", 10)
                                    }    
        
        # Adress setup
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.abspath(os.path.join(self.current_dir, "../../.."))

        # Process Pump variables
        self.pump_history = []                      # for all the cycles
        self.pump_rolling_window = 5                # minimum samples for rolling calculations (matches WINDOW_SHORT)
        self.predicted_rul_process_pump = None

        # Hydraulic Press variables
        self.hydraulic_press_history = []
        self.hydraulic_press_rolling_window = 20
        self.predicted_rul_hydraulic_press = None

        # Load models and features into memory
        process_pump_handler.load_models()
        hydraulic_press_handler.load_models()
    
        # Control flags
        self.process_finished = False
        self.user_input_received = False
        
        # Set-up logs
        self.get_logger().info("Predictor node ready!")
        self.get_logger().info("Listening on 'Sensors/ ' topic. For 2 machine sensors")

    # PROCESS PUMP MACHINE METHOD
    def listener_process_pump_callback(self, msg: String):
        """
        Callback function for Process_Pump subscription.
        Waits till enough history is collected (5 for rolling/mean calculations No _20 feature in saved in training), then calls the prediction handler.
        Resets history on cycle==0 (at new run/after maintenance).
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Process_Pump message")

            if received_data.get("cycle") == 0:
                self.pump_history = []          # reset history at new run / after maintenance
                self.get_logger().info("Reset Process_Pump history for new run.")
                try:
                    process_pump_handler.reset_state()
                except Exception:
                    pass

            # update history and keep only last max_pump_history_length entries
            self.pump_history.append(received_data)

            if len(self.pump_history) < self.pump_rolling_window:
                self.get_logger().warning(
                                            f"Not enough history for Process_Pump: {len(self.pump_history)}/{self.pump_rolling_window}"
                                        )
                return

            if len(self.pump_history) > 500:       # for RPI memory limits
                self.pump_history.pop(0)
                
            # Wait until enough history is collected
            if len(self.pump_history) < self.pump_rolling_window:
                self.get_logger().warning(f"Not enough history for Process_Pump: {len(self.pump_history)}/{self.pump_rolling_window}. ")
                return None
                
            # WHERE MAGIC HAPPENS
            prediction_result = process_pump_handler.predict(self.pump_history)

            if isinstance(prediction_result, dict):
                self.predicted_rul_process_pump = prediction_result.get("rul_min")
                prediction_payload = prediction_result
            else:
                self.get_logger().error("Invalid prediction output format")
                return

            self.publish_generated_data(
                                            prediction_payload = prediction_payload,
                                            cycle=received_data.get("cycle", -1),
                                            machine="process_pump"
                                        )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
    
    def listener_hydraulic_press_callback(self, msg: String):
        """
        Callback function for Hydraulic_Press subscription.
        Waits till enough history is collected (20 for rolling/mean calculations), then calls the prediction handler.
        Resets history on cycle==0 (at new run/after maintenance).
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Hydraulic_Press message")

            if received_data.get("cycle") == 0:
                self.get_logger().info(f"[TEMP:DEBUG] PREDICTOR cycle==0 detected, resetting history (was {len(self.hydraulic_press_history)} entries)")
                self.hydraulic_press_history = []          # reset history at new run / after maintenance
                self.get_logger().info("Reset Hydraulic_Press history for new run.")
                try:
                    hydraulic_press_handler.reset_state()
                except Exception:
                    pass

            # update history and keep only last max_hydraulic_press_history_length entries
            self.hydraulic_press_history.append(received_data)
            if len(self.hydraulic_press_history) < self.hydraulic_press_rolling_window:
                self.get_logger().warning(
                                            f"Not enough history for Hydraulic_Press: {len(self.hydraulic_press_history)}/{self.hydraulic_press_rolling_window}"
                                        )
                return

            if len(self.hydraulic_press_history) > 500:       # for RPI memory limits
                self.hydraulic_press_history.pop(0)
                
            # Wait until enough history is collected
            if len(self.hydraulic_press_history) < self.hydraulic_press_rolling_window:
                self.get_logger().warning(f"Not enough history for Hydraulic_Press: {len(self.hydraulic_press_history)}/{self.hydraulic_press_rolling_window}. ")
                return None
            
            # Debug: log history range before prediction
            first_cycle = self.hydraulic_press_history[0].get("cycle", -1)
            last_cycle = self.hydraulic_press_history[-1].get("cycle", -1)
            self.get_logger().info(f"[TEMP:DEBUG] PREDICTOR calling handler.predict() with history len={len(self.hydraulic_press_history)}, cycles {first_cycle}-{last_cycle}")
                
            # WHERE MAGIC HAPPENS
            prediction_result = hydraulic_press_handler.predict(self.hydraulic_press_history)
            
            if isinstance(prediction_result, dict):
                self.predicted_rul_hydraulic_press = prediction_result.get("rul_min")
                prediction_payload = prediction_result
            else:
                self.get_logger().error("Invalid prediction output format")
                return

            self.publish_generated_data(
                                            prediction_payload = prediction_payload,
                                            cycle=received_data.get("cycle", -1),
                                            machine="hydraulic_press"
                                        )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    def publish_generated_data(self, prediction_payload, cycle:int, machine=None):
        """
        Generic publisher for RUL predictions.
        Selects the correct publisher based on machine type.
        Machine specific logging due to different payload structures.
        """
        if machine is not None and prediction_payload is not None and cycle is not None:

            prediction_msg = String()
            prediction_msg.data = json.dumps({
                                                "machine": machine,
                                                "cycle": cycle,
                                                "rul": prediction_payload
                                            })
            self.prediction_publishers[machine].publish(prediction_msg)

            if machine == "hydraulic_press":
                self.get_logger().info(
                                            f"Published RUL prediction for {machine} at cycle {cycle}: "
                                            f"RUL={prediction_payload.get('rul_min', 'N/A'):.2f} min, "
                                            f"Active Model={prediction_payload.get('active_model', 'N/A')}, "
                                            f"Stage_0_Prob={prediction_payload.get('stage0_prob', 'N/A')}, "
                                            f"Stage_1_Prob={prediction_payload.get('stage1_prob', 'N/A')}, "
                                        )

            elif machine == "process_pump":
                self.get_logger().info(
                                            f"Published RUL prediction for {machine} at cycle {cycle}: "
                                            f"RUL={prediction_payload.get('rul_min', 'N/A'):.2f} min, "
                                            f"stage={prediction_payload.get('stage', 'N/A')}, "
                                            f"crit_prob={float(prediction_payload.get('crit_prob', 0.0)):.3f}"
                                        )

def main(args=None):
    """
    Main entry point for the Predictor_Node.
    """
    rclpy.init(args=args)
    node = Predictor_Node()
    executor = MultiThreadedExecutor(num_threads=4) # Optimized for RPI-4 
    rclpy.spin(node, executor=executor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()