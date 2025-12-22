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
from system_nodes.prediction_handler import process_pump_prediction_handler as pp_handler

from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from rclpy.node import Node

import warnings
import pandas as pd
import numpy as np
import rclpy
import json
import os
import math


class Predictor_Node(Node):

    def __init__(self):
        """
        Initialize the Predictor node, set up subscriptions and publishers.
        Get models and scalers from disk into memory via handlers at ~/prediction_handler.
        set up variables/constants and flags.
        """
        super().__init__('Predictor_Node')
        
        # Subscription to machine_hydraulic_press_node  
        #self.subscription_hydraulic_press_node = self.create_subscription(String, 'Sensors/hydraulic_press', self.listener_hydraulic_press_callback, 10)        
        
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
        self.pump_rolling_window = 60               # rolling window size for feature engineering
        self.predicted_rul_process_pump = None

        # Load models and features into memory
        pp_handler.load_models_and_features()
    
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
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Process_Pump message")

            # update history and keep only last max_pump_history_length entries
            self.pump_history.append(received_data)

            if len(self.pump_history) < self.pump_rolling_window:
                self.get_logger().warning(f"Not enough history for Process_Pump: {len(self.pump_history)}/{self.pump_rolling_window}")
                return

            if len(self.pump_history) > 1000:       # for RPI memory limits
                self.pump_history.pop(0)
            
            # Wait until enough history is collected
            if len(self.pump_history) < self.pump_rolling_window:
                self.get_logger().warning(f"Not enough history for Process_Pump: {len(self.pump_history)}/{self.pump_rolling_window}. ")
                return None
            
            # WHERE MAGIC HAPPENS
            self.predicted_rul_process_pump = pp_handler.get_prediction(self.pump_history)

            self.publish_generated_data(
                                            self.predicted_rul_process_pump, 
                                            cycle=received_data.get("cycle", -1), 
                                            machine="process_pump"
                                        )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
    


    def publish_generated_data(self, rul:float, cycle:int, machine=None):
        """
        Generic publisher for RUL predictions.
        Selects the correct publisher based on machine type.
        
        """
        if machine is not None and rul is not None and cycle is not None:

            prediction_msg = String()
            prediction_msg.data = json.dumps({
                                            "machine": machine,
                                            "cycle": cycle,
                                            "rul": rul
                                        })
            self.prediction_publishers[machine].publish(prediction_msg)
            self.get_logger().info(f"Published RUL prediction for {machine} at cycle {cycle}: RUL={rul}")

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