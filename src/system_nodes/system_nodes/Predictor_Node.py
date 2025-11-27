#!/usr/bin/env python3

"""
ROS2 Node for generating predictions according to the machine sensor data in the system.

Subscribes to Sensors topic for raw data input. 

1- Loads scaler and model to memory on initialization. 
2- Gets the raw data input from Sensors/machine as JSON file.
3- Preprocesses the raw data using training scaler and feature extraction.
4- Generates RUL predictions
5- Publishes RUL predictions to Predictions/machine Topic.

Publishes selected job order to Predictions Topic. 
"""

from scipy.stats import kurtosis
from std_msgs.msg import String
from scipy.stats import skew
from rclpy.node import Node
from tqdm import tqdm
import xgboost as xgb
import pandas as pd
import numpy as np
import rclpy
import joblib
import json
import time
import os

class Predictor_Node(Node):
    """ROS2 Node for generating job orders to the production system."""

    def __init__(self):
        """
        Initialize the Job Scheduler node, set up subscriptions and publishers.
        Get user input for job scheduling.
        """
        super().__init__('Predictor_Node')
        
        # Subscription to machine_hydraulic_press_node  
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
        # Load models and scalers
        self.load_hydraulic_press_model()
        self.load_process_pump_model()

        # Variable setup
        self.received_raw_data = {}
        self.machine = str()
        
        # Hydraulic variables
        self.hydraulic_history = []

        # Defaults
        self.hydraulic_sensors = [
                                    'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',      # 100 Hz
                                    'EPS1',                                        # 100 Hz
                                    'FS1', 'FS2',                                  # 10 Hz
                                    'TS1', 'TS2', 'TS3', 'TS4',                    # 1 Hz
                                    'VS1',                                         # 1 Hz
                                    'CE',                                          # 1 Hz (virtual)
                                    'CP',                                          # 1 Hz (virtual)
                                    'SE'                                           # 1 Hz (virtual)
                                ]

        # Control flags
        self.process_finished = False
        self.user_input_received = False
        
        # Set-up logs
        self.get_logger().info("Job Scheduler node ready!")
        self.get_logger().info("Listening on 'Sensors/' topic. for 2 machine sensors")

    # HYDRAULIC PRESS MACHINE METHODS    
    def load_hydraulic_press_model(self):
        """
        Loads the pre-trained model and scaler for Hydraulic_Press machine.
        """
        model= os.path.join(self.root_path, 'hydraulic_press_data', 'hydraulic_press_rul_xgb.json')
        self.hydraulic_press_model = xgb.XGBRegressor()
        self.hydraulic_press_model.load_model(model)

        scaler= os.path.join(self.root_path, 'hydraulic_press_data', 'hydraulic_scaler.pkl')
        self.hydraulic_press_scaler = joblib.load(scaler)
        self.get_logger().info("Hydraulic_Press model and scaler loaded.")

    def hydraulic_press_feature_extractor(self):
        """
        Feature extraction for Hydraulic_Press machine sensor data.
        """
        features = []
        full_history = self.hydraulic_history.copy()

        
        # Extract from each sensor (use pre-loaded sensor_dfs)
        for sensor in self.hydraulic_sensors:
            arr_list = []
            for past_cycle in full_history:
                if sensor in past_cycle:
                    arr_list.append(np.array(past_cycle[sensor], dtype=np.float64))
            if not arr_list:
                self.get_logger().warning(f"No data for sensor {sensor}, skipping feature extraction. Assigning zeros.")
                features.extend([0.0]*9)  # 9 features per sensor


            single_sensor_feats = self.hydraulic_press_extract_features_with_trend(arr_list, window_size=5)[-1]
            features.extend(single_sensor_feats)
        

        # Feature names (17 sensors x 9 stats)
        stat_names = ['mean', 'std', 'min', 'max', 'rms', 'skew', 'kurt', 'ptp', 'trend']
        feature_names = [f"{sensor}_{stat}" for sensor in self.hydraulic_sensors for stat in stat_names]
        
        if len(features) != 153:
            self.get_logger().error(f"Feature count mismatch! Expected 153, got {len(features)}")
            return None

        return np.array(features).reshape(1, -1)  # (1, 153)
    
    def hydraulic_press_extract_features_with_trend(self, arr_list, window_size=5):
        """
        Extract statistical and temporal trend features.
        Each arr in arr_list represents one cycle for a single sensor.
        """
        feats_all = []
        for i in range(len(arr_list)):
            arr = np.asarray(arr_list[i], dtype=np.float64).flatten()
            mean_ = np.mean(arr)
            std_ = np.std(arr, ddof=0)
            min_ = np.min(arr)
            max_ = np.max(arr)
            rms_ = np.sqrt(np.mean(arr ** 2))
            skew_ = skew(arr)
            kurt_ = kurtosis(arr)
            ptp_ = np.ptp(arr)

            # temporal trend (difference from rolling mean of previous N cycles)
            if i >= window_size:
                prev_vals = [np.mean(arr_list[j]) for j in range(i - window_size, i)]
                trend = mean_ - np.mean(prev_vals)
            else:
                trend = 0.0

            feats_all.append([mean_, std_, min_, max_, rms_, skew_, kurt_, ptp_, trend])
        return np.array(feats_all)
    
    def listener_hydraulic_press_callback(self, msg: String):
        """
        Callback function for Hydraulic_Press subscription.
        """
        try: 
            received_data = json.loads(msg.data)
            self.received_raw_data = received_data
            self.machine = "hydraulic_press"
            # update history and keep only last 6 entries
            self.hydraulic_history.append(self.received_raw_data)
            if len(self.hydraulic_history) > 6:
                self.hydraulic_history.pop(0) 


            self.hydraulic_press_generate_predictions()
        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
        
    def hydraulic_press_generate_predictions(self):
        """
        Extract features and generate RUL predictions for Hydraulic_Press machine.
        """
                
        
        
        X = self.hydraulic_press_feature_extractor()
        if X is None or X.shape != (1, 153):
            self.get_logger().warning("No features extracted, skipping prediction.")
            return None
        try:
            X_scaled = self.hydraulic_press_scaler.transform(X)
            rul = float(self.hydraulic_press_model.predict(X_scaled)[0])
            self.publish_generated_data(rul)
            self.hydraulic_last_predicted_cycle = self.received_raw_data['cycle']
        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")
            return None
    # ----------------------------------------------------------------

    # Done - Leave Alone
    def load_process_pump_model(self):
        """
        Loads the pre-trained model and scaler for Process_Pump machine.
        """
        model= os.path.join(self.root_path, 'pump_sensor_rul_hrs_data', 'pump_rul_xgb.json')
        self.process_pump_model = xgb.XGBRegressor()
        self.process_pump_model.load_model(model)

        scaler= os.path.join(self.root_path, 'pump_sensor_rul_hrs_data', 'scaler_pump_top35_features.pkl')
        self.process_pump_scaler = joblib.load(scaler)
        self.get_logger().info("Process_Pump model and scaler loaded.")

    # TODO: PENDING
    def listener_process_pump_callback(self, msg: String):
        """
        Callback function for Process_Pump subscription.
        """
        self.get_logger().info(f"Received Process_Pump message: {json.dumps(msg.data, indent=2)}")
        # the whole process is pending implementation
        self.machine = "process_pump" # place holder to not forget
    

    # TODO: PENDING either machine specific or generic - will be decided.
    def publish_generated_data(self, rul):
        """
        Publishes generated sensor data to Sensors topic.
        """
        cycle = self.received_raw_data.get('cycle', "unknown")
        prediction_msg = String()
        prediction_msg.data = json.dumps({
                                            "machine": self.machine,
                                            "cycle": cycle,
                                            "rul": rul
                                        })
        self.prediction_publishers[self.machine].publish(prediction_msg)
        self.get_logger().info(f"Published RUL prediction for {self.machine} at cycle {cycle}: RUL={rul}")


def main(args=None):
    """
    Main entry point for the Predictor_Node.
    """
    rclpy.init(args=args)
    node = Predictor_Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()