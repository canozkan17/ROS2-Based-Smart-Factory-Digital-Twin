#!/usr/bin/env python3

"""
ROS2 Node for generating predictions according to the machine sensor data in the system.

Subscribes to Sensors topic for raw data input. 

1- Loads scaler and model to memory on initialization. 
2- Gets the raw data input from Sensors/machine as JSON file.
3- Preprocesses the raw data using training scaler and/or feature extraction.
4- Generates RUL predictions
5- Publishes RUL predictions to Predictions/machine Topic.

Publishes selected job order to Predictions Topic. 
"""

from rclpy.executors import MultiThreadedExecutor
from scipy.stats import kurtosis
from std_msgs.msg import String
from scipy.stats import skew
from rclpy.node import Node

import warnings
import xgboost as xgb
import pandas as pd
import numpy as np
import rclpy
import joblib
import json
import os

# does not use dataframe for RPI performance reasons - ignored specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names") 

class Predictor_Node(Node):

    def __init__(self):
        """
        Initialize the Predictor node, set up subscriptions and publishers.
        Get models and scalers from disk into memory.
        set up variables/constants and flags.
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
        
        
        # Hydraulic variables
        self.hydraulic_history = []

        # Process Pump variables
        self.pump_history = [] # for last 5 cycles - each 60 seconds
        self.max_pump_history_length = 5

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
        # the direct sensors used in 35 features for Process_Pump 
        self.process_pump_direct_sensors = [
                                                'sensor_02', 'sensor_01', 'sensor_03', 'sensor_36', 'sensor_49',
                                                'sensor_42', 'sensor_26', 'sensor_00', 'sensor_13', 'sensor_28',
                                                'sensor_29', 'sensor_45', 'sensor_34'
                                            ] # 13
        # the rolling sensors used in 35 features for Process_Pump - total makes the 35 features the model expects
        self.roll_sensors = [
                                'sensor_02', 'sensor_03', 'sensor_01', 'sensor_36', 'sensor_26',
                                'sensor_28', 'sensor_29', 'sensor_00', 'sensor_13', 'sensor_32',
                                'sensor_23', 'sensor_34', 'sensor_25', 'sensor_07', 'sensor_35',
                                'sensor_06', 'sensor_49', 'sensor_30', 'sensor_22', 'sensor_14',
                                'sensor_42', 'sensor_33'
                            ]   # 22
        # ordered features for Process_Pump model input 
        self.ordered_features = [
                                    'sensor_02_roll_mean', 'sensor_03', 'sensor_02', 'sensor_01_roll_mean', 'sensor_01',
                                    'sensor_03_roll_mean', 'sensor_36_roll_mean', 'sensor_26_roll_mean', 'sensor_28_roll_mean',
                                    'sensor_29_roll_mean', 'sensor_36', 'sensor_00_roll_mean', 'sensor_13_roll_mean',
                                    'sensor_49', 'sensor_42', 'sensor_32_roll_mean', 'sensor_26', 'sensor_23_roll_mean',
                                    'sensor_34_roll_mean', 'sensor_42_roll_mean', 'sensor_00', 'sensor_13', 'sensor_28',
                                    'sensor_25_roll_mean', 'sensor_29', 'sensor_07_roll_mean', 'sensor_35_roll_mean',
                                    'sensor_06_roll_mean', 'sensor_49_roll_mean', 'sensor_30_roll_mean', 'sensor_22_roll_mean',
                                    'sensor_14_roll_mean', 'sensor_45', 'sensor_33_roll_mean', 'sensor_34'
                                ] # 35

        # Control flags
        self.process_finished = False
        self.user_input_received = False
        
        # Set-up logs
        self.get_logger().info("Predictor node ready!")
        self.get_logger().info("Listening on 'Sensors/ ' topic. For 2 machine sensors")

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

    def listener_hydraulic_press_callback(self, msg: String):
        """
        Callback function for Hydraulic_Press subscription.
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Hydraulic_Press message")
            
            # update history and keep only last 6 entries
            self.hydraulic_history.append(received_data)
            if len(self.hydraulic_history) > 6:
                self.hydraulic_history.pop(0) 

            self.hydraulic_press_generate_predictions(received_data)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")

    def hydraulic_press_generate_predictions(self, received_data:dict):
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
            self.publish_generated_data(rul, cycle=received_data['cycle'], machine="hydraulic_press")
            self.hydraulic_last_predicted_cycle = received_data['cycle']
        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")
            return None
    # Helper
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

        X = np.array(features).reshape(1, -1)
        X = pd.DataFrame([features], columns=feature_names)  
        return X.values
    # Helper
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
    # ----------------------------------------------------------------

    # PROCESS PUMP MACHINE METHODS
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
    
    def listener_process_pump_callback(self, msg: String):
        """
        Callback function for Process_Pump subscription.
        """
        try: 
            received_data = json.loads(msg.data)
            self.get_logger().info(f"Received Process_Pump message")

            # update history and keep only last max_pump_history_length entries
            self.pump_history.append(received_data)
            if len(self.pump_history) > self.max_pump_history_length:
                self.pump_history.pop(0)
            # before the required window is filled, pad with current data
            while len(self.pump_history) < self.max_pump_history_length:
                self.pump_history.insert(0, received_data)
            
            self.process_pump_generate_predictions(received_data)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
    
    def process_pump_generate_predictions(self, received_data:dict):
        """
        Generate RUL predictions for Process_Pump machine.
        """
        X = self.process_pump_feature_extractor()
        if X is None or X.shape != (1, 35):
            self.get_logger().warning("No features extracted, skipping prediction.")
            return None
        try:
            X_scaled = self.process_pump_scaler.transform(X)
            rul_hrs = float(self.process_pump_model.predict(X_scaled)[0])
            rul = round(rul_hrs * 60)  # hours -> cycles (since 1 cycle = 60 seconds)
        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")
            return None
        
        self.publish_generated_data(rul, cycle=received_data['cycle'], machine="process_pump")
    # Helper
    def process_pump_feature_extractor(self):
        """
        Feature extraction for Process_Pump machine sensor data.
        """
        history = self.pump_history.copy()
        features = {}

        # Direct sensor features
        for sensor in self.process_pump_direct_sensors:
            sensor_values = [cycle_data.get(sensor, 0.0) for cycle_data in history]
            features[sensor] = sensor_values[-1]  # last value
        
        # Rolling sensor features
        for sensor in self.roll_sensors:
            sensor_values = [cycle_data.get(sensor, 0.0) for cycle_data in history]
            arr = np.array(sensor_values)
            features[f"{sensor}_roll_mean"] = np.mean(arr)
            
        # Arrange features in the expected order
        X = [features.get(feat, 0.0) for feat in self.ordered_features]
        return np.array(X).reshape(1, -1)
    #----------------------------------------------------------------


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