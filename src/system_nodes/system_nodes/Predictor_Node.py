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
import pickle
import os
import math

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
                
        # Hydraulic variables
        self.hydraulic_history = []
        self.hydraulic_last_predicted_cycle = -1

        # Process Pump variables
        self.pump_history = []                      # for all the cycles
        self.last_callback_process_pump_cycle = -1
        self.process_pump_ordered_features = []     # ordered features for Process_Pump model input 
        self.pump_rolling_window = 5

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

        # Load models and scalers
        self.load_hydraulic_press_model()
        self.load_process_pump_model()
        
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
            rul_cycles = float(self.hydraulic_press_model.predict(X_scaled)[0])
            rul = rul_cycles / 60
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
                continue


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

        bundle = joblib.load(os.path.join(self.root_path, "pump_sensor_rul_hrs_data", "scaler_pump_top35_features.pkl"))
        self.process_pump_scaler = bundle if hasattr(bundle, "transform") else bundle["scaler"]

        scaler_features = list(self.process_pump_scaler.feature_names_in_)

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # json will be in the same dir in RPI deployment [Activate in Edge Deployment]
        # json_path = os.path.join(base_dir, "selected_cols_top35.json")
        json_path = os.path.normpath(os.path.join(  base_dir,
                                                    "../../..",  # Currenly (temporarily) 3 levels up to Capstone_Project root
                                                    "pump_sensor_rul_hrs_data/diagnostics_out/selected_cols_top35.json"
                                                ))
        with open(json_path, "r") as f:
            self.process_pump_ordered_features = json.load(f)
        
        json_features = self.process_pump_ordered_features

        self.process_pump_feature_index = [json_features.index(name) for name in scaler_features]
        if sorted(scaler_features) != sorted(json_features):
            raise RuntimeError("Process Pump scaler feature list does NOT match JSON!")
        
        
        if not isinstance(self.process_pump_ordered_features, list) or len(self.process_pump_ordered_features) != 35:
            self.get_logger().error(f"Selected feature list invalid. Expected 35, got {len(self.process_pump_ordered_features)}")
        else:
            self.get_logger().info(f"Loaded {len(self.process_pump_ordered_features)} ordered features.")

        means_path = os.path.join(
            self.root_path,
            "pump_sensor_rul_hrs_data",
            "diagnostics_out",
            "selected_cols_top35_means.json"
        )
        with open(means_path, "r") as f:
            self.process_pump_feature_defaults = json.load(f)

        self.get_logger().info("Process_Pump Model & Scaler & Ordered List loaded.")
    
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
            
            self.process_pump_generate_predictions(received_data)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"User Input JSON parse error: {e}")
    
    def process_pump_generate_predictions(self, received_data: dict):
        """
        Generate RUL predictions for Process_Pump machine.
        Ensures unit consistency: model predicts rul_minutes/MAX_RUL_MINUTES.
        """
        if len(self.pump_history) < self.pump_rolling_window:
            self.get_logger().warning(f"Not enough history: {len(self.pump_history)}/{self.pump_rolling_window}.")
            return None

        feature_vec = self.process_pump_feature_extractor()
        if feature_vec is None:
            return None
        
        feature_vec = feature_vec[:, self.process_pump_feature_index]
        X_scaled = self.process_pump_scaler.transform(feature_vec)

        try:
            y_hat_norm = float(self.process_pump_model.predict(X_scaled)[0])
            
            MAX_RUL_MINUTES = 50249.0 # max rul in training in minutes
            
            # Model output: rul_minutes / MAX_RUL_MINUTES
            rul_minutes = float(np.clip(y_hat_norm, 0.0, 1.0)) * MAX_RUL_MINUTES
            
            # Convert minutes to hours (compatible with Hydraulic Press)
            rul_hrs = rul_minutes / 60.0
            
            # Physical constraint: Remaining RUL cannot exceed maximum lifetime elapsed
            cycle = received_data.get('cycle', len(self.pump_history) - 1)
            elapsed_minutes = float(cycle)
            elapsed_hours = elapsed_minutes / 60.0
            max_remaining = (MAX_RUL_MINUTES / 60.0) - elapsed_hours
            rul_hrs = float(np.clip(rul_hrs, 0.0, max_remaining))
            
            rul = round(rul_hrs, 3)

            if rul < 0:
                self.get_logger().warning(f"Predicted negative RUL ({rul} hrs), setting to 0.")
                rul = 0.0

        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
        
        cycle = received_data.get('cycle', len(self.pump_history) - 1)
        elapsed_hours = received_data.get('elapsed_hours', float(cycle) / 60.0)
        
        self.get_logger().info(
                                f"[Process Pump] Cycle: {cycle} min | "
                                f"Elapsed: {elapsed_hours:.2f}h | "
                                f"Predicted RUL: {rul:.2f}h | "
                                f"Norm output: {y_hat_norm:.6f}"
                            )
        
        self.publish_generated_data(rul, cycle=cycle, machine="process_pump")
    # Helper
    def process_pump_feature_extractor(self):
        """
        Feature extraction for Process_Pump machine sensor data.
        Ensures age_ratio matches training: elapsed_hours / MAX_RUL_HOURS
        """
        history = self.pump_history.copy()
        features = {}
        window = self.pump_rolling_window

        if len(history) < window:
            self.get_logger().warning("Not enough history for Process_Pump feature extraction.")
            return None
        
        # Training dataset constant
        MAX_RUL_HOURS = 837.483  # matching training
    
        # Time features from history
        time_features = {'elapsed_minutes', 'elapsed_hours', 'age_ratio'}
        required_base_sensors = set()
        
        for feat in self.process_pump_ordered_features:
            base_name = feat.replace('_roll_mean', '').replace('_roll_std', '')
            if base_name not in time_features:
                required_base_sensors.add(base_name)

        # Extract time sequences
        elapsed_minutes_seq = []
        elapsed_hours_seq = []
        age_ratio_seq = []
        
        for cycle_data in history:
            # Cycle = in minutes
            cycle_num = cycle_data.get('cycle', 0)
            elapsed_min = cycle_data.get('elapsed_minutes', float(cycle_num))
            elapsed_hrs = cycle_data.get('elapsed_hours', elapsed_min / 60.0)
            
            # age_ratio = elapsed_hours / MAX_RUL_HOURS (same as training)
            age_ratio = float(np.clip(elapsed_hrs / MAX_RUL_HOURS, 1e-6, 1.0))
            
            elapsed_minutes_seq.append(float(elapsed_min))
            elapsed_hours_seq.append(float(elapsed_hrs))
            age_ratio_seq.append(age_ratio)

        def last_roll(seq):
            """Calculate rolling mean and std for last window."""
            if len(seq) < window:
                return seq[-1], 0.0
            series = pd.Series(seq, dtype=float)
            rolling_mean = series.rolling(window=window, min_periods=1).mean().iloc[-1]
            rolling_std = series.rolling(window=window, min_periods=1).std(ddof=1).iloc[-1]
            if pd.isna(rolling_std):
                rolling_std = 0.0
            return float(rolling_mean), float(rolling_std)

        # Calculate rolling features for time variables
        elapsed_min_mean, elapsed_min_std = last_roll(elapsed_minutes_seq)
        elapsed_hrs_mean, elapsed_hrs_std = last_roll(elapsed_hours_seq)
        age_ratio_mean, age_ratio_std = last_roll(age_ratio_seq)

        # Add time features in order
        if 'elapsed_minutes_roll_mean' in self.process_pump_ordered_features:
            features['elapsed_minutes_roll_mean'] = elapsed_min_mean
        if 'elapsed_minutes_roll_std' in self.process_pump_ordered_features:
            features['elapsed_minutes_roll_std'] = elapsed_min_std
        if 'elapsed_hours_roll_mean' in self.process_pump_ordered_features:
            features['elapsed_hours_roll_mean'] = elapsed_hrs_mean
        if 'elapsed_hours_roll_std' in self.process_pump_ordered_features:
            features['elapsed_hours_roll_std'] = elapsed_hrs_std
        if 'age_ratio_roll_mean' in self.process_pump_ordered_features:
            features['age_ratio_roll_mean'] = age_ratio_mean
        if 'age_ratio_roll_std' in self.process_pump_ordered_features:
            features['age_ratio_roll_std'] = age_ratio_std
        if 'age_ratio' in self.process_pump_ordered_features:
            features['age_ratio'] = age_ratio_seq[-1]

        # Sensor rolling features
        for sensor_name in required_base_sensors:
            all_values = []
            for cycle_data in history:
                val = cycle_data.get(sensor_name)
                if val is not None:
                    all_values.append(float(val))
                else:
                    default_val = self.process_pump_feature_defaults.get(sensor_name, 0.0)
                    all_values.append(default_val)

            if len(all_values) == 0:
                continue

            series = pd.Series(all_values, dtype=float)
            
            # Current value
            if sensor_name in self.process_pump_ordered_features:
                features[sensor_name] = float(series.iloc[-1])

            # Rolling mean and std
            roll_mean_name = f"{sensor_name}_roll_mean"
            roll_std_name = f"{sensor_name}_roll_std"

            if roll_mean_name in self.process_pump_ordered_features:
                current_roll_mean = series.rolling(window=window, min_periods=1).mean().iloc[-1]
                if pd.notna(current_roll_mean):
                    features[roll_mean_name] = float(current_roll_mean)

            if roll_std_name in self.process_pump_ordered_features:
                current_roll_std = series.rolling(window=window, min_periods=1).std(ddof=1).iloc[-1]
                if pd.isna(current_roll_std):
                    current_roll_std = 0.0
                features[roll_std_name] = float(current_roll_std)

        # Arrange features in expected order
        X = []
        missing_feats = []
        for feat in self.process_pump_ordered_features:
            val = features.get(feat, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = self.process_pump_feature_defaults.get(feat, 0.0)
                missing_feats.append(feat)
            X.append(val)
        
        if missing_feats:
            self.get_logger().debug(f"Filled defaults for: {missing_feats[:5]}...")
        
        return np.array(X, dtype=np.float64).reshape(1, -1)
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