import os
import json
import math
import xgboost as xgb
import numpy as np

# CONFIGURATION -> will be updated when moved to RPI environment
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "../../../.."))
sources_path = os.path.join(root_path, "models_and_features", "process_pump")

# Global Containers
base_model = None
stg1_classifier = None
stg2a_regressor = None
stg2b_regressor = None
selected_features = []

# Buffer Limit: ~45-50 steps needed for acceleration calculation. 
# 60 is safe and lightweight.
MAX_BUFFER_SIZE = 60 

def load_models_and_features():
    """
    Loads XGBoost models and feature list into memory once.
    """
    global base_model, stg1_classifier, stg2a_regressor, stg2b_regressor, selected_features

    print("Loading Process Pump models (RPI Optimized)...")
    
    # Initialize models
    base_model      = xgb.XGBRegressor()
    stg1_classifier = xgb.XGBClassifier()
    stg2a_regressor = xgb.XGBRegressor()
    stg2b_regressor = xgb.XGBRegressor()
    
    # Load Weights
    base_model.load_model(os.path.join(sources_path, "process_pump_base_model.json"))
    stg1_classifier.load_model(os.path.join(sources_path, "process_pump_stage1_classifier.json"))
    stg2a_regressor.load_model(os.path.join(sources_path, "process_pump_stage2a_regressor.json"))
    stg2b_regressor.load_model(os.path.join(sources_path, "process_pump_stage2b_regressor.json"))

    # Load Feature List
    with open(os.path.join(sources_path, "selected_process_pump_features.json"), "r") as f:
        selected_features = json.load(f)

    print("Models loaded successfully.")

# HELPER FUNCTIONS (Instead of Pandas for RPI efficiency)
def get_mean(data_list):
    """Calculates mean of a list."""
    if not data_list: return 0.0
    return sum(data_list) / len(data_list)

def get_std(data_list, mean_val=None):
    """Calculates standard deviation."""
    if len(data_list) < 2: return 0.0
    if mean_val is None: mean_val = get_mean(data_list)
    variance = sum((x - mean_val) ** 2 for x in data_list) / (len(data_list) - 1)
    return math.sqrt(variance)

def get_rolling_stats(full_list, window):
    """Returns the slice for the window and its mean."""
    if len(full_list) < 1: return [], 0.0
    # If not enough data, take what there is
    start_idx = max(0, len(full_list) - window)
    window_data = full_list[start_idx:]
    return window_data, get_mean(window_data)


def engineer_features(history):
    """
    Performs Feature Engineering using pure Python lists (No Pandas).
    Replicates the exact logic from Training Notebook.
    """
    if not history:
        return None

    # Prepare Data Streams (Extract columns to lists)
    # Only keep the last N records
    buffer = history[-MAX_BUFFER_SIZE:] if len(history) > MAX_BUFFER_SIZE else history
    
    # Extract raw sensors into simple lists for fast looping
    vibration = [d.get('vibration', 0) for d in buffer]
    temprature = [d.get('temprature_motor', 0) for d in buffer]
    pressure= [d.get('pressure', 0) for d in buffer]
    vibration_motor = [d.get('vibration_motor', 0) for d in buffer]
    
    # Current values (Last item)
    current_vibration = vibration[-1]
    current_temprature = temprature[-1]
    current_pressure = pressure[-1]
    current_vibration_motor = vibration_motor[-1]
    
    # Initialize the feature row
    features = {}
    
    # Base Time Feature
    features['time_min'] = float(buffer[-1].get('elapsed_minutes', 0))

    # Constants from Training
    sensors = {'vibration': vibration, 'temprature_motor': temprature, 'pressure': pressure, 'vibration_motor': vibration_motor}
    WINDOW_SHORT = 5
    WINDOW_LONG = 20

    # Rolling Window Calculations
    for name, data in sensors.items():
        # Rolling Mean Short (5) 
        _, mean_short = get_rolling_stats(data, WINDOW_SHORT)
        features[f"{name}_roll_mean_{WINDOW_SHORT}"] = mean_short

        # Rolling Mean Long (20) 
        _, mean_long = get_rolling_stats(data, WINDOW_LONG)
        features[f"{name}_roll_mean_{WINDOW_LONG}"] = mean_long
        
        # Rolling Std Long (20) 
        slice_long = data[-WINDOW_LONG:] if len(data) >= WINDOW_LONG else data
        std_long = get_std(slice_long, mean_long)
        features[f"{name}_roll_std_{WINDOW_LONG}"] = std_long

        # Deviation 
        # Current Value - Long Mean
        features[f"{name}_dev_long"] = data[-1] - mean_long
        
        # Slope (Trend) 
        # Logic: Current Mean(20) - Mean(20) from 20 steps ago
        # If < 40, estimate or return 0.
        if len(data) >= (2 * WINDOW_LONG):
            # Previous window: [-40 : -20]
            prev_slice = data[-(2 * WINDOW_LONG) : -WINDOW_LONG]
            prev_mean = get_mean(prev_slice)
            slope = mean_long - prev_mean
        else:
            slope = 0.0 # Startup phase
        features[f"{name}_slope_{WINDOW_LONG}"] = slope

        # Acceleration 
        # Logic: Current Slope - Slope from 5 steps ago
        # Accel = Slope_Current - Slope_Prev
        if len(data) >= (2 * WINDOW_LONG + 5):
            # Window shifted back by 5
            current_mean_delayed = get_mean(data[-(WINDOW_LONG+5) : -5])
            previous_mean_delayed = get_mean(data[-(2*WINDOW_LONG+5) : -(WINDOW_LONG+5)])
            slope_delayed = current_mean_delayed - previous_mean_delayed
            acceleration = slope - slope_delayed
        else:
            acceleration = 0.0
        
        # Only specific sensors need acceleration per training (vibration, temprature)
        if name in ['vibrationration', 'temprature_motor']:
            features[f"{name}_acceleration"] = acceleration

        # Volatility Surge 
        # Std(5) - Std(20)
        slice_short = data[-WINDOW_SHORT:] if len(data) >= WINDOW_SHORT else data
        std_short = get_std(slice_short, mean_short)
        features[f"{name}_volatility_surge"] = std_short - std_long
        
        #  Recent Max (10) 
        if name in ['vibrationration', 'vibration_motor']:
            slice_10 = data[-10:] if len(data) >= 10 else data
            features[f"{name}_max_10"] = max(slice_10) if slice_10 else 0.0

    # Interaction Features
    # Motor Pump Coupling
    if abs(current_vibration) > 1e-6:
        coupling = current_vibration_motor / current_vibration
    else:
        coupling = 1.0
    # Clip to [0, 10]
    features['motor_pump_coupling'] = max(0.0, min(10.0, coupling))

    # Energy
    features['vibration_energy'] = features[f"vibrationration_roll_mean_{WINDOW_SHORT}"] ** 2

    # Pressure Stability (Inverse of std)
    p_std = features['pressure_roll_std_20']
    features['pressure_stability'] = 1.0 / (p_std + 1e-6)

    # Degradation Index (Normalized)
    # Using local min/max of the buffer for context
    v_min, v_max = min(vibration), max(vibration)
    t_min, t_max = min(temprature), max(temprature)
    p_min, p_max = min(pressure), max(pressure)

    def normalize(val, vmin, vmax):
        div = (vmax - vmin) + 1e-6
        return (val - vmin) / div

    v_n = normalize(current_vibration, v_min, v_max)
    t_n = normalize(current_temprature, t_min, t_max)
    p_n = 1 - normalize(current_pressure, p_min, p_max) # Inverse for pressure usually

    features['degradation_index'] = (v_n + t_n + p_n) / 3.0
    # Feature Selection & vectorization
    input_vector = []
    
    for feature_name in selected_features:
        # Get value, default to 0.0 if calculation failed or missing
        val = features.get(feature_name, 0.0)
        input_vector.append(val)
        
    # Convert to Numpy array shape (1, N) for XGBoost
    return np.array([input_vector], dtype=np.float32)


def get_prediction(history):
    """
    Main Interface for Predictor Node.
    """
    # Load Resources (First run only)
    if base_model is None:
        load_models_and_features()

    try:
        # Engineer Features
        X_input = engineer_features(history)
        
        if X_input is None:
            return -1.0

        # Inference Pipeline
        
        # Stage 1: Critical Classifier
        # proba returns [[prob_0, prob_1]]
        crit_prob = stg1_classifier.predict_proba(X_input)[0][1]    # type: ignore
        is_critical = crit_prob >= 0.30

        # Base Model
        rul = float(base_model.predict(X_input)[0])                 # type: ignore 

        # Stage 2: Refinement
        if is_critical:
            # Stage 2A
            rul_2a = float(stg2a_regressor.predict(X_input)[0])     # type: ignore
            rul = rul_2a
            
            # Stage 2B (Ultra Critical)
            if rul_2a <= 20:
                rul_2b = float(stg2b_regressor.predict(X_input)[0]) # type: ignore
                rul = rul_2b

        # Safety Bounds
        return max(0.0, min(rul, 48000.0))

    except Exception as e:
        print(f"Error in Process Pump Handler: {e}")
        return -1.0