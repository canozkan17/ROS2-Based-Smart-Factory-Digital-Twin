import os
import json
import numpy as np
import xgboost as xgb
import onnxruntime as ort

# IMPLEMENT TYPE CONTROL LATER !! 


# CONFIGURATION -> will be updated when moved to RPI environment
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "../../../.."))
sources_path = os.path.join(root_path, "models_and_features", "process_pump")

# Global Containers
# For ONNX
base_model_session = None
stg1_classifier_session = None
stg2a_regressor_session = None
stg2b_regressor_session = None

# Fallback to XGB
base_model = None
stg1_classifier = None
stg2a_regressor = None
stg2b_regressor = None

# Feature List: loaded from disk once on initialization
selected_features = []

# Buffer Limit: ~45-50 steps needed for acceleration calculation. 
# 60 is safe and lightweight.
MAX_BUFFER_SIZE = 60 

# GLOBAL SCALING LIMITS (limits from generate_pump_data.py)
# representing the space the models "recognize"
VIB_MIN = 0.02
VIB_MAX = 6.0
TEMP_MIN = 40.0
TEMP_MAX = 110.0 
PRESS_MIN = 4.0
PRESS_MAX = 8.5

def load_models_and_features():
    """
    Loads ONNX models as primary
    fallback to XGBoost models and feature list into memory once.
    """
    global base_model_session, stg1_classifier_session, stg2a_regressor_session, stg2b_regressor_session, selected_features, base_model, stg1_classifier, stg2a_regressor, stg2b_regressor

    print("Loading Process Pump models (RPI Optimized)...")

    try:
        # Initialize ONNX Runtime sessions
        base_model_session = ort.InferenceSession(os.path.join(sources_path, "process_pump_base_model.onnx"))
        stg1_classifier_session = ort.InferenceSession(os.path.join(sources_path, "process_pump_stage1_classifier.onnx"))
        stg2a_regressor_session = ort.InferenceSession(os.path.join(sources_path, "process_pump_stage2a_regressor.onnx"))
        stg2b_regressor_session = ort.InferenceSession(os.path.join(sources_path, "process_pump_stage2b_regressor.onnx"))
        print("ONNX models loaded successfully.")

    except Exception as e:
        print(f"ONNX model loading failed: {e}")
        print("Falling back to XGBoost models...")

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

    print("Models and features have been loaded successfully.")


def engineer_features(history):
    """
    Performs Feature Engineering using pure Python lists for RPI environment performance concerns.
    Replicates the exact logic from Training Notebook.
    """
    if len(history) < 10:
        return None
    
    entries = list(history)
    last_entry = entries[-1]
    
    # physical limits check (based on training data)
    vib_norm = (last_entry['vibration'] - VIB_MIN) / (VIB_MAX - VIB_MIN + 1e-6)
    temp_norm = (last_entry['temp_motor'] - TEMP_MIN) / (TEMP_MAX - TEMP_MIN + 1e-6)
    press_norm = (last_entry['pressure'] - PRESS_MIN) / (PRESS_MAX - PRESS_MIN + 1e-6)

    # Degradation Index (to catch the trend)
    degradation_index = (vib_norm + temp_norm + (1 - press_norm)) / 3.0
    
    # Rolling Features
    vibration_values = [entry['vibration'] for entry in entries]
    temp_values = [entry['temp_motor'] for entry in entries]
    
    # standard deviation of last 20 (Volatility)
    vib_std_20 = np.std(vibration_values[-20:]) if len(vibration_values) >= 20 else np.std(vibration_values)
    
    # Slope calculation
    def get_slope(data_list):
        if len(data_list) < 5: return 0.0
        y = np.array(data_list[-10:])
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    # Feature set expected by the model
    features = {
                    "vibration": last_entry['vibration'],
                    "temp_motor": last_entry['temp_motor'],
                    "pressure": last_entry['pressure'],
                    "degradation_index": degradation_index,
                    "vibration_roll_std_20": vib_std_20,
                    "temp_motor_slope_20": get_slope(temp_values),
                    "vibration_volatility_surge": vib_std_20 / (np.mean(vibration_values) + 1e-6),
                    "pressure_stability": np.std([e['pressure'] for e in entries[-10:]]),
                    "vib_motor_dev_long": last_entry['vib_motor'] - np.mean([e['vib_motor'] for e in entries]),
                    "vibration_dev_long": last_entry['vibration'] - np.mean(vibration_values),
                    "motor_pump_coupling": last_entry['vibration'] / (last_entry['vib_motor'] + 1e-6)
                }

    # Fill missing features with default (0.0)
    input_vector = []
    for feature_name in selected_features:
        input_vector.append(features.get(feature_name, 0.0))
        
    return np.array([input_vector], dtype=np.float32)
    

def get_prediction(history):
    if base_model_session is None and base_model is None:
        load_models_and_features()

    try:
        X_input = engineer_features(history)
        if X_input is None:
            return -1.0

        if base_model_session is not None:
            # Stage 1: Critical classifier
            stg1_input = stg1_classifier_session.get_inputs()[0].name                           # type: ignore
            crit_out = stg1_classifier_session.run(None, {stg1_input: X_input})[0]              # type: ignore
            
            # Robust check for classification output shape
            if crit_out.ndim > 1 and crit_out.shape[1] > 1:                                     # type: ignore
                crit_prob = float(crit_out[0][1])                                               # type: ignore
            else:
                crit_prob = float(np.squeeze(crit_out))                                         # type: ignore

            is_critical = crit_prob >= 0.30

            # Stage 0: Base regressor
            base_input = base_model_session.get_inputs()[0].name
            rul_out = base_model_session.run(None, {base_input: X_input})[0]
            rul = float(np.squeeze(rul_out))                                                    # type: ignore

            if is_critical:
                # Stage 2A
                s2a_input = stg2a_regressor_session.get_inputs()[0].name                        # type: ignore
                rul_short_out = stg2a_regressor_session.run(None, {s2a_input: X_input})[0]      # type: ignore
                rul_short = float(np.squeeze(rul_short_out))                                    # type: ignore

                if rul_short < 1000:
                    rul = rul_short

                # Stage 2B
                if rul_short <= 20:
                    s2b_input = stg2b_regressor_session.get_inputs()[0].name                    # type: ignore
                    rul_vshort_out = stg2b_regressor_session.run(None, {s2b_input: X_input})[0] # type: ignore
                    rul = float(np.squeeze(rul_vshort_out))                                     # type: ignore
        
        else:
            # XGBoost fallback
            crit_prob = stg1_classifier.predict_proba(X_input)[0][1]                            # type: ignore
            is_critical = crit_prob >= 0.30
            rul = float(base_model.predict(X_input)[0])                                         # type: ignore

            if is_critical:
                rul_short = float(stg2a_regressor.predict(X_input)[0])                          # type: ignore
                if rul_short < 100:
                    rul = rul_short
                if rul_short <= 20:
                    rul = float(stg2b_regressor.predict(X_input)[0])                            # type: ignore
        return max(0.0, float(rul))

    except Exception as e:
        print(f"Prediction Error: {e}")
        return -1.0