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

# MODEL OUTPUT UNIT CONTRACT
# System contract: RUL is published in minutes (1 cycle = 1 minute).
# Training notebook: Stage-2A/2B regressors are trained on `current_rul` which is in minutes.
# Keep everything consistent: treat all model outputs as minutes.
STAGE_2A_OUTPUT_IS_HOURS = False
STAGE_2B_OUTPUT_IS_HOURS = False

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
    last = entries[-1]

    # Notebook constants
    window_short = 5
    window_long = 20
    sensors = ['vibration', 'temp_motor', 'pressure', 'vib_motor']

    def _roll_mean(values: np.ndarray, window: int) -> np.ndarray:
        out = np.empty_like(values, dtype=float)
        for i in range(len(values)):
            start = max(0, i - window + 1)
            out[i] = float(np.mean(values[start:i+1]))
        return out

    def _roll_std(values: np.ndarray, window: int, min_periods: int = 2) -> np.ndarray:
        out = np.zeros(len(values), dtype=float)
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start:i+1]
            if len(chunk) < min_periods:
                out[i] = 0.0
            else:
                # pandas std() defaults to ddof=1
                out[i] = float(np.std(chunk, ddof=1))
        return out

    def _diff(values: np.ndarray, periods: int) -> np.ndarray:
        out = np.zeros(len(values), dtype=float)
        for i in range(len(values)):
            if i - periods >= 0:
                out[i] = float(values[i] - values[i - periods])
            else:
                out[i] = 0.0
        return out

    series = {s: np.array([float(e[s]) for e in entries], dtype=float) for s in sensors}

    roll_mean_5 = {s: _roll_mean(series[s], window_short) for s in sensors}
    roll_mean_20 = {s: _roll_mean(series[s], window_long) for s in sensors}
    roll_std_20 = {s: _roll_std(series[s], window_long, min_periods=2) for s in sensors}

    # Deviation from long-term mean (notebook: raw - roll_mean_20)
    dev_long = {s: (series[s] - roll_mean_20[s]) for s in sensors}

    # Slope (notebook: diff of roll_mean_20 with period=20)
    slope_20 = {s: _diff(roll_mean_20[s], window_long) for s in sensors}

    # Acceleration (notebook: diff of slope_20 with period=5) for vibration & temp_motor
    accel = {
                'vibration': _diff(slope_20['vibration'], 5),
                'temp_motor': _diff(slope_20['temp_motor'], 5),
            }

    # Volatility surge (notebook: roll_std_short(5) - roll_std_long(20))
    roll_std_5 = {s: _roll_std(series[s], window_short, min_periods=2) for s in sensors}
    volatility_surge = {s: (roll_std_5[s] - roll_std_20[s]) for s in sensors}

    # Degradation index (notebook: normalized using global min/max)
    vib_norm = (last['vibration'] - VIB_MIN) / (VIB_MAX - VIB_MIN + 1e-6)
    temp_norm = (last['temp_motor'] - TEMP_MIN) / (TEMP_MAX - TEMP_MIN + 1e-6)
    press_norm = (last['pressure'] - PRESS_MIN) / (PRESS_MAX - PRESS_MIN + 1e-6)
    degradation_index = (vib_norm + temp_norm + (1.0 - press_norm)) / 3.0

    # Coupling (notebook: vib_motor / vibration)
    with np.errstate(divide='ignore', invalid='ignore'):
        coupling = float(last['vib_motor']) / (float(last['vibration']) + 1e-6)
    if not np.isfinite(coupling):
        coupling = 1.0
    coupling = float(np.clip(coupling, 0.0, 10.0))

    # Pressure stability (notebook: 1 / pressure_roll_std_20, 0 -> 1e-6)
    pressure_std = float(roll_std_20['pressure'][-1])
    if pressure_std == 0.0:
        pressure_std = 1e-6
    pressure_stability = 1.0 / pressure_std

    features = {
                    # Raw sensors
                    'vibration': float(last['vibration']),
                    'temp_motor': float(last['temp_motor']),
                    'pressure': float(last['pressure']),

                    # Core engineered
                    'degradation_index': float(degradation_index),
                    'motor_pump_coupling': float(coupling),
                    'pressure_stability': float(pressure_stability),

                    # Rolling std (20)
                    'vibration_roll_std_20': float(roll_std_20['vibration'][-1]),
                    'temp_motor_roll_std_20': float(roll_std_20['temp_motor'][-1]),
                    'pressure_roll_std_20': float(roll_std_20['pressure'][-1]),

                    # Dev long
                    'vibration_dev_long': float(dev_long['vibration'][-1]),
                    'temp_motor_dev_long': float(dev_long['temp_motor'][-1]),
                    'vib_motor_dev_long': float(dev_long['vib_motor'][-1]),

                    # Slope 20
                    'temp_motor_slope_20': float(slope_20['temp_motor'][-1]),
                    'pressure_slope_20': float(slope_20['pressure'][-1]),

                    # Acceleration
                    'temp_motor_acceleration': float(accel['temp_motor'][-1]),
                    'vibration_acceleration': float(accel['vibration'][-1]),

                    # Volatility surge
                    'vibration_volatility_surge': float(volatility_surge['vibration'][-1]),
                    'temp_motor_volatility_surge': float(volatility_surge['temp_motor'][-1]),
                    'vib_motor_volatility_surge': float(volatility_surge['vib_motor'][-1]),
                    'pressure_volatility_surge': float(volatility_surge['pressure'][-1]),
                }

    input_vector = [float(features.get(feature_name, 0.0)) for feature_name in selected_features]
    return np.array([input_vector], dtype=np.float32)
    

def get_prediction(history):
    if base_model_session is None and base_model is None:
        load_models_and_features()

    try:
        X_input = engineer_features(history)
        if X_input is None:
            return -1.0

        stage = "BASE"
        crit_prob = 0.0
        is_critical = False

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

            # Stage 0: Base regressor (minutes)
            base_input = base_model_session.get_inputs()[0].name
            rul_out = base_model_session.run(None, {base_input: X_input})[0]
            rul = float(np.squeeze(rul_out))                                                    # type: ignore

            if is_critical:
                # Stage 2A
                s2a_input = stg2a_regressor_session.get_inputs()[0].name                        # type: ignore
                rul_short_out = stg2a_regressor_session.run(None, {s2a_input: X_input})[0]      # type: ignore
                rul_short_raw = float(np.squeeze(rul_short_out))                                # type: ignore
                rul_short_min = rul_short_raw * 60.0 if STAGE_2A_OUTPUT_IS_HOURS else rul_short_raw

                # Notebook routing: if Stage-1 says critical, use Stage-2A output.
                rul = rul_short_min
                stage = "STAGE_2A"

                # Stage 2B (very short horizon)
                if rul_short_min <= 20:
                    s2b_input = stg2b_regressor_session.get_inputs()[0].name                # type: ignore
                    rul_vshort_out = stg2b_regressor_session.run(None, {s2b_input: X_input})[0] # type: ignore
                    rul_vshort_raw = float(np.squeeze(rul_vshort_out))                      # type: ignore
                    rul = rul_vshort_raw * 60.0 if STAGE_2B_OUTPUT_IS_HOURS else rul_vshort_raw
                    stage = "STAGE_2B"
        
        else:
            # XGBoost fallback
            crit_prob = stg1_classifier.predict_proba(X_input)[0][1]                            # type: ignore
            is_critical = crit_prob >= 0.30
            rul = float(base_model.predict(X_input)[0])                                         # type: ignore

            if is_critical:
                rul_short_raw = float(stg2a_regressor.predict(X_input)[0])                      # type: ignore
                rul_short_min = rul_short_raw * 60.0 if STAGE_2A_OUTPUT_IS_HOURS else rul_short_raw

                rul = rul_short_min
                stage = "STAGE_2A"

                if rul_short_min <= 20:
                    rul_vshort_raw = float(stg2b_regressor.predict(X_input)[0])             # type: ignore
                    rul = rul_vshort_raw * 60.0 if STAGE_2B_OUTPUT_IS_HOURS else rul_vshort_raw
                    stage = "STAGE_2B"

        return {
            "rul_min": float(np.clip(float(rul), 0.0, 48000.0)),
            "stage": stage,
            "is_critical": is_critical,
            "crit_prob": float(crit_prob),
            "unit": "minutes"
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return -1.0