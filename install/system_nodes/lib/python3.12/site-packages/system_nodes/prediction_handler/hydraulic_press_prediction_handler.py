#!/usr/bin/env python3
"""
Hydraulic Press Prediction Handler

Multi-resolution RUL prediction pipeline for hydraulic press machine.

Architecture:
    Stage-0 Classifier: Regime detector (LONG_TERM vs NEAR_TERM)
        Threshold: RUL <= 50000 minutes -> NEAR_TERM
    
    Base Model: Long-term RUL tracker (XGBRegressor)
        Target: log1p(RUL_hours) where RUL_hours = RUL_minutes / 60
        Used for: General health trend monitoring in Long-Life scenario
    
    Stage-1 Classifier: Critical detector (NON_CRITICAL vs CRITICAL)
        Threshold: RUL <= 600 minutes -> CRITICAL
        Hysteresis: ENTER >= 0.6, EXIT <= 0.2
    
    Stage-2A Regressor: Critical RUL prediction
        Target: log1p(RUL_minutes)
        Range: 0 - 600 minutes
        Used for: High-resolution prediction in critical regime

Inference Flow:
    Long-Life Scenario: Stage-0 (regime) -> Base Model
    Short-Life Scenario: Stage-1 (critical check) -> Stage-2A

Optimized for Raspberry Pi deployment.
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


# Sensor-RUL lookup table for interpolation-based estimation
_sensor_lookup: Optional[Dict] = None


SENSOR_COLUMNS = [
    "hydraulic_pressure",
    "oil_temperature",
    "oil_contamination",
    "ram_position_deviation",
    "press_force",
    "vibration",
    "flow_rate",
    "motor_current"
]

WINDOW_SHORT = 5
WINDOW_LONG = 20

# SELECTED_FEATURES must match training notebook's selected_hydraulic_press_features.json EXACTLY
# These 20 features were selected during training via importance filtering
SELECTED_FEATURES = [
    "vibration_roll_std_20",
    "degradation_index",
    "vibration",
    "vibration_energy",
    "vibration_dev_long",
    "hydraulic_pressure_dev_long",
    "flow_rate_roll_mean_5",
    "oil_temperature_roll_std_20",
    "force_pressure_coupling",
    "press_force_roll_mean_5",
    "flow_rate_roll_mean_20",
    "hydraulic_pressure_roll_mean_5",
    "flow_rate",
    "motor_current_roll_mean_20",
    "oil_temperature_roll_mean_5",
    "oil_temperature_roll_mean_20",
    "vibration_volatility_surge",
    "press_force_roll_mean_20",
    "press_force",
    "motor_current"
]

# STAGE0_FEATURES: same as SELECTED_FEATURES (20 features)
STAGE0_FEATURES = SELECTED_FEATURES.copy()

# BASE_MODEL_FEATURES: selected_features minus roll_std/energy/volatility_surge (16 features)
BASE_MODEL_FEATURES = [
    "degradation_index",
    "vibration",
    "vibration_dev_long",
    "hydraulic_pressure_dev_long",
    "flow_rate_roll_mean_5",
    "force_pressure_coupling",
    "press_force_roll_mean_5",
    "flow_rate_roll_mean_20",
    "hydraulic_pressure_roll_mean_5",
    "flow_rate",
    "motor_current_roll_mean_20",
    "oil_temperature_roll_mean_5",
    "oil_temperature_roll_mean_20",
    "press_force_roll_mean_20",
    "press_force",
    "motor_current"
]

# STAGE2A_FEATURES: priority keywords from training (14 features)
# vibration, motor_current, oil_temperature, pressure, force_pressure
STAGE2A_FEATURES = [
    "vibration_roll_std_20",
    "vibration",
    "vibration_energy",
    "vibration_dev_long",
    "vibration_volatility_surge",
    "motor_current",
    "motor_current_roll_mean_20",
    "oil_temperature_roll_std_20",
    "oil_temperature_roll_mean_5",
    "oil_temperature_roll_mean_20",
    "hydraulic_pressure_roll_mean_5",
    "hydraulic_pressure_dev_long",
    "force_pressure_coupling",
    "press_force"
]

STAGE0_THRESHOLD_MIN = 50000
STAGE1_THRESHOLD_MIN = 600
STAGE2A_MAX_RUL_MIN = 600

THRESH_ENTER = 0.6
THRESH_EXIT = 0.2

MAX_BUFFER_SIZE = WINDOW_LONG + 10

_stage0_classifier = None
_base_model = None
_stage1_classifier = None
_stage2a_regressor = None

_feature_stats: Optional[Dict] = None
_data_buffer: List[Dict[str, float]] = []
_buffer_lock = threading.Lock()

_in_critical_state = False

_initialized = False


def _find_model_directory() -> Path:
    """Locate the models_and_features/hydraulic_press directory."""
    base_dir = Path(__file__).parent.resolve()
    
    for parent in [base_dir] + list(base_dir.parents):
        candidate = parent / "models_and_features" / "hydraulic_press"
        if candidate.exists():
            return candidate
    
    for parent in base_dir.parents:
        candidate = parent.parent.parent.parent.parent / "models_and_features" / "hydraulic_press"
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(f"Cannot locate models_and_features/hydraulic_press from {base_dir}")


def load_models() -> None:
    """
    Load all models and feature stats.
    Primary: ONNX Runtime (optimized for RPi)
    Fallback: Joblib/XGBoost
    """
    global _stage0_classifier, _base_model, _stage1_classifier, _stage2a_regressor
    global _feature_stats, _initialized, _sensor_lookup
    
    if _initialized:
        return
    
    model_dir = _find_model_directory()
    print(f"[hydraulic_press_handler] Loading models from: {model_dir}")
    
    stats_path = model_dir / "hydraulic_press_feature_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            _feature_stats = json.load(f)
        print("[hydraulic_press_handler] Loaded feature stats")
    
    # Load sensor-RUL lookup table for interpolation
    lookup_path = model_dir / "sensor_lifecycle_lookup.json"
    if lookup_path.exists():
        with open(lookup_path) as f:
            _sensor_lookup = json.load(f)
        print(f"[hydraulic_press_handler] Loaded sensor lookup table with {len(_sensor_lookup.get('lookup', []))} entries")
    
    try:
        import onnxruntime as ort
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ["CPUExecutionProvider"]
        
        _stage0_classifier = ort.InferenceSession(
            str(model_dir / "hydraulic_press_stage0_classifier.onnx"),
            sess_options=opts,
            providers=providers
        )
        _base_model = ort.InferenceSession(
            str(model_dir / "hydraulic_press_base_model.onnx"),
            sess_options=opts,
            providers=providers
        )
        _stage1_classifier = ort.InferenceSession(
            str(model_dir / "hydraulic_press_stage1_classifier.onnx"),
            sess_options=opts,
            providers=providers
        )
        _stage2a_regressor = ort.InferenceSession(
            str(model_dir / "hydraulic_press_stage2a_regressor.onnx"),
            sess_options=opts,
            providers=providers
        )
        _initialized = True
        print("[hydraulic_press_handler] ONNX models loaded successfully")
        return
        
    except ImportError:
        print("[hydraulic_press_handler] ONNX Runtime not available, trying joblib/XGBoost")
    except Exception as e:
        print(f"[hydraulic_press_handler] ONNX load failed: {e}, trying joblib/XGBoost")
    
    try:
        import joblib
        
        _stage0_classifier = joblib.load(str(model_dir / "hydraulic_press_stage0_classifier.pkl"))
        _base_model = joblib.load(str(model_dir / "hydraulic_press_base_model.pkl"))
        _stage1_classifier = joblib.load(str(model_dir / "hydraulic_press_stage1_classifier.pkl"))
        _stage2a_regressor = joblib.load(str(model_dir / "hydraulic_press_stage2a_regressor.pkl"))
        
        _initialized = True
        print("[hydraulic_press_handler] Joblib models loaded successfully")
        return
        
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")


def _add_to_buffer(sensor_data: Dict[str, Any]) -> None:
    """Add new sensor reading to rolling buffer."""
    global _data_buffer
    
    with _buffer_lock:
        entry = {
            "cycle": int(sensor_data.get("cycle", 0)),
            "hydraulic_pressure": float(sensor_data.get("hydraulic_pressure", 0.0)),
            "oil_temperature": float(sensor_data.get("oil_temperature", 0.0)),
            "oil_contamination": float(sensor_data.get("oil_contamination", 0.0)),
            "ram_position_deviation": float(sensor_data.get("ram_position_deviation", 0.0)),
            "press_force": float(sensor_data.get("press_force", 0.0)),
            "vibration": float(sensor_data.get("vibration", 0.0)),
            "flow_rate": float(sensor_data.get("flow_rate", 0.0)),
            "motor_current": float(sensor_data.get("motor_current", 0.0))
        }
        _data_buffer.append(entry)
        
        if len(_data_buffer) > MAX_BUFFER_SIZE:
            _data_buffer = _data_buffer[-MAX_BUFFER_SIZE:]


def reset_state() -> None:
    """Reset all state for new job or after maintenance."""
    global _data_buffer, _in_critical_state, _last_seen_cycle
    
    with _buffer_lock:
        old_critical = _in_critical_state
        old_last_cycle = _last_seen_cycle
        _data_buffer = []
        _in_critical_state = False
        _last_seen_cycle = -1
    
    print(f"[TEMP:DEBUG] reset_state() called: _in_critical_state {old_critical} -> False, _last_seen_cycle {old_last_cycle} -> -1")
    print("[hydraulic_press_handler] State reset")


_last_seen_cycle: int = -1


def _sync_buffer_from_history(history: List[Dict[str, Any]]) -> None:
    """Sync internal buffer from Predictor_Node history."""
    global _data_buffer, _in_critical_state, _last_seen_cycle
    
    with _buffer_lock:
        recent = history[-MAX_BUFFER_SIZE:] if len(history) > MAX_BUFFER_SIZE else history
        
        if len(recent) > 0:
            first_cycle = int(recent[0].get("cycle", 0))
            last_cycle = int(recent[-1].get("cycle", 0))
            
            print(f"[TEMP:DEBUG] _sync_buffer: first_cycle={first_cycle}, last_cycle={last_cycle}, "
                  f"_last_seen_cycle={_last_seen_cycle}, _in_critical_state={_in_critical_state}")
            
            if first_cycle < 100 and _last_seen_cycle > 1000:
                print(f"[TEMP:DEBUG] LIFECYCLE RESET DETECTED! first_cycle={first_cycle} < 100, "
                      f"_last_seen_cycle={_last_seen_cycle} > 1000 -> setting _in_critical_state=False")
                print(f"[hydraulic_press_handler] Detected lifecycle reset: "
                      f"first_cycle={first_cycle}, last_seen={_last_seen_cycle}")
                _in_critical_state = False
            
            _last_seen_cycle = last_cycle
        
        _data_buffer = []
        for d in recent:
            entry = {
                "cycle": int(d.get("cycle", 0)),
                "hydraulic_pressure": float(d.get("hydraulic_pressure", 0.0)),
                "oil_temperature": float(d.get("oil_temperature", 0.0)),
                "oil_contamination": float(d.get("oil_contamination", 0.0)),
                "ram_position_deviation": float(d.get("ram_position_deviation", 0.0)),
                "press_force": float(d.get("press_force", 0.0)),
                "vibration": float(d.get("vibration", 0.0)),
                "flow_rate": float(d.get("flow_rate", 0.0)),
                "motor_current": float(d.get("motor_current", 0.0))
            }
            _data_buffer.append(entry)


def _compute_features() -> Optional[Dict[str, np.ndarray]]:
    """
    Compute all features from buffer matching training EXACTLY.
    
    Returns dictionary with feature arrays for each model:
        - selected_features: Full feature set (20 features)
        - stage0_features: Same as selected (20 features)
        - base_model_features: Subset for base model (16 features)
        - stage2a_features: Subset for stage-2a (14 features)
    """
    with _buffer_lock:
        if len(_data_buffer) < WINDOW_SHORT:
            return None
        
        n = len(_data_buffer)
        
        # Extract sensor arrays
        hydraulic_pressure = np.array([d["hydraulic_pressure"] for d in _data_buffer])
        oil_temperature = np.array([d["oil_temperature"] for d in _data_buffer])
        oil_contamination = np.array([d["oil_contamination"] for d in _data_buffer])
        ram_position_deviation = np.array([d["ram_position_deviation"] for d in _data_buffer])
        press_force = np.array([d["press_force"] for d in _data_buffer])
        vibration = np.array([d["vibration"] for d in _data_buffer])
        flow_rate = np.array([d["flow_rate"] for d in _data_buffer])
        motor_current = np.array([d["motor_current"] for d in _data_buffer])
        
        # Latest values
        latest_hydraulic_pressure = hydraulic_pressure[-1]
        latest_oil_temperature = oil_temperature[-1]
        latest_oil_contamination = oil_contamination[-1]
        latest_ram_position_deviation = ram_position_deviation[-1]
        latest_press_force = press_force[-1]
        latest_vibration = vibration[-1]
        latest_flow_rate = flow_rate[-1]
        latest_motor_current = motor_current[-1]
        
        window_short = min(WINDOW_SHORT, n)
        window_long = min(WINDOW_LONG, n)
        
        def rolling_mean(arr, window):
            return float(np.mean(arr[-window:]))
        
        def rolling_std(arr, window):
            if len(arr) < 2:
                return 0.0
            window_data = arr[-window:]
            return float(np.std(window_data, ddof=1)) if len(window_data) > 1 else 0.0
        
        # Rolling mean features (window 5)
        hydraulic_pressure_roll_mean_5 = rolling_mean(hydraulic_pressure, window_short)
        oil_temperature_roll_mean_5 = rolling_mean(oil_temperature, window_short)
        press_force_roll_mean_5 = rolling_mean(press_force, window_short)
        flow_rate_roll_mean_5 = rolling_mean(flow_rate, window_short)
        vibration_roll_mean_5 = rolling_mean(vibration, window_short)
        
        # Rolling mean features (window 20)
        hydraulic_pressure_roll_mean_20 = rolling_mean(hydraulic_pressure, window_long)
        oil_temperature_roll_mean_20 = rolling_mean(oil_temperature, window_long)
        press_force_roll_mean_20 = rolling_mean(press_force, window_long)
        flow_rate_roll_mean_20 = rolling_mean(flow_rate, window_long)
        motor_current_roll_mean_20 = rolling_mean(motor_current, window_long)
        vibration_roll_mean_20 = rolling_mean(vibration, window_long)
        
        # Rolling std features (window 20)
        vibration_roll_std_20 = rolling_std(vibration, window_long)
        oil_temperature_roll_std_20 = rolling_std(oil_temperature, window_long)
        
        # Dev long features (deviation from long-term mean)
        vibration_dev_long = latest_vibration - vibration_roll_mean_20
        hydraulic_pressure_dev_long = latest_hydraulic_pressure - hydraulic_pressure_roll_mean_20
        
        # Energy features: vibration_roll_mean_5 ** 2
        vibration_energy = vibration_roll_mean_5 ** 2
        
        # Volatility surge features: max(0, short_std - long_std)
        vibration_short_std = rolling_std(vibration, window_short)
        vibration_volatility_surge = max(0.0, vibration_short_std - vibration_roll_std_20)
        
        # Force-pressure coupling
        if latest_hydraulic_pressure > 0:
            force_pressure_coupling = latest_press_force / latest_hydraulic_pressure
            force_pressure_coupling = float(np.clip(force_pressure_coupling, 0, 10))
        else:
            force_pressure_coupling = 1.0
        
        # Degradation index: (vib_norm + temp_norm + (1 - pressure_norm) + contam_norm) / 4.0
        # CRITICAL FIX: Use GLOBAL ranges from feature_stats instead of buffer-local min/max
        # Training used cycle-level normalization (future knowledge), so we approximate with global ranges
        eps = 1e-9
        
        # Get global normalization ranges from feature_stats
        global_ranges = None
        if _feature_stats and "degradation_index_global_ranges" in _feature_stats:
            global_ranges = _feature_stats["degradation_index_global_ranges"]
        
        def normalize_global(value, sensor_name):
            """Normalize using global ranges from training data distribution."""
            if global_ranges is None:
                # Fallback to reasonable defaults based on training data analysis
                defaults = {
                    "vibration": (0.1, 15.0),
                    "oil_temperature": (45.0, 850.0),
                    "hydraulic_pressure": (70.0, 220.0),
                    "oil_contamination": (1.5, 30.0)
                }
                g_min, g_max = defaults.get(sensor_name, (0.0, 1.0))
            else:
                g_min = global_ranges.get(f"{sensor_name}_min", 0.0)
                g_max = global_ranges.get(f"{sensor_name}_max", 1.0)
            
            if g_max - g_min < eps:
                return 0.5
            # Clip to [0, 1] range after normalization
            normalized = (value - g_min) / (g_max - g_min + eps)
            return float(np.clip(normalized, 0.0, 1.0))
        
        vib_norm = normalize_global(latest_vibration, "vibration")
        temp_norm = normalize_global(latest_oil_temperature, "oil_temperature")
        pressure_norm = normalize_global(latest_hydraulic_pressure, "hydraulic_pressure")
        contam_norm = normalize_global(latest_oil_contamination, "oil_contamination")
        
        degradation_index = (vib_norm + temp_norm + (1.0 - pressure_norm) + contam_norm) / 4.0
        
        # Build feature dictionary
        feature_dict = {
            "vibration": latest_vibration,
            "flow_rate": latest_flow_rate,
            "press_force": latest_press_force,
            "motor_current": latest_motor_current,
            "vibration_roll_std_20": vibration_roll_std_20,
            "oil_temperature_roll_std_20": oil_temperature_roll_std_20,
            "vibration_dev_long": vibration_dev_long,
            "hydraulic_pressure_dev_long": hydraulic_pressure_dev_long,
            "vibration_energy": vibration_energy,
            "vibration_volatility_surge": vibration_volatility_surge,
            "flow_rate_roll_mean_5": flow_rate_roll_mean_5,
            "press_force_roll_mean_5": press_force_roll_mean_5,
            "hydraulic_pressure_roll_mean_5": hydraulic_pressure_roll_mean_5,
            "oil_temperature_roll_mean_5": oil_temperature_roll_mean_5,
            "flow_rate_roll_mean_20": flow_rate_roll_mean_20,
            "motor_current_roll_mean_20": motor_current_roll_mean_20,
            "oil_temperature_roll_mean_20": oil_temperature_roll_mean_20,
            "press_force_roll_mean_20": press_force_roll_mean_20,
            "force_pressure_coupling": force_pressure_coupling,
            "degradation_index": degradation_index,
        }
        
        try:
            selected_feature_values = [feature_dict[f] for f in SELECTED_FEATURES]
            selected_features_array = np.array(selected_feature_values, dtype=np.float32).reshape(1, -1)
            
            stage0_feature_values = [feature_dict[f] for f in STAGE0_FEATURES]
            stage0_features_array = np.array(stage0_feature_values, dtype=np.float32).reshape(1, -1)
            
            base_feature_values = [feature_dict[f] for f in BASE_MODEL_FEATURES]
            base_features_array = np.array(base_feature_values, dtype=np.float32).reshape(1, -1)
            
            stage2a_feature_values = [feature_dict[f] for f in STAGE2A_FEATURES]
            stage2a_features_array = np.array(stage2a_feature_values, dtype=np.float32).reshape(1, -1)
            
        except KeyError as e:
            print(f"[hydraulic_press_handler] Missing feature in dict: {e}")
            print(f"[hydraulic_press_handler] Available features: {list(feature_dict.keys())}")
            return None
        
        return {
            "selected_features": selected_features_array,
            "stage0_features": stage0_features_array,
            "base_model_features": base_features_array,
            "stage2a_features": stage2a_features_array
        }


def _run_stage0_onnx(X: np.ndarray) -> float:
    """Run Stage-0 regime classifier using ONNX."""
    input_name = _stage0_classifier.get_inputs()[0].name
    outputs = _stage0_classifier.run(None, {input_name: X})
    
    try:
        proba_array = np.asarray(outputs[1]).squeeze()
        if proba_array.ndim == 0:
            near_term_prob = float(proba_array)
        elif proba_array.size >= 2:
            near_term_prob = float(proba_array[1])
        else:
            near_term_prob = float(proba_array)
    except Exception:
        label = int(np.asarray(outputs[0]).squeeze())
        near_term_prob = float(label)
    
    return near_term_prob


def _run_stage0_sklearn(X: np.ndarray) -> float:
    """Run Stage-0 regime classifier using sklearn/XGBoost."""
    near_term_prob = float(_stage0_classifier.predict_proba(X)[0, 1])
    return near_term_prob


def _run_base_model_onnx(X: np.ndarray) -> float:
    """Run Base Model using ONNX. Returns RUL in minutes.
    
    Base Model outputs log1p(RUL_hours) where RUL_hours = RUL_minutes / 60.
    Decode: expm1(prediction) * 60 = RUL in minutes
    """
    input_name = _base_model.get_inputs()[0].name
    pred_log_hours = float(_base_model.run(None, {input_name: X})[0].squeeze())
    
    pred_hours = np.expm1(pred_log_hours)
    pred_minutes = pred_hours * 60.0
    
    max_rul = _feature_stats.get('base_model_max_rul', 711434.0) if _feature_stats else 711434.0
    pred_minutes = float(np.clip(pred_minutes, 0.0, max_rul))
    
    return pred_minutes


def _run_base_model_sklearn(X: np.ndarray) -> float:
    """Run Base Model using sklearn/XGBoost. Returns RUL in minutes."""
    pred_log_hours = float(_base_model.predict(X)[0])
    
    pred_hours = np.expm1(pred_log_hours)
    pred_minutes = pred_hours * 60.0
    
    max_rul = _feature_stats.get('base_model_max_rul', 711434.0) if _feature_stats else 711434.0
    pred_minutes = float(np.clip(pred_minutes, 0.0, max_rul))
    
    return pred_minutes


def _run_stage1_onnx(X: np.ndarray) -> float:
    """Run Stage-1 critical classifier using ONNX."""
    input_name = _stage1_classifier.get_inputs()[0].name
    outputs = _stage1_classifier.run(None, {input_name: X})
    
    try:
        proba_array = np.asarray(outputs[1]).squeeze()
        if proba_array.ndim == 0:
            critical_prob = float(proba_array)
        elif proba_array.size >= 2:
            critical_prob = float(proba_array[1])
        else:
            critical_prob = float(proba_array)
    except Exception:
        label = int(np.asarray(outputs[0]).squeeze())
        critical_prob = float(label)
    
    return critical_prob


def _run_stage1_sklearn(X: np.ndarray) -> float:
    """Run Stage-1 critical classifier using sklearn/XGBoost."""
    critical_prob = float(_stage1_classifier.predict_proba(X)[0, 1])
    return critical_prob


def _run_stage2a_onnx(X: np.ndarray) -> float:
    """Run Stage-2A critical regressor using ONNX. Returns RUL in minutes."""
    input_name = _stage2a_regressor.get_inputs()[0].name
    pred_log = float(_stage2a_regressor.run(None, {input_name: X})[0].squeeze())
    pred_minutes = np.expm1(pred_log)
    pred_minutes = float(np.clip(pred_minutes, 0.0, STAGE2A_MAX_RUL_MIN))
    return pred_minutes


def _run_stage2a_sklearn(X: np.ndarray) -> float:
    """Run Stage-2A critical regressor using sklearn/XGBoost. Returns RUL in minutes."""
    pred_log = float(_stage2a_regressor.predict(X)[0])
    pred_minutes = np.expm1(pred_log)
    pred_minutes = float(np.clip(pred_minutes, 0.0, STAGE2A_MAX_RUL_MIN))
    return pred_minutes


def _estimate_rul_from_sensors(sensor_data: Dict[str, Any]) -> Optional[float]:
    """
    Estimate RUL using sensor-based interpolation from training data.
    This provides a physics-informed estimate that doesn't suffer from
    the look-ahead bias in degradation_index.
    
    Returns RUL in minutes, or None if lookup table not available.
    """
    if _sensor_lookup is None or "lookup" not in _sensor_lookup:
        return None
    
    lookup = _sensor_lookup["lookup"]
    if len(lookup) < 2:
        return None
    
    # Get current sensor values
    vibration = float(sensor_data.get("vibration", 0.0))
    oil_temperature = float(sensor_data.get("oil_temperature", 0.0))
    hydraulic_pressure = float(sensor_data.get("hydraulic_pressure", 200.0))
    oil_contamination = float(sensor_data.get("oil_contamination", 0.0))
    
    # Extract arrays from lookup
    ruls = np.array([e["rul"] for e in lookup])
    vibs = np.array([e["vibration"] for e in lookup])
    temps = np.array([e["oil_temperature"] for e in lookup])
    pres = np.array([e["hydraulic_pressure"] for e in lookup])
    contams = np.array([e["oil_contamination"] for e in lookup])
    
    def estimate_from_sensor(sensor_value, sensor_array, direction="increasing"):
        """
        Estimate RUL from a single sensor value.
        direction: "increasing" = higher value means lower RUL (vibration, temp, contamination)
                   "decreasing" = lower value means lower RUL (pressure)
        """
        if direction == "increasing":
            if sensor_value <= sensor_array.min():
                return float(ruls.max())
            if sensor_value >= sensor_array.max():
                return float(ruls.min())
            # Find closest match by linear interpolation
            try:
                from scipy.interpolate import interp1d
                interp = interp1d(sensor_array, ruls, bounds_error=False, 
                                  fill_value=(float(ruls.max()), float(ruls.min())))
                return float(interp(sensor_value))
            except ImportError:
                # Fallback: nearest neighbor
                idx = np.argmin(np.abs(sensor_array - sensor_value))
                return float(ruls[idx])
        else:
            if sensor_value >= sensor_array.max():
                return float(ruls.max())
            if sensor_value <= sensor_array.min():
                return float(ruls.min())
            try:
                from scipy.interpolate import interp1d
                # Reverse arrays for decreasing relationship
                interp = interp1d(sensor_array[::-1], ruls[::-1], bounds_error=False,
                                  fill_value=(float(ruls.max()), float(ruls.min())))
                return float(interp(sensor_value))
            except ImportError:
                idx = np.argmin(np.abs(sensor_array - sensor_value))
                return float(ruls[idx])
    
    # Estimate RUL from each sensor
    rul_from_vib = estimate_from_sensor(vibration, vibs, "increasing")
    rul_from_temp = estimate_from_sensor(oil_temperature, temps, "increasing")
    rul_from_pres = estimate_from_sensor(hydraulic_pressure, pres, "decreasing")
    rul_from_contam = estimate_from_sensor(oil_contamination, contams, "increasing")
    
    # Weighted average - vibration and temperature are more reliable
    weights = [0.35, 0.35, 0.15, 0.15]
    weighted_rul = np.average([rul_from_vib, rul_from_temp, rul_from_pres, rul_from_contam], weights=weights)
    
    return float(weighted_rul)


def predict(sensor_data) -> Dict[str, Any]:
    """
    Main prediction function called by Predictor_Node.
    """
    global _in_critical_state
    
    if not _initialized:
        load_models()
    
    if isinstance(sensor_data, list):
        if len(sensor_data) == 0:
            return {
                "rul_min": -1.0,
                "unit": "min",
                "stage": "NO_DATA",
                "crit_prob": 0.0,
                "regime": "UNKNOWN"
            }
        _sync_buffer_from_history(sensor_data)
        current_data = sensor_data[-1]
    else:
        current_data = sensor_data
        _add_to_buffer(current_data)
    
    current_cycle = current_data.get("cycle", 0)
    
    vibration = current_data.get("vibration", -1)
    pressure = current_data.get("hydraulic_pressure", -1)
    oil_temp = current_data.get("oil_temperature", -1)
    print(f"[TEMP:DEBUG] predict() cycle={current_cycle}, vibration={vibration:.4f}, "
          f"pressure={pressure:.2f}, oil_temp={oil_temp:.2f}, _in_critical_state={_in_critical_state}")
    
    if current_cycle == 0:
        print(f"[TEMP:DEBUG] cycle==0 detected, calling reset_state()")
        reset_state()
        _add_to_buffer(current_data)
    
    feature_arrays = _compute_features()
    
    if feature_arrays is None:
        return {
            "rul_min": -1.0,
            "unit": "min",
            "stage": "INSUFFICIENT_DATA",
            "crit_prob": 0.0,
            "regime": "UNKNOWN"
        }
    
    X_selected = feature_arrays["selected_features"]
    X_stage0 = feature_arrays["stage0_features"]
    X_base = feature_arrays["base_model_features"]
    X_stage2a = feature_arrays["stage2a_features"]
    
    try:
        is_onnx = hasattr(_stage0_classifier, "run")
        
        if is_onnx:
            near_term_prob = _run_stage0_onnx(X_stage0)
        else:
            near_term_prob = _run_stage0_sklearn(X_stage0)
        
        regime = "NEAR_TERM" if near_term_prob >= 0.5 else "LONG_TERM"
        
        if is_onnx:
            critical_prob = _run_stage1_onnx(X_selected)
        else:
            critical_prob = _run_stage1_sklearn(X_selected)
        
        # SENSOR-BASED GUARD: Override Stage-1 classifier when sensors indicate healthy state
        # This compensates for the look-ahead bias in degradation_index during training
        # Thresholds based on training data analysis: healthy state has low degradation sensors
        vibration = current_data.get("vibration", 0.0)
        oil_temperature = current_data.get("oil_temperature", 0.0)
        oil_contamination = current_data.get("oil_contamination", 0.0)
        hydraulic_pressure = current_data.get("hydraulic_pressure", 200.0)
        
        # Conservative thresholds from training data percentiles:
        # - vibration p75 = 2.96, so healthy if < 3.0
        # - oil_temperature p50 = 175, so healthy if < 200
        # - oil_contamination p50 = 6.04, so healthy if < 7.0
        # - hydraulic_pressure p50 = 178, so healthy if > 165
        sensors_indicate_healthy = (
            vibration < 3.0 and 
            oil_temperature < 200.0 and 
            oil_contamination < 7.0 and 
            hydraulic_pressure > 165.0
        )
        
        if sensors_indicate_healthy:
            # Force BASE model usage when sensors clearly indicate healthy state
            effective_critical_prob = 0.0
            print(f"[TEMP:DEBUG] SENSOR GUARD: Overriding crit_prob={critical_prob:.4f} -> 0.0 "
                  f"(vib={vibration:.2f}<3.0, temp={oil_temperature:.1f}<200, contam={oil_contamination:.2f}<7, pres={hydraulic_pressure:.1f}>165)")
        else:
            # Additional sensor-based check: STAGE_2A should only be for truly critical state
            # Based on training data: RUL < 5000 typically has vibration > 4.0, temp > 270, pressure < 165
            sensors_indicate_critical = (
                vibration > 4.0 or 
                oil_temperature > 270.0 or 
                oil_contamination > 10.0 or
                hydraulic_pressure < 160.0
            )
            
            if sensors_indicate_critical:
                effective_critical_prob = critical_prob
            else:
                # Sensors in degraded but not critical range - reduce critical probability
                # to prefer BASE model with interpolation blending
                effective_critical_prob = min(critical_prob, 0.4)
                print(f"[TEMP:DEBUG] SENSOR MODERATE: Reducing crit_prob={critical_prob:.4f} -> {effective_critical_prob:.4f} "
                      f"(vib={vibration:.2f}, temp={oil_temperature:.1f}, contam={oil_contamination:.2f}, pres={hydraulic_pressure:.1f})")
        
        old_critical_state = _in_critical_state
        if not _in_critical_state:
            if effective_critical_prob >= THRESH_ENTER:
                _in_critical_state = True
                print(f"[TEMP:DEBUG] ENTERING CRITICAL: crit_prob={effective_critical_prob:.4f} >= THRESH_ENTER={THRESH_ENTER}")
        else:
            if effective_critical_prob <= THRESH_EXIT:
                _in_critical_state = False
                print(f"[TEMP:DEBUG] EXITING CRITICAL: crit_prob={effective_critical_prob:.4f} <= THRESH_EXIT={THRESH_EXIT}")
        
        if _in_critical_state:
            # Even in critical state, use sensor interpolation for better accuracy
            sensor_based_rul = _estimate_rul_from_sensors(current_data)
            
            if is_onnx:
                model_rul = _run_stage2a_onnx(X_stage2a)
            else:
                model_rul = _run_stage2a_sklearn(X_stage2a)
            
            if sensor_based_rul is not None:
                # In critical state, give more weight to interpolation since STAGE_2A
                # model also has bias issues, but cap at STAGE2A_MAX_RUL_MIN
                if sensor_based_rul < STAGE2A_MAX_RUL_MIN * 2:
                    # Actually close to critical - blend with higher interpolation weight
                    interp_weight = 0.7
                    model_weight = 0.3
                    final_rul = sensor_based_rul * interp_weight + model_rul * model_weight
                    final_rul = min(final_rul, STAGE2A_MAX_RUL_MIN)
                else:
                    # Not actually critical based on sensors - use interpolation directly
                    # but cap at reasonable maximum for critical state
                    final_rul = min(sensor_based_rul, STAGE2A_MAX_RUL_MIN * 50)  # ~30000 min
                print(f"[TEMP:DEBUG] STAGE_2A blending: interp={sensor_based_rul:.0f}, model={model_rul:.0f}, final={final_rul:.0f}")
            else:
                final_rul = model_rul
            stage = "STAGE_2A"
        else:
            # HYBRID APPROACH: Use sensor-based interpolation for better accuracy
            # The ONNX base model has look-ahead bias from degradation_index
            sensor_based_rul = _estimate_rul_from_sensors(current_data)
            
            if sensor_based_rul is not None and sensors_indicate_healthy:
                # When sensors clearly indicate healthy state, trust interpolation
                final_rul = sensor_based_rul
                stage = "BASE_INTERP"
                print(f"[TEMP:DEBUG] Using sensor interpolation: {final_rul:.0f} min")
            else:
                # Use ONNX model when sensors indicate degradation or lookup not available
                if is_onnx:
                    model_rul = _run_base_model_onnx(X_base)
                else:
                    model_rul = _run_base_model_sklearn(X_base)
                
                # Blend with sensor-based estimate if available
                if sensor_based_rul is not None:
                    # Weight based on how "healthy" the sensors look
                    # Use normalized sensor values to calculate degradation score
                    # Healthy ranges: vib < 1.0, temp < 100, pres > 190, contam < 3.0
                    # Degraded ranges: vib > 4.0, temp > 270, pres < 160, contam > 10.0
                    vib_score = min(1.0, max(0.0, (vibration - 1.0) / 3.0))
                    temp_score = min(1.0, max(0.0, (oil_temperature - 100.0) / 170.0))
                    pres_score = min(1.0, max(0.0, (190.0 - hydraulic_pressure) / 30.0))
                    contam_score = min(1.0, max(0.0, (oil_contamination - 3.0) / 7.0))
                    
                    degradation_score = (vib_score + temp_score + pres_score + contam_score) / 4.0
                    
                    # Blend: low degradation = more interpolation, high degradation = balanced
                    # Never fully trust model alone since it has bias issues
                    interp_weight = max(0.3, 1.0 - degradation_score * 0.7)
                    model_weight = 1.0 - interp_weight
                    
                    final_rul = sensor_based_rul * interp_weight + model_rul * model_weight
                    stage = f"BASE_BLEND({interp_weight:.1f})"
                    print(f"[TEMP:DEBUG] Blending: interp={sensor_based_rul:.0f}, model={model_rul:.0f}, "
                          f"deg_score={degradation_score:.2f}, weights=({interp_weight:.2f}, {model_weight:.2f}), final={final_rul:.0f}")
                else:
                    final_rul = model_rul
                    stage = "BASE"
        
        print(f"[TEMP:DEBUG] PREDICTION: cycle={current_cycle}, regime={regime}, near_term_prob={near_term_prob:.4f}, "
              f"crit_prob={critical_prob:.4f}, effective_crit={effective_critical_prob:.4f}, _in_critical={_in_critical_state}, stage={stage}, rul={final_rul:.2f}min")
        
        final_rul = max(0.0, final_rul)
        
        return {
            "rul_min": float(final_rul),
            "unit": "min",
            "stage": stage,
            "crit_prob": float(np.clip(effective_critical_prob, 0.0, 1.0)),
            "regime": regime
        }
        
    except Exception as e:
        print(f"[hydraulic_press_handler] Inference error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "rul_min": -1.0,
            "unit": "min",
            "stage": "ERROR",
            "crit_prob": 0.0,
            "regime": "UNKNOWN"
        }


try:
    load_models()
except Exception as e:
    print(f"[hydraulic_press_handler] Deferred model loading: {e}")
