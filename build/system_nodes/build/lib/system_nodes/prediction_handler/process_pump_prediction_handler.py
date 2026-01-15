#!/usr/bin/env python3
"""
Process Pump Prediction Handler
================================
RUL prediction using two-stage XGBoost models for process pump machine.

Architecture:
- BASE Model: XGBoost regressor for general RUL prediction
- Stage-1 Classifier: Binary classifier detecting critical region (RUL <= 100)
- Stage-2A Regressor: Specialized regressor for critical region (RUL <= 50)

Decision Logic (with hysteresis):
- SAFE -> CRITICAL: crit_prob >= 0.6
- CRITICAL -> SAFE: crit_prob <= 0.2

Optimized for Raspberry Pi deployment.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np


# CONSTANTS - Must Match Training Exactly
# Sensor names
SENSORS = ['vibration', 'temp_motor', 'pressure', 'vib_motor']

# Rolling window sizes (from training)
WINDOW_SHORT = 5
WINDOW_LONG = 20

# Selected features (order matters - must match model training)
SELECTED_FEATURES = [
                        "vibration_energy",
                        "vibration",
                        "temp_motor",
                        "pressure",
                        "vibration_roll_std_20",
                        "vib_motor",
                        "vib_motor_roll_std_20",
                        "temp_motor_roll_std_20",
                        "pressure_stability",
                        "motor_pump_coupling",
                        "pressure_roll_std_20"
                    ]

# Stage-1 Classifier thresholds (from training notebook)
THRESH_ENTER = 0.95   # Enter critical state
THRESH_EXIT = 0.20    # Exit critical state

# Buffer size for rolling calculations
MAX_BUFFER_SIZE = WINDOW_LONG + 10


# GLOBAL STATE
# Models (ONNX sessions or XGBoost objects)
_base_model = None
_stage1_classifier = None
_stage2a_regressor = None

# Feature normalization stats
_feature_stats: Optional[Dict] = None

# Data buffer for rolling window calculations
_data_buffer: List[Dict[str, float]] = []
_buffer_lock = threading.Lock()

# Hysteresis state for classifier
_in_critical_state = False

# Module initialization flag
_initialized = False


# MODEL LOADING
def _find_model_directory() -> Path:
    """Locate the models_and_features/process_pump directory."""
    base_dir = Path(__file__).parent.resolve()
    
    # Search up the directory tree
    for parent in [base_dir] + list(base_dir.parents):
        candidate = parent / "models_and_features" / "process_pump"
        if candidate.exists():
            return candidate
    
    # Fallback for typical ROS2 workspace structure
    # /workspace/install/pkg/lib/python3.x/site-packages/pkg/handler/
    # Models are at /workspace/models_and_features/process_pump/
    for parent in base_dir.parents:
        candidate = parent.parent.parent.parent.parent / "models_and_features" / "process_pump"
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
                                f"Cannot locate models_and_features/process_pump from {base_dir}"
                            )


def load_models() -> None:
    """
    Load all models, feature stats, selected_features and stage-1 thresholds.
    Primary: ONNX Runtime (optimized for RPi)
    Fallback: XGBoost JSON

    The function will also read the following files from the model directory:
      - process_pump_feature_stats.json  (min/max per raw sensor)
      - selected_process_pump_features.json (ordered list of features used by models)
      - process_pump_stage1_config.json (BEST_WEIGHT and thresholds)

    If found, these will update module globals: _feature_stats, SELECTED_FEATURES, THRESH_ENTER, THRESH_EXIT
    """
    global _base_model, _stage1_classifier, _stage2a_regressor
    global _feature_stats, _initialized, SELECTED_FEATURES, THRESH_ENTER, THRESH_EXIT

    if _initialized:
        return

    model_dir = _find_model_directory()
    print(f"[process_pump_handler] Loading models from: {model_dir}")

    # Load feature stats for normalization
    stats_path = model_dir / "process_pump_feature_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            _feature_stats = json.load(f)
        print(f"[process_pump_handler] Loaded feature stats")
    else:
        print(f"[process_pump_handler] WARNING: Feature stats not found, using defaults")
        _feature_stats = {
                            "vibration": {"min": 0.0, "max": 8.0},
                            "temp_motor": {"min": 40.0, "max": 1000.0},
                            "pressure": {"min": -100.0, "max": 10.0}
                        }

    # Load selected features list if available (ensures ordering matches training)
    sf_path = model_dir / "selected_process_pump_features.json"
    if sf_path.exists():
        try:
            with open(sf_path, 'r') as f:
                sf = json.load(f)
            if isinstance(sf, list) and len(sf) > 0:
                SELECTED_FEATURES = sf
                print(f"[process_pump_handler] Loaded selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
            else:
                print(f"[process_pump_handler] WARNING: selected features file malformed, using defaults")
        except Exception as e:
            print(f"[process_pump_handler] WARNING: Failed to read selected features: {e}")
    else:
        print(f"[process_pump_handler] WARNING: selected features file not found, using built-in defaults")

    # Load training config (thresholds, weights)
    cfg_path = model_dir / "process_pump_stage1_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            # Update thresholds if present
            THRESH_ENTER = float(cfg.get('THRESH_ENTER', THRESH_ENTER))
            THRESH_EXIT = float(cfg.get('THRESH_EXIT', THRESH_EXIT))
            print(f"[process_pump_handler] Loaded stage1 config: THRESH_ENTER={THRESH_ENTER}, THRESH_EXIT={THRESH_EXIT}")
        except Exception as e:
            print(f"[process_pump_handler] WARNING: Failed to read stage1 config: {e}")
    else:
        print(f"[process_pump_handler] WARNING: stage1 config not found, using defaults THRESH_ENTER={THRESH_ENTER}, THRESH_EXIT={THRESH_EXIT}")

    # Try ONNX first (optimized for RPi)
    try:
        import onnxruntime as ort

        # Session options optimized for RPi
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']

        _base_model = ort.InferenceSession(
                                            str(model_dir / "process_pump_base_model.onnx"),
                                            sess_options=opts,
                                            providers=providers
                                        )

        _stage1_classifier = ort.InferenceSession(
                                                    str(model_dir / "process_pump_stage1_classifier.onnx"),
                                                    sess_options=opts,
                                                    providers=providers
                                                )

        _stage2a_regressor = ort.InferenceSession(
                                                    str(model_dir / "process_pump_stage2a_regressor.onnx"),
                                                    sess_options=opts,
                                                    providers=providers
                                                )

        _initialized = True
        print("[process_pump_handler] ONNX models loaded successfully")
        return

    except ImportError:
        print("[process_pump_handler] ONNX Runtime not available, trying XGBoost")
    except Exception as e:
        print(f"[process_pump_handler] ONNX load failed: {e}, trying XGBoost")

    # Fallback to XGBoost JSON
    try:
        from xgboost import XGBRegressor, XGBClassifier

        _base_model = XGBRegressor()
        _base_model.load_model(str(model_dir / "process_pump_base_model.json"))

        _stage1_classifier = XGBClassifier()
        _stage1_classifier.load_model(str(model_dir / "process_pump_stage1_classifier.json"))

        _stage2a_regressor = XGBRegressor()
        _stage2a_regressor.load_model(str(model_dir / "process_pump_stage2a_regressor.json"))

        _initialized = True
        print("[process_pump_handler] XGBoost JSON models loaded successfully")
        return

    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")



# BUFFER MANAGEMENT
def _add_to_buffer(sensor_data: Dict[str, Any]) -> None:
    """Add new sensor reading to rolling buffer."""
    global _data_buffer
    
    with _buffer_lock:
        _data_buffer.append({
                                'cycle': int(sensor_data.get('cycle', 0)),
                                'vibration': float(sensor_data.get('vibration', 0.0)),
                                'temp_motor': float(sensor_data.get('temp_motor', 0.0)),
                                'pressure': float(sensor_data.get('pressure', 0.0)),
                                'vib_motor': float(sensor_data.get('vib_motor', 0.0)),
                            })
        
        # Trim to max size
        if len(_data_buffer) > MAX_BUFFER_SIZE:
            _data_buffer = _data_buffer[-MAX_BUFFER_SIZE:]


def reset_state() -> None:
    """Reset all state (call on new job/machine)."""
    global _data_buffer, _in_critical_state
    
    with _buffer_lock:
        _data_buffer = []
        _in_critical_state = False
    
    print("[process_pump_handler] State reset")


def _sync_buffer_from_history(history: List[Dict[str, Any]]) -> None:
    """
    Sync internal buffer from Predictor_Node's history.
    
    This handles the case where Predictor_Node maintains its own history
    and passes it to the handler. We take the last MAX_BUFFER_SIZE entries.
    
    Also handles non-sequential cycles by using the data as-is.
    Rolling calculations will work on available samples.
    """
    global _data_buffer
    
    with _buffer_lock:
        # Take last N entries from history
        recent = history[-MAX_BUFFER_SIZE:] if len(history) > MAX_BUFFER_SIZE else history
        
        _data_buffer = []
        for d in recent:
            _data_buffer.append({
                                    'cycle': int(d.get('cycle', 0)),
                                    'vibration': float(d.get('vibration', 0.0)),
                                    'temp_motor': float(d.get('temp_motor', 0.0)),
                                    'pressure': float(d.get('pressure', 0.0)),
                                    'vib_motor': float(d.get('vib_motor', 0.0)),
                                })



# FEATURE ENGINEERING
def _compute_features() -> Optional[np.ndarray]:
    """
    Compute features from the internal buffer to match training features.

    - Implementation is pandas-free and uses numpy over the in-memory buffer.
    - It supports a set of well-known features created during training and will
      build an output array in the exact order of `SELECTED_FEATURES`.

    Returns:
        np.ndarray shape (1, n_features) or None if insufficient data
    """
    with _buffer_lock:
        if len(_data_buffer) < 1:
            return None

        n = len(_data_buffer)
        # Convert buffer to numpy arrays for efficient ops
        vibration = np.array([d['vibration'] for d in _data_buffer], dtype=float)
        temp_motor = np.array([d['temp_motor'] for d in _data_buffer], dtype=float)
        pressure = np.array([d['pressure'] for d in _data_buffer], dtype=float)
        vib_motor = np.array([d['vib_motor'] for d in _data_buffer], dtype=float)

        latest_vib = float(vibration[-1])
        latest_temp = float(temp_motor[-1])
        latest_pres = float(pressure[-1])
        latest_vib_motor = float(vib_motor[-1])

        def mean_last(arr, w):
            w = min(int(w), len(arr))
            if w <= 0:
                return float(arr[-1])
            return float(np.mean(arr[-w:]))

        def std_last(arr, w):
            w = min(int(w), len(arr))
            if w <= 1:
                return 0.0
            # pandas uses ddof=1 by default for rolling.std(), match that
            return float(np.std(arr[-w:], ddof=1))

        def max_last(arr, w):
            w = min(int(w), len(arr))
            return float(np.max(arr[-w:]))

        # Compute a set of candidate features (covering typical selected features)
        feat = {}

        # Raw values
        feat['vibration'] = latest_vib
        feat['temp_motor'] = latest_temp
        feat['pressure'] = latest_pres
        feat['vib_motor'] = latest_vib_motor

        # Rolling means
        feat['vibration_roll_mean_5'] = mean_last(vibration, WINDOW_SHORT)
        feat['vibration_roll_mean_20'] = mean_last(vibration, WINDOW_LONG)

        # Rolling stds (long window = 20; short = 5)
        feat['vibration_roll_std_20'] = std_last(vibration, WINDOW_LONG)
        feat['vib_motor_roll_std_20'] = std_last(vib_motor, WINDOW_LONG)
        feat['temp_motor_roll_std_20'] = std_last(temp_motor, WINDOW_LONG)
        feat['pressure_roll_std_20'] = std_last(pressure, WINDOW_LONG)

        feat['vibration_roll_std_5'] = std_last(vibration, WINDOW_SHORT)
        feat['vib_motor_roll_std_5'] = std_last(vib_motor, WINDOW_SHORT)
        feat['temp_motor_roll_std_5'] = std_last(temp_motor, WINDOW_SHORT)
        feat['pressure_roll_std_5'] = std_last(pressure, WINDOW_SHORT)

        # Energy / volatility
        feat['vibration_energy'] = (feat['vibration_roll_mean_5']) ** 2
        feat['vibration_max_10'] = max_last(vibration, 10)

        # Deviation from long mean
        feat['vibration_dev_long'] = latest_vib - feat['vibration_roll_mean_20']

        # Pressure stability (matching training formula)
        pressure_std = max(feat.get('pressure_roll_std_20', 0.0), 1e-4)
        feat['pressure_stability'] = float(np.log1p(1.0 / pressure_std))

        # Motor-pump coupling with clipping (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            coupling = latest_vib_motor / latest_vib if latest_vib != 0 else np.nan
        if not np.isfinite(coupling):
            coupling = 1.0
        feat['motor_pump_coupling'] = float(np.clip(coupling, 0.0, 10.0))

        # Degradation index (approx using global min/max from training)
        if _feature_stats:
            vib_min = _feature_stats['vibration']['min']
            vib_max = _feature_stats['vibration']['max']
            temp_min = _feature_stats['temp_motor']['min']
            temp_max = _feature_stats['temp_motor']['max']
            pres_min = _feature_stats['pressure']['min']
            pres_max = _feature_stats['pressure']['max']
        else:
            vib_min, vib_max = 0.0, 8.0
            temp_min, temp_max = 40.0, 1000.0
            pres_min, pres_max = -100.0, 10.0

        eps = 1e-6
        vib_norm = (latest_vib - vib_min) / (vib_max - vib_min + eps)
        temp_norm = (latest_temp - temp_min) / (temp_max - temp_min + eps)
        pres_norm = 1.0 - (latest_pres - pres_min) / (pres_max - pres_min + eps)

        vib_norm = float(np.clip(vib_norm, 0.0, 1.0))
        temp_norm = float(np.clip(temp_norm, 0.0, 1.0))
        pres_norm = float(np.clip(pres_norm, 0.0, 1.0))

        feat['degradation_index'] = (vib_norm + temp_norm + pres_norm) / 3.0

        # Volatility surge (short std - long std, clipped)
        for s_name, arr in [('vibration', vibration), ('temp_motor', temp_motor), ('pressure', pressure), ('vib_motor', vib_motor)]:
            short_std = std_last(arr, WINDOW_SHORT)
            long_std = std_last(arr, WINDOW_LONG)
            feat[f'{s_name}_volatility_surge'] = float(max(0.0, short_std - long_std))

        # Provide a generic slope estimate (diff of long rolling mean)
        # Slope approximated as current long_mean - previous long_mean (if available)
        if len(vibration) >= 2:
            prev_long_mean = float(np.mean(vibration[-(WINDOW_LONG+1):-1])) if len(vibration) > WINDOW_LONG else float(np.mean(vibration[:-1]))
            feat['vibration_slope_20'] = feat['vibration_roll_mean_20'] - prev_long_mean
        else:
            feat['vibration_slope_20'] = 0.0

        # Now produce feature vector ordered by SELECTED_FEATURES
        if not isinstance(SELECTED_FEATURES, list) or len(SELECTED_FEATURES) == 0:
            # Fallback to a sensible default ordering (backwards compatible)
            sel = ['degradation_index', 'vibration_energy', 'vibration', 'temp_motor', 'pressure']
        else:
            sel = SELECTED_FEATURES

        missing = [f for f in sel if f not in feat]
        if missing:
            raise ValueError(f"Missing feature(s) in runtime feature computation: {missing}")

        arr = np.array([feat[f] for f in sel], dtype=np.float32).reshape(1, -1)
        return arr



# INFERENCE
def _run_inference_onnx(X: np.ndarray) -> tuple:
    """Run inference using ONNX Runtime."""
    input_name = _base_model.get_inputs()[0].name           # type: ignore
    
    # Stage-1: Classifier
    stg1_outputs = _stage1_classifier.run(None, {input_name: X})  # type: ignore
    label = int(np.asarray(stg1_outputs[0]).squeeze())
    
    # Extract probability (output[1] is probabilities for XGBoost classifier)
    try:
        proba_array = np.asarray(stg1_outputs[1]).squeeze()
        if proba_array.ndim == 0:
            # Scalar output
            crit_prob = float(proba_array)
        elif proba_array.size >= 2:
            # [prob_class0, prob_class1] - we want class 1 (critical)
            crit_prob = float(proba_array[1])
        else:
            crit_prob = float(label)
    except Exception:
        crit_prob = float(label)
    
    # BASE Model
    base_rul = float(_base_model.run(None, {input_name: X})[0].squeeze())   # type: ignore
    
    # Stage-2A Regressor
    stg2a_rul = float(_stage2a_regressor.run(None, {input_name: X})[0].squeeze())   # type: ignore
    
    return crit_prob, base_rul, stg2a_rul


def _run_inference_xgb(X: np.ndarray) -> tuple:
    """Run inference using XGBoost directly."""
    # Stage-1: Classifier
    crit_prob = float(_stage1_classifier.predict_proba(X)[0, 1])    # type: ignore
    
    # BASE Model
    base_rul = float(_base_model.predict(X)[0])  # type: ignore
    
    # Stage-2A Regressor
    stg2a_rul = float(_stage2a_regressor.predict(X)[0]) # type: ignore
    
    return crit_prob, base_rul, stg2a_rul



# MAIN PREDICTION FUNCTION
def predict(sensor_data) -> Dict[str, Any]:
    """
    Main prediction function called by Predictor_Node.
    
    Implements two-stage prediction with hysteresis:
    1. Stage-1 Classifier determines if machine is in critical region
    2. If critical: use Stage-2A specialized regressor
       If safe: use BASE general regressor
    3. Hysteresis prevents oscillation between states
    
    Args:
        sensor_data: Either:
            - Dict with keys: vibration, temp_motor, pressure, vib_motor, cycle
            - List[Dict] - history buffer from Predictor_Node (uses last entry)
    
    Returns:
        Dict with keys:
            - rul_min: float (predicted RUL in minutes)
            - unit: str ('min')
            - stage: str ('BASE' or 'STAGE_2A')
            - crit_prob: float (probability of critical state)
    """
    global _in_critical_state
    
    # Ensure models are loaded
    if not _initialized:
        load_models()
    
    # Handle both single dict and list of dicts
    if isinstance(sensor_data, list):
        if len(sensor_data) == 0:
            return {
                        'rul_min': -1.0,
                        'unit': 'min',
                        'stage': 'NO_DATA',
                        'crit_prob': 0.0
                    }
        # Use the entire history for rolling calculations
        _sync_buffer_from_history(sensor_data)
        current_data = sensor_data[-1]
    else:
        current_data = sensor_data
        # Add single data point to buffer
        _add_to_buffer(current_data)
    
    # Check for reset condition (new job starts at cycle 0)
    current_cycle = current_data.get('cycle', 0)
    if current_cycle == 0:
        reset_state()
        _add_to_buffer(current_data)
    
    # Compute features
    X = _compute_features()
    
    if X is None:
        return {
                    'rul_min': -1.0,
                    'unit': 'min',
                    'stage': 'INSUFFICIENT_DATA',
                    'crit_prob': 0.0
                }
    
    # Run inference
    try:
        # Check if using ONNX (has 'run' method) or XGBoost
        if hasattr(_base_model, 'run'):
            crit_prob, base_rul, stg2a_rul = _run_inference_onnx(X)
        else:
            crit_prob, base_rul, stg2a_rul = _run_inference_xgb(X)
    except Exception as e:
        print(f"[process_pump_handler] Inference error: {e}")
        return {
                    'rul_min': -1.0,
                    'unit': 'min',
                    'stage': 'ERROR',
                    'crit_prob': 0.0
                }
    
    # Decision Logic with Hysteresis
    # 
    # Stage-1 classifier determines if we're in critical region (RUL <= 100 min)
    # If crit_prob >= THRESH_ENTER -> switch to Stage-2A for fine-grained RUL
    # If crit_prob <= THRESH_EXIT -> switch back to BASE model
    #
    # NOTE: Stage-1 was trained on short-life dataset. If runtime uses long-life
    # pattern, classifier may behave differently. Consider retraining if needed.
    
    if not _in_critical_state:
        # Currently in SAFE state - use BASE model
        if crit_prob >= THRESH_ENTER:
            _in_critical_state = True
            stage = "STAGE_2A"
            final_rul = stg2a_rul
        else:
            stage = "BASE"
            final_rul = base_rul
    else:
        # Currently in CRITICAL state - use Stage-2A
        if crit_prob <= THRESH_EXIT:
            _in_critical_state = False
            stage = "BASE"
            final_rul = base_rul
        else:
            stage = "STAGE_2A"
            final_rul = stg2a_rul
    
    # Ensure RUL is non-negative
    final_rul = max(0.0, final_rul)
    
    return {
                'rul_min': float(final_rul),
                'unit': 'min',
                'stage': stage,
                'crit_prob': float(np.clip(crit_prob, 0.0, 1.0))
            }



# MODULE INITIALIZATION
# Attempt to pre-load models on import
try:
    load_models()
except Exception as e:
    print(f"[process_pump_handler] Deferred model loading: {e}")
