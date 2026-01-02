#!/usr/bin/env python3
"""
Hydraulic Press Prediction Handler

Multi-resolution RUL prediction pipeline for hydraulic press machine.

Architecture:
    Stage-0 Classifier: Regime detector (LONG_TERM vs NEAR_TERM)
        Threshold: RUL <= 5000 minutes -> NEAR_TERM
    
    Base Model: Long-term RUL tracker (HistGradientBoostingRegressor)
        Target: log1p(RUL_minutes / 60) -> hours in log scale
        Used for: General health trend monitoring
    
    Stage-1 Classifier: Critical detector (NON_CRITICAL vs CRITICAL)
        Threshold: RUL <= 600 minutes -> CRITICAL
        Hysteresis: ENTER >= 0.6, EXIT <= 0.2
    
    Stage-2A Regressor: Critical RUL prediction
        Target: log1p(RUL_minutes)
        Range: 0 - 600 minutes
        Used for: High-resolution prediction in critical regime

Inference Flow:
    Sensor data -> Stage-0 (regime) -> Stage-1 (critical check)
    If CRITICAL: Stage-2A prediction
    Else: Base Model prediction

Optimized for Raspberry Pi deployment.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np


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

SELECTED_FEATURES = [
                        "vibration_energy",
                        "vibration",
                        "degradation_index",
                        "flow_rate_roll_std_20",
                        "press_force",
                        "oil_contamination",
                        "oil_temperature",
                        "press_force_roll_std_20",
                        "vibration_roll_std_20",
                        "flow_rate",
                        "force_pressure_coupling",
                        "hydraulic_pressure",
                        "motor_current",
                        "ram_position_deviation_roll_std_20",
                        "oil_temperature_roll_std_20",
                        "motor_current_roll_std_20"
                    ]

BASE_MODEL_FEATURES = [
                        "vibration",
                        "degradation_index",
                        "press_force",
                        "oil_contamination",
                        "oil_temperature",
                        "flow_rate",
                        "force_pressure_coupling",
                        "hydraulic_pressure",
                        "motor_current"
                    ]

STAGE2A_FEATURES = [
                        "degradation_index",
                        "vibration",
                        "vibration_energy",
                        "motor_current",
                        "oil_temperature",
                        "force_pressure_coupling",
                        "press_force",
                        "vibration_roll_std_20",
                        "flow_rate_roll_std_20",
                        "press_force_roll_std_20"
                    ]

STAGE0_THRESHOLD_MIN = 5000
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
    global _feature_stats, _initialized
    
    if _initialized:
        return
    
    model_dir = _find_model_directory()
    print(f"[hydraulic_press_handler] Loading models from: {model_dir}")
    
    stats_path = model_dir / "hydraulic_press_feature_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            _feature_stats = json.load(f)
        print("[hydraulic_press_handler] Loaded feature stats")
    else:
        print("[hydraulic_press_handler] WARNING: Feature stats not found, using defaults")
        _feature_stats = {
                            "vibration": {"min": 0.0, "max": 10.0},
                            "oil_temperature": {"min": 30.0, "max": 100.0},
                            "hydraulic_pressure": {"min": 100.0, "max": 350.0},
                            "oil_contamination": {"min": 0.0, "max": 100.0}
                        }
                    
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
        from xgboost import XGBClassifier, XGBRegressor
        
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
    global _data_buffer, _in_critical_state
    
    with _buffer_lock:
        _data_buffer = []
        _in_critical_state = False
    
    print("[hydraulic_press_handler] State reset")


def _sync_buffer_from_history(history: List[Dict[str, Any]]) -> None:
    """Sync internal buffer from Predictor_Node history."""
    global _data_buffer
    
    with _buffer_lock:
        recent = history[-MAX_BUFFER_SIZE:] if len(history) > MAX_BUFFER_SIZE else history
        
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
    Compute all features from buffer matching training exactly.
    
    Returns dictionary with feature arrays for each model:
        - selected_features: Full feature set (16 features)
        - base_model_features: Subset for base model (9 features)
        - stage2a_features: Subset for stage-2a (10 features)
    """
    with _buffer_lock:
        if len(_data_buffer) < 1:
            return None
        
        n = len(_data_buffer)
        
        hydraulic_pressure = np.array([d["hydraulic_pressure"] for d in _data_buffer])
        oil_temperature = np.array([d["oil_temperature"] for d in _data_buffer])
        oil_contamination = np.array([d["oil_contamination"] for d in _data_buffer])
        ram_position_deviation = np.array([d["ram_position_deviation"] for d in _data_buffer])
        press_force = np.array([d["press_force"] for d in _data_buffer])
        vibration = np.array([d["vibration"] for d in _data_buffer])
        flow_rate = np.array([d["flow_rate"] for d in _data_buffer])
        motor_current = np.array([d["motor_current"] for d in _data_buffer])
        
        latest_hydraulic_pressure = hydraulic_pressure[-1]
        latest_oil_temperature = oil_temperature[-1]
        latest_oil_contamination = oil_contamination[-1]
        latest_press_force = press_force[-1]
        latest_vibration = vibration[-1]
        latest_flow_rate = flow_rate[-1]
        latest_motor_current = motor_current[-1]
        
        window_short = min(WINDOW_SHORT, n)
        vib_roll_mean = np.mean(vibration[-window_short:])
        vibration_energy = vib_roll_mean ** 2
        
        window_long = min(WINDOW_LONG, n)
        
        def rolling_std(arr, window):
            if len(arr) < 2:
                return 0.0
            window_data = arr[-window:]
            return float(np.std(window_data, ddof=1)) if len(window_data) > 1 else 0.0
        
        vibration_roll_std_20 = rolling_std(vibration, window_long)
        flow_rate_roll_std_20 = rolling_std(flow_rate, window_long)
        press_force_roll_std_20 = rolling_std(press_force, window_long)
        oil_temperature_roll_std_20 = rolling_std(oil_temperature, window_long)
        motor_current_roll_std_20 = rolling_std(motor_current, window_long)
        ram_position_deviation_roll_std_20 = rolling_std(ram_position_deviation, window_long)
        
        if latest_hydraulic_pressure > 0:
            force_pressure_coupling = latest_press_force / latest_hydraulic_pressure
            force_pressure_coupling = float(np.clip(force_pressure_coupling, 0, 10))
        else:
            force_pressure_coupling = 1.0
        
        eps = 1e-9
        
        vib_min = vibration.min()
        vib_max = vibration.max()
        vib_norm = (latest_vibration - vib_min) / (vib_max - vib_min + eps) if vib_max > vib_min else 0.0
        
        temp_min = oil_temperature.min()
        temp_max = oil_temperature.max()
        temp_norm = (latest_oil_temperature - temp_min) / (temp_max - temp_min + eps) if temp_max > temp_min else 0.0
        
        pres_min = hydraulic_pressure.min()
        pres_max = hydraulic_pressure.max()
        pres_norm = 1.0 - (latest_hydraulic_pressure - pres_min) / (pres_max - pres_min + eps) if pres_max > pres_min else 0.0
        
        contam_min = oil_contamination.min()
        contam_max = oil_contamination.max()
        contam_norm = (latest_oil_contamination - contam_min) / (contam_max - contam_min + eps) if contam_max > contam_min else 0.0
        
        vib_norm = float(np.clip(vib_norm, 0.0, 1.0))
        temp_norm = float(np.clip(temp_norm, 0.0, 1.0))
        pres_norm = float(np.clip(pres_norm, 0.0, 1.0))
        contam_norm = float(np.clip(contam_norm, 0.0, 1.0))
        
        degradation_index = (vib_norm + temp_norm + pres_norm + contam_norm) / 4.0
        
        feature_dict = {
                            "vibration_energy": vibration_energy,
                            "vibration": latest_vibration,
                            "degradation_index": degradation_index,
                            "flow_rate_roll_std_20": flow_rate_roll_std_20,
                            "press_force": latest_press_force,
                            "oil_contamination": latest_oil_contamination,
                            "oil_temperature": latest_oil_temperature,
                            "press_force_roll_std_20": press_force_roll_std_20,
                            "vibration_roll_std_20": vibration_roll_std_20,
                            "flow_rate": latest_flow_rate,
                            "force_pressure_coupling": force_pressure_coupling,
                            "hydraulic_pressure": latest_hydraulic_pressure,
                            "motor_current": latest_motor_current,
                            "ram_position_deviation_roll_std_20": ram_position_deviation_roll_std_20,
                            "oil_temperature_roll_std_20": oil_temperature_roll_std_20,
                            "motor_current_roll_std_20": motor_current_roll_std_20
                        }
        
        selected_feature_values = [feature_dict[f] for f in SELECTED_FEATURES]
        selected_features_array = np.array(selected_feature_values, dtype=np.float32).reshape(1, -1)
        
        base_feature_values = [feature_dict[f] for f in BASE_MODEL_FEATURES]
        base_features_array = np.array(base_feature_values, dtype=np.float32).reshape(1, -1)
        
        stage2a_feature_values = [feature_dict[f] for f in STAGE2A_FEATURES]
        stage2a_features_array = np.array(stage2a_feature_values, dtype=np.float32).reshape(1, -1)
        
        return {
                    "selected_features": selected_features_array,
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
    """Run Base Model using ONNX. Returns RUL in minutes."""
    input_name = _base_model.get_inputs()[0].name
    pred_log_hours = float(_base_model.run(None, {input_name: X})[0].squeeze())
    pred_hours = np.expm1(pred_log_hours)
    pred_minutes = pred_hours * 60.0
    return pred_minutes


def _run_base_model_sklearn(X: np.ndarray) -> float:
    """Run Base Model using sklearn. Returns RUL in minutes."""
    pred_log_hours = float(_base_model.predict(X)[0])
    pred_hours = np.expm1(pred_log_hours)
    pred_minutes = pred_hours * 60.0
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


def predict(sensor_data) -> Dict[str, Any]:
    """
    Main prediction function called by Predictor_Node.
    
    Implements multi-resolution RUL prediction:
        Stage-0: Regime detection (LONG_TERM vs NEAR_TERM)
        Stage-1: Critical detection with hysteresis
        Base Model: Long-term RUL tracking
        Stage-2A: Critical regime high-resolution prediction
    
    Args:
        sensor_data: Either:
            - Dict with sensor keys
            - List[Dict] - history buffer from Predictor_Node
    
    Returns:
        Dict with keys:
            - rul_min: float (predicted RUL in minutes)
            - unit: str ('min')
            - stage: str ('BASE', 'STAGE_2A', etc.)
            - crit_prob: float (probability of critical state)
            - regime: str ('LONG_TERM' or 'NEAR_TERM')
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
    if current_cycle == 0:
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
    X_base = feature_arrays["base_model_features"]
    X_stage2a = feature_arrays["stage2a_features"]
    
    try:
        is_onnx = hasattr(_stage0_classifier, "run")
        
        if is_onnx:
            near_term_prob = _run_stage0_onnx(X_selected)
        else:
            near_term_prob = _run_stage0_sklearn(X_selected)
        
        regime = "NEAR_TERM" if near_term_prob >= 0.5 else "LONG_TERM"
        
        if is_onnx:
            critical_prob = _run_stage1_onnx(X_selected)
        else:
            critical_prob = _run_stage1_sklearn(X_selected)
        
        if not _in_critical_state:
            if critical_prob >= THRESH_ENTER:
                _in_critical_state = True
        else:
            if critical_prob <= THRESH_EXIT:
                _in_critical_state = False
        
        if _in_critical_state:
            if is_onnx:
                final_rul = _run_stage2a_onnx(X_stage2a)
            else:
                final_rul = _run_stage2a_sklearn(X_stage2a)
            stage = "STAGE_2A"
        else:
            if is_onnx:
                final_rul = _run_base_model_onnx(X_base)
            else:
                final_rul = _run_base_model_sklearn(X_base)
            stage = "BASE"
        
        final_rul = max(0.0, final_rul)
        
        return {
                    "rul_min": float(final_rul),
                    "unit": "min",
                    "stage": stage,
                    "crit_prob": float(np.clip(critical_prob, 0.0, 1.0)),
                    "regime": regime
                }
        
    except Exception as e:
        print(f"[hydraulic_press_handler] Inference error: {e}")
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
