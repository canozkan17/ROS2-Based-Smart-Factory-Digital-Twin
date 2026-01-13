#!/usr/bin/env python3
"""
Hydraulic Press Prediction Handler

Multi-resolution RUL prediction pipeline for hydraulic press machine.

Architecture:
    Stage-0 Classifier: Regime detector (LONG_TERM vs NEAR_TERM)
        Threshold: RUL <= 5000 minutes -> NEAR_TERM

    Base Model: Long-term RUL tracker (XGBRegressor)
        Target: log1p(RUL_hours) where RUL_hours = RUL_minutes / 60

    Stage-1 Classifier: Critical detector (NON_CRITICAL vs CRITICAL)
        Threshold: RUL <= 600 minutes -> CRITICAL

    Stage-2A Regressor: Critical RUL prediction
        Target: log1p(RUL_minutes)
        Range: 0 - 600 minutes

Inference Flow:
    Long-Life Scenario: Stage-0 (regime) -> Base Model
    Short-Life Scenario: Stage-1 (critical check) -> Stage-2A
"""
import os
import json
import numpy as np
import joblib
from typing import List, Dict, Any
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

NORMALIZE_SENSORS = [
                        'hydraulic_pressure', 'oil_temperature', 'oil_contamination',
                        'vibration', 'press_force', 'flow_rate', 'motor_current'
                    ]


# internal state
global_baseline: Dict[str, float] = {}
sensors_raw: Dict[str, np.ndarray] = {}
selected_features: List[str] = []
base_config: Dict[str, Any] = {}
stage1_config: Dict[str, Any] = {}
stage2a_config: Dict[str, Any] = {}
stage0_threshold: Dict[str, Any] = {}
models: Dict[str, Any] = {"stage0": None, "stage1": None, "base": None, "stage2a": None}
_eps = 1e-9

# State management
def reset_state():
    """
    Reset temporary buffers.
    """
    global sensors_raw

    sensors_raw = {}


# Model & config loading
def load_models(models_dir: str = None): # type: ignore
    """
    Load models and configs from models_and_features/hydraulic_press.
    Expects:
      - selected_hydraulic_press_features.json
      - hydraulic_press_base_model_config.json
      - hydraulic_press_stage1_config.json (optional)
      - hydraulic_press_stage0_classifier.pkl etc (optional)
    """
    global global_baseline, selected_features, base_config, stage1_config, models

    if models_dir is None:
        models_dir = os.path.join(BASE_DIR, "models_and_features", "hydraulic_press")

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    # configs
    base_cfg_path = os.path.join(models_dir, "hydraulic_press_base_model_config.json")
    stage1_cfg_path = os.path.join(models_dir, "hydraulic_press_stage1_config.json")
    stage2a_cfg_path = os.path.join(models_dir, "hydraulic_press_stage2a_config.json")
    stage0_th_path = os.path.join(models_dir, "hydraulic_press_stage0_threshold.json")
    sel_feat_path = os.path.join(models_dir, "selected_hydraulic_press_features.json")

    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, "r") as f:
            base_config = json.load(f)
            global_baseline.update(base_config.get("global_baseline", {}))
            # keep in module var
            globals()['base_config'] = base_config

    if os.path.exists(stage1_cfg_path):
        with open(stage1_cfg_path, "r") as f:
            stage1_config = json.load(f)
            globals()['stage1_config'] = stage1_config

    # load stage2a config if present
    if os.path.exists(stage2a_cfg_path):
        with open(stage2a_cfg_path, "r") as f:
            stage2a_cfg = json.load(f)
            globals()['stage2a_config'] = stage2a_cfg

    # load stage0 threshold file (contains prob_threshold)
    if os.path.exists(stage0_th_path):
        with open(stage0_th_path, "r") as f:
            s0 = json.load(f)
            globals()['stage0_threshold'] = s0

    if os.path.exists(sel_feat_path):
        with open(sel_feat_path, "r") as f:
            selected_features = json.load(f)
            globals()['selected_features'] = selected_features

    # models
    def try_load(name):
        """
        Prefer ONNX if available, fallback to .pkl joblib model.
        Returns either:
          - ONNX wrapper object with predict(X) and predict_proba(X)
          - joblib-loaded model
          - None if neither loads
        """
        pkl_path = os.path.join(models_dir, name)
        onnx_path = os.path.splitext(pkl_path)[0] + ".onnx"

        # Try ONNX first
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                output_names = [o.name for o in sess.get_outputs()]

                class ONNXWrapper:
                    def __init__(self, sess, input_name, output_names):
                        self._sess = sess
                        self._input = input_name
                        self._outputs = output_names

                    def _run(self, X):
                        Xf = np.asarray(X, dtype=np.float32)
                        return self._sess.run(self._outputs, {self._input: Xf})

                    def predict(self, X):
                        outs = self._run(X)
                        # Prefer a 2D output interpreted as class probabilities
                        for o in outs:
                            if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[1] > 1:
                                return np.argmax(o, axis=1)
                        # fallback: use first output flattened
                        return np.asarray(outs[0]).ravel()

                    def predict_proba(self, X):
                        outs = self._run(X)
                        # Return first suitable 2D output as probabilities
                        for o in outs:
                            if isinstance(o, np.ndarray) and o.ndim == 2:
                                return o
                        # fallback: construct one-hot from predict()
                        labels = np.asarray(self.predict(X), dtype=int)
                        if labels.size == 0:
                            return np.zeros((0, 2), dtype=np.float32)
                        ncls = int(labels.max()) + 1
                        probs = np.zeros((labels.shape[0], ncls), dtype=np.float32)
                        probs[np.arange(labels.shape[0]), labels] = 1.0
                        return probs

                return ONNXWrapper(sess, input_name, output_names)
            except Exception:
                # If ONNX import or session fails, continue to try pkl
                pass

        # Fallback to joblib .pkl
        if os.path.exists(pkl_path):
            try:
                return joblib.load(pkl_path)
            except Exception:
                return None

        return None

    models["stage0"] = try_load("hydraulic_press_stage0_classifier.pkl")
    models["stage1"] = try_load("hydraulic_press_stage1_classifier.pkl")
    models["base"] = try_load("hydraulic_press_base_model.pkl")
    models["stage2a"] = try_load("hydraulic_press_stage2a_regressor.pkl")


# Helpers for arrays
def _arr_from_sensor_data(sensor_data: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for s in sensor_data:
        v = s.get(key, np.nan)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals, dtype=np.float32)


# Normalization & feature engineering
def normalize_sensors_and_features(sensor_data: List[Dict[str, Any]]):
    """
    Populate sensors_raw with normalized sensor numpy arrays (per- passed sensor data window).
    """
    global sensors_raw, global_baseline
    sensors_raw = {}
    eps = 1e-6
    for sensor in NORMALIZE_SENSORS:
        if sensor not in global_baseline:
            raise KeyError(f"Missing baseline for sensor: {sensor}")
        baseline_val = float(global_baseline[sensor])
        arr = _arr_from_sensor_data(sensor_data, sensor)
        norm = (arr - baseline_val) / (abs(baseline_val) + eps)
        sensors_raw[f"{sensor}_norm"] = norm.astype(np.float32)


def apply_feature_engineering(sensor_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Return engineered features for last sample in sensor_data.
    Provides:
      - normalized_degradation
      - time_position (0..1)
      - per-sensor last, mean, std, max, slope
      - normalized last values for normalized sensors
    """
    engineered = {}
    n = len(sensor_data)
    if n == 0:
        return engineered

    # normalized sensors available in sensors_raw (must have been called)
    # normalized_degradation: simple composite using available normalized sensors
    pos_keys = ["vibration_norm", "oil_temperature_norm", "oil_contamination_norm", "motor_current_norm"]
    neg_keys = ["hydraulic_pressure_norm", "press_force_norm", "flow_rate_norm"]
    pos_sum = 0.0; pos_count = 0
    neg_sum = 0.0; neg_count = 0
    for k in pos_keys:
        if k in sensors_raw:
            val = float(sensors_raw[k][-1])
            pos_sum += val; pos_count += 1
    for k in neg_keys:
        if k in sensors_raw:
            val = -float(sensors_raw[k][-1])
            neg_sum += val; neg_count += 1
    denom = max(1, pos_count + neg_count)
    engineered["normalized_degradation"] = (pos_sum + neg_sum) / denom

    # time_position: use elapsed_minutes/time if available otherwise index position
    time_arr = _arr_from_sensor_data(sensor_data, "elapsed_minutes")
    if not np.all(np.isnan(time_arr)):
        tmin = float(np.nanmin(time_arr))
        tmax = float(np.nanmax(time_arr))
        last = float(time_arr[-1])
        engineered["time_position"] = (last - tmin) / ( (tmax - tmin) + _eps )
    else:
        engineered["time_position"] = float((n - 1) / max(1, n - 1))

    # per-sensor stats (discover sensors from last sample)
    last_sample = sensor_data[-1]
    exclude_keys = {"cycle", "elapsed_hours", "elapsed_minutes", "cycle_id", "time_min", "total_rul", "current_rul", "timestamp"}
    sensor_keys = [k for k in last_sample.keys() if k not in exclude_keys]
    for s in sensor_keys:
        arr = _arr_from_sensor_data(sensor_data, s)
        engineered[f"{s}_last"] = float(arr[-1]) if arr.size > 0 and not np.isnan(arr[-1]) else 0.0
        engineered[f"{s}_mean_win"] = float(np.nanmean(arr)) if arr.size > 0 else 0.0
        engineered[f"{s}_std_win"] = float(np.nanstd(arr)) if arr.size > 0 else 0.0
        engineered[f"{s}_max_win"] = float(np.nanmax(arr)) if arr.size > 0 else 0.0
        if arr.size >= 2 and not np.isnan(arr[0]) and not np.isnan(arr[-1]):
            engineered[f"{s}_slope_win"] = float((arr[-1] - arr[0]) / (arr.size - 1))
        else:
            engineered[f"{s}_slope_win"] = 0.0

    # normalized last vals
    for k, v in sensors_raw.items():
        engineered[k] = float(v[-1]) if v.size > 0 and not np.isnan(v[-1]) else 0.0

    return engineered


# Feature matrix preparation
def prepare_feature_matrices(sensor_data: List[Dict[str, Any]], engineered: Dict[str, float]):
    """
    Build X_clf (for classifiers/regressors using selected_features) and X_base (for base model).
    """
    global selected_features, base_config, sensors_raw
    if not selected_features:
        # fallback: try to load from base_config 'classifier_features'
        selected = base_config.get("classifier_features", []) if base_config else []
    else:
        selected = selected_features

    # Build classifier vector
    clf_vec = []
    last_sample = sensor_data[-1]
    for feat in selected:
        if feat in engineered:
            clf_vec.append(float(engineered[feat]))
            continue
        if feat in last_sample:
            try:
                clf_vec.append(float(last_sample.get(feat, 0.0)))
                continue
            except Exception:
                pass
        # roll/derived patterns
        if "roll_mean" in feat or "roll_std" in feat or "_slope_" in feat or "_max_" in feat or "pct_rank" in feat or "exp_degradation" in feat:
            parts = feat.split("_")
            sensor_candidate = parts[0]
            # simple parsing for roll size
            num = None
            for p in reversed(parts):
                if p.isdigit():
                    num = int(p); break
            arr = _arr_from_sensor_data(sensor_data, sensor_candidate)
            if "roll_mean" in feat and num:
                val = float(np.nanmean(arr[-num:])) if arr.size > 0 else 0.0
                clf_vec.append(val); continue
            if "roll_std" in feat and num:
                val = float(np.nanstd(arr[-num:])) if arr.size > 0 else 0.0
                clf_vec.append(val); continue
            if "slope" in feat:
                if arr.size >= 2:
                    val = float((np.nanmean(arr[-min(20, arr.size):]) - np.nanmean(arr[:min(20, arr.size)])) / max(1, min(20, arr.size)))
                else:
                    val = 0.0
                clf_vec.append(val); continue
            if "max" in feat and num:
                val = float(np.nanmax(arr[-num:])) if arr.size > 0 else 0.0
                clf_vec.append(val); continue
            if "pct_rank" in feat:
                lastv = arr[-1] if arr.size > 0 else np.nan
                mx = np.nanmax(arr) if arr.size > 0 else np.nan
                val = float((lastv / (mx + _eps)) if not np.isnan(lastv) else 0.0)
                clf_vec.append(val); continue
            if "exp_degradation" in feat:
                if arr.size > 0:
                    arr_min = float(np.nanmin(arr)); arr_max = float(np.nanmax(arr)); lastv = float(arr[-1])
                    denom = max(_eps, arr_max - arr_min)
                    val = (lastv - arr_min) / denom if denom > 0 else 0.0
                else:
                    val = 0.0
                clf_vec.append(val); continue

        # normalized fallback
        if feat in sensors_raw:
            arr = sensors_raw[feat]
            clf_vec.append(float(arr[-1]) if arr.size > 0 else 0.0)
            continue

        # final fallback
        clf_vec.append(0.0)

    X_clf = np.asarray(clf_vec, dtype=np.float32).reshape(1, -1)

    # Build base features
    base_feat_names = base_config.get("features") if base_config else None
    if not base_feat_names:
        base_feat_names = [f"{s}_norm" for s in NORMALIZE_SENSORS] + ["normalized_degradation", "time_position"]
    base_vec = []
    for b in base_feat_names:
        if b in engineered:
            base_vec.append(float(engineered[b])); continue
        if b in sensors_raw:
            arr = sensors_raw[b]; base_vec.append(float(arr[-1]) if arr.size > 0 else 0.0); continue
        if b in last_sample:
            try:
                base_vec.append(float(last_sample.get(b, 0.0))); continue
            except Exception:
                pass
        base_vec.append(0.0)
    X_base = np.asarray(base_vec, dtype=np.float32).reshape(1, -1)

    return X_clf, X_base


# Prediction pipeline
def predict(sensor_data: List[Dict[str, Any]]):
    """
    Run the multi-stage inference pipeline.
    sensor_data: list of samples. predictor_node passes.
        
        - stage0: LONG_TERM vs NEAR_TERM (Classifier)
            if LONG_TERM: BASE MODEL (regression)
            if NEAR_TERM:
                - stage1: NON_CRITICAL vs CRITICAL (Classifier)
                    if NON_CRITICAL: BASE MODEL (regression)
                    if CRITICAL: STAGE2A regression model
    
    Returns dict with: rul_min, regime, active_model, stage0_prob, stage1_prob
    """
    global models, base_config, stage1_config, selected_features

    if not isinstance(sensor_data, list) or len(sensor_data) == 0:
        raise ValueError("predict requires non-empty list of dicts")

    # ensure models/configs loaded
    if models["base"] is None and (not base_config):
        load_models()

    # normalization and engineering
    normalize_sensors_and_features(sensor_data)
    engineered = apply_feature_engineering(sensor_data)
    X_clf, X_base = prepare_feature_matrices(sensor_data, engineered)

    # Stage-0
    stage0_prob = None
    stage0_pred = None

    if models.get("stage0") is not None:
        try:
            probs = models["stage0"].predict_proba(X_clf)
            stage0_prob = float(probs[:, 1].ravel()[0])
            # threshold can be in base_config or use default 0.7
            thresh = float(base_config.get("stage0_threshold", 0.7)) if base_config else 0.7
            stage0_pred = int(stage0_prob >= thresh)
        except Exception:
            try:
                p = int(models["stage0"].predict(X_clf).ravel()[0])
                stage0_pred = p; stage0_prob = float(p)
            except Exception:
                stage0_pred = 1; stage0_prob = 1.0
    else:
        # default assume NEAR_TERM to exercise short-term pipeline
        stage0_pred = 1; stage0_prob = 1.0

    # Stage-0 threshold 
    thresh = 0.7

    # LONG_TERM -> Base model
    if stage0_pred == 0:
        if models.get("base") is None:
            raise RuntimeError("Base model not loaded")
        base_log = float(models["base"].predict(X_base).ravel()[0])
        base_hr = float(np.expm1(base_log))
        base_min = base_hr * 60.0
        base_min = _clamp_rul_minutes(base_min)
        return {
                    "rul_min": float(base_min),
                    "regime": "LONG_TERM",
                    "active_model": "Base Model",
                    "stage0_prob": stage0_prob,
                    "stage1_prob": None
                }

    # NEAR_TERM -> Stage-1
    stage1_prob = None; stage1_pred = None
    if models.get("stage1") is not None:
        try:
            probs1 = models["stage1"].predict_proba(X_clf)
            stage1_prob = float(probs1[:, 1].ravel()[0])
            thresh1 = float(stage1_config.get("optimal_prob_threshold", 0.5)) if stage1_config else 0.5
            stage1_pred = int(stage1_prob >= thresh1)
        except Exception:
            try:
                p = int(models["stage1"].predict(X_clf).ravel()[0])
                stage1_pred = p; stage1_prob = float(p)
            except Exception:
                stage1_pred = 0; stage1_prob = 0.0
    else:
        stage1_pred = 0; stage1_prob = 0.0

    # NON_CRITICAL -> Base Model
    if stage1_pred == 0:
        if models.get("base") is None:
            raise RuntimeError("Base model not loaded")
        base_log = float(models["base"].predict(X_base).ravel()[0])
        base_hr = float(np.expm1(base_log))
        base_min = base_hr * 60.0
        base_min = _clamp_rul_minutes(base_min)
        return {
                    "rul_min": float(base_min),
                    "regime": "NEAR_TERM",
                    "active_model": "Base Model",
                    "stage0_prob": stage0_prob,
                    "stage1_prob": stage1_prob
                }

    # CRITICAL -> Stage-2A
    if models.get("stage2a") is None:
        # fallback to base
        if models.get("base") is None:
            raise RuntimeError("No suitable model loaded for CRITICAL prediction")
        base_log = float(models["base"].predict(X_base).ravel()[0])
        base_hr = float(np.expm1(base_log))
        base_min = base_hr * 60.0
        base_min = _clamp_rul_minutes(base_min)
        return {
                "rul_min": float(base_min),
                "regime": "CRITICAL",
                "active_model": "Base Model (fallback)",
                "stage0_prob": stage0_prob,
                "stage1_prob": stage1_prob
            }

    try:
        pred_log = float(models["stage2a"].predict(X_clf).ravel()[0])
        pred_min = float(np.expm1(pred_log))
        # clamp according to config (min from base_config and max from stage2a_config)
        pred_min = _clamp_rul_minutes(pred_min)
        return {
                    "rul_min": pred_min,
                    "regime": "CRITICAL",
                    "active_model": "Stage-2A",
                    "stage0_prob": stage0_prob,
                    "stage1_prob": stage1_prob
                }
    except Exception as e:
        return {
                    "rul_min": 0.0,
                    "regime": "CRITICAL",
                    "active_model": "Stage-2A (failed)",
                    "stage0_prob": stage0_prob,
                    "stage1_prob": stage1_prob,
                    "notes": str(e)
                }

def _clamp_rul_minutes(val_min: float) -> float:
    global base_config, stage2a_config
    v = float(val_min)

    # minimum: base_config['min_rul_hours'] -> minutes
    if base_config and "min_rul_hours" in base_config:
        try:
            min_hours = float(base_config["min_rul_hours"])
            v = max(v, min_hours * 60.0)
        except Exception:
            pass

    # maximum: prefer stage2a_config['max_rul_min']
    max_min = None
    if stage2a_config and "max_rul_min" in stage2a_config:
        try:
            max_min = float(stage2a_config["max_rul_min"])
        except Exception:
            max_min = None


    if max_min is not None:
        v = min(v, max_min)

    return float(max(0.0, v))