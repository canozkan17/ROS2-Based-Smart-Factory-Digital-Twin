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

# Running cycle timing state (set on first incoming window, cleared in reset_state)
_cycle_start_minute = None
_cycle_max_seen_minute = None

# State management
def reset_state():
    """
    Reset temporary buffers.
    """
    global sensors_raw, _cycle_start_minute, _cycle_max_seen_minute

    sensors_raw = {}
    # Clear recorded cycle timing state so subsequent runs start fresh
    _cycle_start_minute = None
    _cycle_max_seen_minute = None


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

    # TEMP:DEBUG - model directory
    try:
        print(f"[TEMP:DEBUG] hydraulic_press handler loading models from: {models_dir}")
    except Exception:
        pass

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
            try:
                print(f"[TEMP:DEBUG] hydraulic_press base_config loaded: keys={list(base_config.keys())}")
                # explicit helpful debug values
                min_rul_h = base_config.get('min_rul_hours', None)
                print(f"[TEMP:DEBUG] hydraulic_press base_config: min_rul_hours={min_rul_h}")
            except Exception:
                pass

    if os.path.exists(stage1_cfg_path):
        with open(stage1_cfg_path, "r") as f:
            stage1_config = json.load(f)
            globals()['stage1_config'] = stage1_config
            try:
                print(f"[TEMP:DEBUG] hydraulic_press stage1_config loaded: keys={list(stage1_config.keys())}")
            except Exception:
                pass

    # load stage2a config if present
    if os.path.exists(stage2a_cfg_path):
        with open(stage2a_cfg_path, "r") as f:
            stage2a_cfg = json.load(f)
            globals()['stage2a_config'] = stage2a_cfg
            try:
                print(f"[TEMP:DEBUG] hydraulic_press stage2a_config loaded: keys={list(stage2a_cfg.keys())}")
            except Exception:
                pass

    # load stage0 threshold file (contains prob_threshold)
    if os.path.exists(stage0_th_path):
        with open(stage0_th_path, "r") as f:
            s0 = json.load(f)
            # Enforce training notebook policy: use prob_threshold=0.7 for Stage-0
            try:
                s0['prob_threshold'] = 0.7
            except Exception:
                pass
            globals()['stage0_threshold'] = s0
            try:
                print(f"[TEMP:DEBUG] hydraulic_press stage0_threshold loaded/overridden: keys={list(s0.keys())}, prob_threshold={s0.get('prob_threshold')}")
            except Exception:
                pass

    if os.path.exists(sel_feat_path):
        with open(sel_feat_path, "r") as f:
            selected_features = json.load(f)
            globals()['selected_features'] = selected_features
            try:
                print(f"[TEMP:DEBUG] hydraulic_press selected_features loaded: count={len(selected_features)}")
            except Exception:
                pass

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

                try:
                    print(f"[TEMP:DEBUG] hydraulic_press try_load: ONNX model found and loaded: {onnx_path}")
                except Exception:
                    pass
                return ONNXWrapper(sess, input_name, output_names)
            except Exception as e:
                try:
                    print(f"[TEMP:DEBUG] hydraulic_press try_load: ONNX load failed for {onnx_path}: {e}")
                except Exception:
                    pass
                # If ONNX import or session fails, continue to try pkl
                pass

        # Fallback to joblib .pkl
        if os.path.exists(pkl_path):
            try:
                m = joblib.load(pkl_path)
                try:
                    print(f"[TEMP:DEBUG] hydraulic_press try_load: PKL model loaded: {pkl_path}")
                except Exception:
                    pass
                return m
            except Exception as e:
                try:
                    print(f"[TEMP:DEBUG] hydraulic_press try_load: PKL load failed for {pkl_path}: {e}")
                except Exception:
                    pass
                return None

        try:
            print(f"[TEMP:DEBUG] hydraulic_press try_load: No model files found for {name} (looked for {pkl_path} and {onnx_path})")
        except Exception:
            pass

        return None

        return None

    models["stage0"] = try_load("hydraulic_press_stage0_classifier.pkl")
    models["stage1"] = try_load("hydraulic_press_stage1_classifier.pkl")
    models["base"] = try_load("hydraulic_press_base_model.pkl")
    models["stage2a"] = try_load("hydraulic_press_stage2a_regressor.pkl")

    # TEMP:DEBUG - summary of loaded models
    try:
        for k, v in models.items():
            if v is None:
                print(f"[TEMP:DEBUG] hydraulic_press model summary: {k} = None")
            else:
                tname = type(v).__name__
                print(f"[TEMP:DEBUG] hydraulic_press model summary: {k} = {tname}")
    except Exception:
        pass


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
      - normalized_degradation, degradation_index
      - time_position, time_progress (0..1)
      - vibration_energy, vibration_cum_mean
      - pressure_stability, force_pressure_coupling, current_pressure_ratio
      - exp_degradation features
      - per-sensor last, mean, std, max, slope, trend_slope
      - normalized last values for normalized sensors

    Optimized for RPI: uses numpy vectorization, minimal allocations.
    """
    engineered = {}
    n = len(sensor_data)
    if n == 0:
        return engineered

    last_sample = sensor_data[-1]
    
    # GLOBAL SENSOR RANGES for degradation_index (must match training EXACTLY)
    GLOBAL_RANGES = {
        'vibration': (0.11, 12.60),
        'oil_temperature': (46.23, 815.08),
        'hydraulic_pressure': (76.03, 201.0),
        'oil_contamination': (1.63, 27.85),
    }
    
    # Cache sensor arrays (RPI optimization: compute once)
    _sensor_cache = {}
    def get_arr(key):
        if key not in _sensor_cache:
            _sensor_cache[key] = _arr_from_sensor_data(sensor_data, key)
        return _sensor_cache[key]
    
    # degradation_index (matches training: global fixed ranges)
    def norm_global(key):
        if key not in GLOBAL_RANGES:
            return 0.0
        arr = get_arr(key)
        if arr.size == 0:
            return 0.0
        g_min, g_max = GLOBAL_RANGES[key]
        val = float(arr[-1])
        return max(0.0, min(1.0, (val - g_min) / (g_max - g_min + _eps)))
    
    vib_n = norm_global('vibration')
    temp_n = norm_global('oil_temperature')
    pres_n = norm_global('hydraulic_pressure')
    cont_n = norm_global('oil_contamination')
    engineered["degradation_index"] = (vib_n + temp_n + (1 - pres_n) + cont_n) / 4.0
    
    # normalized_degradation (existing baseline-based)
    pos_keys = ["vibration_norm", "oil_temperature_norm", "oil_contamination_norm", "motor_current_norm"]
    neg_keys = ["hydraulic_pressure_norm", "press_force_norm", "flow_rate_norm"]
    pos_sum = 0.0; pos_count = 0
    neg_sum = 0.0; neg_count = 0
    for k in pos_keys:
        if k in sensors_raw:
            pos_sum += float(sensors_raw[k][-1]); pos_count += 1
    for k in neg_keys:
        if k in sensors_raw:
            neg_sum += -float(sensors_raw[k][-1]); neg_count += 1
    engineered["normalized_degradation"] = (pos_sum + neg_sum) / max(1, pos_count + neg_count)

    # time_position & time_progress
    time_arr = get_arr("elapsed_minutes")
    global _cycle_start_minute, _cycle_max_seen_minute, base_config
    if time_arr.size > 0 and not np.all(np.isnan(time_arr)):
        tlast = float(time_arr[-1])
        tmin_window = float(np.nanmin(time_arr))
        tmax_window = float(np.nanmax(time_arr))
        # initialize cycle start minute from first observed sample (first window's first minute)
        if _cycle_start_minute is None:
            _cycle_start_minute = float(time_arr[0])
        # update running maximum observed elapsed minute (proxy for observed progress)
        if _cycle_max_seen_minute is None or tmax_window > _cycle_max_seen_minute:
            _cycle_max_seen_minute = float(tmax_window)

        baseline_samples = int(base_config.get('baseline_samples', 100)) if base_config else 100
        # Estimate cycle length in minutes using training min_rul_hours as a lower bound
        min_rul_hours = float(base_config.get('min_rul_hours', 0.0)) if base_config else 0.0
        est_cycle_length_min = max(_cycle_max_seen_minute - _cycle_start_minute, min_rul_hours * 60.0, float(baseline_samples))

        # time_position: fraction of elapsed minutes vs estimated cycle length
        engineered["time_position"] = (tlast - _cycle_start_minute) / max(_eps, est_cycle_length_min)
        engineered["time_position"] = float(min(1.0, max(0.0, engineered["time_position"])))

        # time_progress: fraction of observed samples vs baseline_samples (conservative warm-up)
        engineered["time_progress"] = float(min(1.0, float(n) / max(1, baseline_samples)))
    else:
        # fallback to conservative progress estimates when no time array available
        engineered["time_position"] = 0.0
        engineered["time_progress"] = float(min(1.0, float(n) / max(1, int(base_config.get('baseline_samples', 100)) if base_config else 100)))    
    # vibration_energy: rolling_mean(5) ** 2
    vib_arr = get_arr('vibration')
    if vib_arr.size >= 5:
        vib_roll_mean = float(np.nanmean(vib_arr[-5:]))
        engineered["vibration_energy"] = vib_roll_mean ** 2
    elif vib_arr.size > 0:
        engineered["vibration_energy"] = float(vib_arr[-1]) ** 2
    else:
        engineered["vibration_energy"] = 0.0
    
    # vibration_cum_mean: cumulative mean
    if vib_arr.size > 0:
        engineered["vibration_cum_mean"] = float(np.nanmean(vib_arr))
    else:
        engineered["vibration_cum_mean"] = 0.0
    
    # pressure_stability: log1p(1 / std_20)
    hp_arr = get_arr('hydraulic_pressure')
    if hp_arr.size >= 20:
        hp_std = float(np.nanstd(hp_arr[-20:]))
        engineered["pressure_stability"] = float(np.log1p(1.0 / max(1e-4, hp_std)))
    elif hp_arr.size > 0:
        hp_std = float(np.nanstd(hp_arr))
        engineered["pressure_stability"] = float(np.log1p(1.0 / max(1e-4, hp_std + 0.1)))
    else:
        engineered["pressure_stability"] = 0.0
    
    # force_pressure_coupling: press_force / hydraulic_pressure
    pf_arr = get_arr('press_force')
    if pf_arr.size > 0 and hp_arr.size > 0:
        pf_last = float(pf_arr[-1])
        hp_last = float(hp_arr[-1])
        engineered["force_pressure_coupling"] = min(10.0, max(0.0, pf_last / (hp_last + _eps)))
    else:
        engineered["force_pressure_coupling"] = 1.0
    
    # current_pressure_ratio: motor_current / hydraulic_pressure
    mc_arr = get_arr('motor_current')
    if mc_arr.size > 0 and hp_arr.size > 0:
        mc_last = float(mc_arr[-1])
        hp_last = float(hp_arr[-1])
        engineered["current_pressure_ratio"] = mc_last / max(1.0, hp_last)
    else:
        engineered["current_pressure_ratio"] = 0.0
    
    # exp_degradation features (expanding window)
    degradation_sensors = ['vibration', 'oil_temperature', 'oil_contamination', 'motor_current']
    exp_vals = []
    for s in degradation_sensors:
        arr = get_arr(s)
        if arr.size > 0:
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            arr_last = float(arr[-1])
            rng = max(1e-6, arr_max - arr_min)
            exp_val = max(0.0, min(1.0, (arr_last - arr_min) / rng))
            engineered[f"{s}_exp_degradation"] = exp_val
            exp_vals.append(exp_val)
        else:
            engineered[f"{s}_exp_degradation"] = 0.0
    
    # exp_degradation_combined
    engineered["exp_degradation_combined"] = sum(exp_vals) / max(1, len(exp_vals)) if exp_vals else 0.0
    
    # trend_slope features (long window slope)
    trend_sensors = ['vibration', 'oil_temperature', 'motor_current']
    for s in trend_sensors:
        arr = get_arr(s)
        if arr.size >= 20:
            # slope over long window (diff of rolling means)
            early_mean = float(np.nanmean(arr[:20]))
            late_mean = float(np.nanmean(arr[-20:]))
            engineered[f"{s}_trend_slope"] = (late_mean - early_mean) / max(1, arr.size - 20)
        elif arr.size >= 2:
            engineered[f"{s}_trend_slope"] = float((arr[-1] - arr[0]) / (arr.size - 1))
        else:
            engineered[f"{s}_trend_slope"] = 0.0
    
    # pct_rank features
    for s in ['vibration', 'oil_temperature']:
        arr = get_arr(s)
        if arr.size > 0:
            lastv = float(arr[-1])
            mx = float(np.nanmax(arr))
            engineered[f"{s}_pct_rank"] = lastv / (mx + _eps) if mx > _eps else 0.0
        else:
            engineered[f"{s}_pct_rank"] = 0.0
    
    # per-sensor stats (discover sensors from last sample)
    exclude_keys = {"cycle", "elapsed_hours", "elapsed_minutes", "cycle_id", "time_min", "total_rul", "current_rul", "timestamp"}
    sensor_keys = [k for k in last_sample.keys() if k not in exclude_keys]
    for s in sensor_keys:
        arr = get_arr(s)
        if arr.size > 0:
            engineered[f"{s}_last"] = float(arr[-1]) if not np.isnan(arr[-1]) else 0.0
            engineered[f"{s}_mean_win"] = float(np.nanmean(arr))
            engineered[f"{s}_std_win"] = float(np.nanstd(arr))
            engineered[f"{s}_max_win"] = float(np.nanmax(arr))
            if arr.size >= 2 and not np.isnan(arr[0]) and not np.isnan(arr[-1]):
                engineered[f"{s}_slope_win"] = float((arr[-1] - arr[0]) / (arr.size - 1))
            else:
                engineered[f"{s}_slope_win"] = 0.0
        else:
            engineered[f"{s}_last"] = 0.0
            engineered[f"{s}_mean_win"] = 0.0
            engineered[f"{s}_std_win"] = 0.0
            engineered[f"{s}_max_win"] = 0.0
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
            # TEMP:DEBUG - print raw probs to verify class ordering
            try:
                print(f"[TEMP:DEBUG] stage0 raw_probs={probs}")
            except Exception:
                pass
            # By convention we previously used column 1 as "prob of critical/near-term" —
            # but ONNX models may order classes differently. Extract both safely.
            try:
                # If probs is 2D with 2 columns, take column 1 as originally used
                if isinstance(probs, (list, tuple)):
                    probs_arr = np.asarray(probs)
                else:
                    probs_arr = probs
                if hasattr(probs_arr, 'ndim') and probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                    stage0_prob = float(probs_arr[:, 1].ravel()[0])
                else:
                    # fallback: use first value
                    stage0_prob = float(np.asarray(probs).ravel()[0])
            except Exception:
                stage0_prob = float(np.asarray(probs).ravel()[0])

            # Use stage0_threshold 'prob_threshold' if available, else default 0.5
            try:
                thresh = float(stage0_threshold.get('prob_threshold', 0.5)) if stage0_threshold else 0.5
            except Exception:
                thresh = 0.5

            stage0_pred = int(stage0_prob >= thresh)
            try:
                print(f"[TEMP:DEBUG] stage0: prob_col_used={stage0_prob}, thresh={thresh}, stage0_pred={stage0_pred}")
            except Exception:
                pass
        except Exception:
            try:
                p = int(models["stage0"].predict(X_clf).ravel()[0])
                stage0_pred = p; stage0_prob = float(p)
            except Exception:
                stage0_pred = 1; stage0_prob = 1.0
    else:
        # default assume NEAR_TERM to exercise short-term pipeline
        stage0_pred = 1; stage0_prob = 1.0

    # TEMP:DEBUG - stage0 decision and input shape
    try:
        thresh_dbg = float(base_config.get("stage0_threshold", 0.7)) if base_config else 0.7
    except Exception:
        thresh_dbg = 0.7
    print(f"[TEMP:DEBUG] stage0: pred={stage0_pred}, prob={stage0_prob}, thresh={thresh_dbg}, X_clf_shape={X_clf.shape}")

    # Interpret stage0_pred: 1 => NEAR_TERM, 0 => LONG_TERM (match notebook labeling)
    # LONG_TERM -> Base model
    if stage0_pred == 0:
        if models.get("base") is None:
            raise RuntimeError("Base model not loaded")
        base_log = float(models["base"].predict(X_base).ravel()[0])
        base_hr = float(np.expm1(base_log))
        base_min = base_hr * 60.0

        # TEMP:DEBUG - log base model raw and converted values
        try:
            print(f"[TEMP:DEBUG] LONG_TERM base (interpreting stage0_pred==0 as LONG_TERM): base_log={base_log}, base_hr(expm1)={base_hr}, base_min_before_clamp={base_hr*60.0}, X_base_shape={X_base.shape}")
        except Exception:
            pass

        base_min = _clamp_rul_minutes("base", base_min)
        print(f"[TEMP:DEBUG] LONG_TERM base: base_min_after_clamp={base_min}")

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

        # TEMP:DEBUG - log base model raw and converted values
        try:
            print(f"[TEMP:DEBUG] NEAR_TERM base: base_log={base_log}, base_hr(expm1)={base_hr}, base_min_before_clamp={base_hr*60.0}, X_base_shape={X_base.shape}")
        except Exception:
            pass

        base_min = _clamp_rul_minutes("base", base_min)
        print(f"[TEMP:DEBUG] NEAR_TERM base: base_min_after_clamp={base_min}")

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

        # TEMP:DEBUG - CRITICAL fallback values
        try:
            print(f"[TEMP:DEBUG] CRITICAL fallback base: base_log={base_log}, base_hr(expm1)={base_hr}, base_min_before_clamp={base_hr*60.0}, X_base_shape={X_base.shape}")
        except Exception:
            pass

        base_min = _clamp_rul_minutes("base", base_min)
        print(f"[TEMP:DEBUG] CRITICAL fallback base: base_min_after_clamp={base_min}")

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

        # TEMP:DEBUG - Stage2A raw and converted values
        try:
            print(f"[TEMP:DEBUG] STAGE2A: pred_log={pred_log}, pred_min_before_clamp={pred_min}, X_clf_shape={X_clf.shape}")
        except Exception:
            pass

        # clamp according to config (min from base_config and max from stage2a_config)
        pred_min = _clamp_rul_minutes("stage2a", pred_min)
        print(f"[TEMP:DEBUG] STAGE2A: pred_min_after_clamp={pred_min}")

        return {
                    "rul_min": pred_min,
                    "regime": "CRITICAL",
                    "active_model": "Stage-2A",
                    "stage0_prob": stage0_prob,
                    "stage1_prob": stage1_prob
                }
    except Exception as e:
        print(f"[TEMP:DEBUG] STAGE2A failed: {e}")
        return {
                    "rul_min": 0.0,
                    "regime": "CRITICAL",
                    "active_model": "Stage-2A (failed)",
                    "stage0_prob": stage0_prob,
                    "stage1_prob": stage1_prob,
                    "notes": str(e)
                }

def _clamp_rul_minutes(model: str, val_min: float) -> float:
    """Clamp model outputs to training pipeline operating ranges.

    - Base model: enforce minimum RUL (minutes) from base_config['min_rul_hours'] if available
      (fallback to 5000 minutes if not present) to match training pipeline behavior.
    - Stage-2A: enforce maximum RUL if present in stage2a_config['max_rul_min'] (unchanged).
    """
    global base_config, stage2a_config
    v = float(val_min)

    # Minimum clamp for Base model (match training notebook behavior)
    min_min = None
    if model == "base":
        try:
            if base_config and "min_rul_hours" in base_config:
                min_min = float(base_config["min_rul_hours"]) * 60.0
            else:
                # Fallback to the notebook's BASE_MODEL_MIN_RUL_MIN
                min_min = 5000.0
        except Exception:
            min_min = 5000.0

    if min_min is not None:
        try:
            v = max(v, float(min_min))
            try:
                print(f"[TEMP:DEBUG] Applied base min clamp: min_min={min_min}, value_after_min_clamp={v}")
            except Exception:
                pass
        except Exception:
            pass

    # Maximum: prefer stage2a_config['max_rul_min'] for Stage-2A only
    max_min = None
    if model == "stage2a" and stage2a_config and "max_rul_min" in stage2a_config:
        try:
            max_min = float(stage2a_config["max_rul_min"])
        except Exception:
            max_min = None

    if max_min is not None:
        v = min(v, max_min)

    return float(max(0.0, v))