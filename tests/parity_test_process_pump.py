"""Parity test: compare prediction outputs from handler.predict(...) with manual pipeline using models and configs

Similar style to existing hydraulic parity test.
"""
import sys, os
sys.path.insert(0, 'src')
from system_nodes.prediction_handler import process_pump_prediction_handler as h
import pandas as pd
import numpy as np

# Load models
h.reset_state()
h.load_models()
print('Models loaded:')
# print a summary of available models
try:
    print(' base:', type(h._base_model).__name__ if h._base_model is not None else None)
    print(' stage1:', type(h._stage1_classifier).__name__ if h._stage1_classifier is not None else None)
    print(' stage2a:', type(h._stage2a_regressor).__name__ if h._stage2a_regressor is not None else None)
except Exception:
    pass

# Paths
LONG = 'synthetic_pump_data_long_life.csv'
SHORT = 'synthetic_pump_data_short_life.csv'

# Read datasets
df_long = pd.read_csv(LONG)
df_short = pd.read_csv(SHORT)

# helper to pick cycles by last-sample current_rul
def last_samples(df):
    return df.sort_values(['cycle_id','time_min']).groupby('cycle_id').tail(1)

last_long = last_samples(df_long)
last_short = last_samples(df_short)

c_candidates = []
# 1. Critical (<=20) from short
rows = last_short[last_short['current_rul'] <= 20]
if len(rows) > 0:
    c_candidates.append(('critical', int(rows.iloc[0]['cycle_id']), df_short))
# 2. Near-term (<=100 but >20) from short
rows = last_short[(last_short['current_rul'] > 20) & (last_short['current_rul'] <= 100)]
if len(rows) > 0:
    c_candidates.append(('near_critical', int(rows.iloc[0]['cycle_id']), df_short))
# 3. Long safe (large rul) from long
rows = last_long[last_long['current_rul'] >= 10000]
if len(rows) > 0:
    c_candidates.append(('safe_long', int(rows.iloc[0]['cycle_id']), df_long))
# 4. Edge case: insufficient data (single sample)
c_candidates.append(('insufficient', int(last_short.iloc[0]['cycle_id']), df_short.head(1)))

results = []
for name, cid, df in c_candidates:
    print('\nProcessing', name, 'cycle', cid)
    cycle_df = df[df['cycle_id'] == cid].sort_values('time_min')
    # For normal cases take full cycle history (all rows for that cycle)
    if name != 'insufficient':
        sensor_data = []
        for _, r in cycle_df.iterrows():
            d = {
                    'cycle': int(r['cycle_id']),
                    'vibration': float(r['vibration']),
                    'temp_motor': float(r['temp_motor']),
                    'pressure': float(r['pressure']),
                    'vib_motor': float(r['vib_motor']),
                }
            sensor_data.append(d)
    else:
        # single-row history (insufficient)
        r = df.iloc[0]
        sensor_data = [{'cycle': int(r['cycle_id']), 'vibration': float(r['vibration']), 'temp_motor': float(r['temp_motor']), 'pressure': float(r['pressure']), 'vib_motor': float(r['vib_motor'])}]

    # handler prediction
    h.reset_state()
    pred_handler = h.predict(sensor_data)

    # manual pipeline using internal functions
    h.reset_state()
    h._sync_buffer_from_history(sensor_data)
    X = h._compute_features()
    if X is None:
        manual = {'rul_min': -1.0, 'unit': 'min', 'stage': 'INSUFFICIENT_DATA', 'crit_prob': 0.0}
    else:
        # run inference (prefer ONNX path if available)
        if hasattr(h._base_model, 'run'):
            crit_prob, base_rul, stg2a_rul = h._run_inference_onnx(X)
        else:
            crit_prob, base_rul, stg2a_rul = h._run_inference_xgb(X)

        # simulate decision logic with hysteresis starting from False
        _in_critical_state = False
        THRESH_ENTER = h.THRESH_ENTER
        THRESH_EXIT = h.THRESH_EXIT
        CRITICAL_MAX_RUL = h.CRITICAL_MAX_RUL

        if not _in_critical_state:
            if crit_prob >= THRESH_ENTER:
                _in_critical_state = True
                if np.isfinite(stg2a_rul) and stg2a_rul <= CRITICAL_MAX_RUL:
                    stage = 'STAGE_2A'
                    final_rul = stg2a_rul
                else:
                    stage = 'BASE'
                    final_rul = base_rul
            else:
                stage = 'BASE'
                final_rul = base_rul
        else:
            if crit_prob <= THRESH_EXIT:
                _in_critical_state = False
                stage = 'BASE'
                final_rul = base_rul
            else:
                if np.isfinite(stg2a_rul) and stg2a_rul <= CRITICAL_MAX_RUL:
                    stage = 'STAGE_2A'
                    final_rul = stg2a_rul
                else:
                    stage = 'BASE'
                    final_rul = base_rul

        if not np.isfinite(final_rul):
            final_rul = float(base_rul) if np.isfinite(base_rul) else 0.0
        final_rul = float(np.clip(final_rul, 0.0, 48000.0))

        manual = {'rul_min': float(final_rul), 'unit': 'min', 'stage': stage, 'crit_prob': float(np.clip(crit_prob, 0.0, 1.0))}

    results.append((name, cid, pred_handler, manual))

# Show comparison
for name, cid, handler_pred, manual_pred in results:
    print('\n====', name, 'cycle', cid, '====')
    print('Handler:', handler_pred)
    print('Manual :', manual_pred)

    # compare
    try:
        assert handler_pred.get('stage') == manual_pred.get('stage')
        # allow small tolerance for floats
        assert abs(float(handler_pred.get('rul_min', -9999.0)) - float(manual_pred.get('rul_min', -9999.0))) < 1e-3
        assert abs(float(handler_pred.get('crit_prob', 0.0)) - float(manual_pred.get('crit_prob', 0.0))) < 1e-3
        print('Parity: OK')
    except AssertionError:
        print('Parity: MISMATCH')

print('\nProcess pump parity test complete')
