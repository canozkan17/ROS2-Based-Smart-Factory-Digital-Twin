"""Parity test: compare prediction outputs from handler.predict(...) with manual pipeline using models and configs
"""
import sys, os
sys.path.insert(0, 'src')
from system_nodes.prediction_handler import hydraulic_press_prediction_handler as h
import pandas as pd
import numpy as np

# Load models
h.reset_state()
h.load_models()
print('Models loaded:')
for k,v in h.models.items():
    print(' ', k, '=>', type(v).__name__ if v is not None else None)

# Paths
LONG = 'synthetic_hydraulic_press_data_long_life.csv'
SHORT = 'synthetic_hydraulic_press_data_short_life.csv'

# Read datasets
df_long = pd.read_csv(LONG)
df_short = pd.read_csv(SHORT)

# helper to pick cycles by last-sample current_rul
def last_samples(df):
    return df.sort_values(['cycle_id','time_min']).groupby('cycle_id').tail(1)

last_long = last_samples(df_long)
last_short = last_samples(df_short)

BASE_MIN = int(h.base_config.get('min_rul_hours', 83.33333333333333)*60)
print('BASE_MODEL_MIN_RUL_MIN:', BASE_MIN)

# candidates
c_candidates = []
# 1. Long-term (>= BASE_MIN)
rows = last_long[last_long['current_rul'] >= BASE_MIN]
if len(rows)>0:
    c_candidates.append(('long', int(rows.iloc[0]['cycle_id']), df_long))
# 2. Edge: just below base min
rows = last_long[(last_long['current_rul'] < BASE_MIN) & (last_long['current_rul'] >= BASE_MIN-100)]
if len(rows)>0:
    c_candidates.append(('edge', int(rows.iloc[0]['cycle_id']), df_long))
# 3. Near-term non-critical (600 < rul < BASE_MIN)
rows = last_long[(last_long['current_rul']>600) & (last_long['current_rul']<BASE_MIN)]
if len(rows)>0:
    c_candidates.append(('near', int(rows.iloc[0]['cycle_id']), df_long))
# 4. Critical (<=100) from short
rows = last_short[last_short['current_rul'] <= 100]
if len(rows)>0:
    c_candidates.append(('critical', int(rows.iloc[0]['cycle_id']), df_short))

if not c_candidates:
    print('No cycles found for tests; exiting')
    sys.exit(1)

results = []
for name, cid, df in c_candidates:
    print('\nProcessing', name, 'cycle', cid)
    cycle_df = df[df['cycle_id']==cid].sort_values('time_min')
    sensor_data = []
    for _, r in cycle_df.iterrows():
        d = r.to_dict()
        d['elapsed_minutes'] = float(d.get('time_min', d.get('elapsed_minutes', 0)))
        sensor_data.append(d)

    # handler prediction
    h.reset_state()
    pred_handler = h.predict(sensor_data)

    # manual pipeline
    h.reset_state()
    h.normalize_sensors_and_features(sensor_data)
    engineered = h.apply_feature_engineering(sensor_data)
    X_clf, X_base = h.prepare_feature_matrices(sensor_data, engineered)

    # Stage0
    s0_prob = None
    s0_pred = None
    if h.models.get('stage0') is not None:
        try:
            probs = h.models['stage0'].predict_proba(X_clf)
            probs_arr = np.asarray(probs)
            if probs_arr.ndim==2 and probs_arr.shape[1]>=2:
                s0_prob = float(probs_arr[:,1].ravel()[0])
            else:
                s0_prob = float(np.asarray(probs).ravel()[0])
        except Exception:
            s0_prob = float(np.asarray(h.models['stage0'].predict(X_clf)).ravel()[0])
        s0_pred = int(s0_prob >= 0.7)
    else:
        s0_prob = 1.0; s0_pred = 1

    manual = {'stage0_prob': s0_prob}
    if s0_pred==0:
        base_log = float(h.models['base'].predict(X_base).ravel()[0])
        base_hr = float(np.expm1(base_log))
        base_min = base_hr * 60.0
        base_min = h._clamp_rul_minutes('base', base_min)
        manual.update({'rul_min': float(base_min), 'regime':'LONG_TERM', 'active_model':'Base Model'})
    else:
        # stage1
        if h.models.get('stage1') is not None:
            try:
                probs1 = h.models['stage1'].predict_proba(X_clf)
                probs1_arr = np.asarray(probs1)
                stage1_prob = float(probs1_arr[:,1].ravel()[0])
            except Exception:
                stage1_prob = float(np.asarray(h.models['stage1'].predict(X_clf)).ravel()[0])
            thresh1 = float(h.stage1_config.get('optimal_prob_threshold', 0.5)) if h.stage1_config else 0.5
            stage1_pred = int(stage1_prob >= thresh1)
        else:
            stage1_prob = 0.0; stage1_pred = 0

        if stage1_pred==0:
            base_log = float(h.models['base'].predict(X_base).ravel()[0])
            base_hr = float(np.expm1(base_log))
            base_min = base_hr * 60.0
            base_min = h._clamp_rul_minutes('base', base_min)
            manual.update({'rul_min': float(base_min), 'regime':'NEAR_TERM', 'active_model':'Base Model', 'stage1_prob': stage1_prob})
        else:
            if h.models.get('stage2a') is None:
                base_log = float(h.models['base'].predict(X_base).ravel()[0])
                base_hr = float(np.expm1(base_log))
                base_min = base_hr * 60.0
                base_min = h._clamp_rul_minutes('base', base_min)
                manual.update({'rul_min': float(base_min), 'regime':'CRITICAL', 'active_model':'Base Model (fallback)', 'stage1_prob': stage1_prob})
            else:
                pred_log = float(h.models['stage2a'].predict(X_clf).ravel()[0])
                pred_min = float(np.expm1(pred_log))
                pred_min = h._clamp_rul_minutes('stage2a', pred_min)
                manual.update({'rul_min': float(pred_min), 'regime':'CRITICAL', 'active_model':'Stage-2A', 'stage1_prob': stage1_prob})

    results.append((name, cid, pred_handler, manual))

# Show comparison
for name, cid, handler_pred, manual_pred in results:
    print('\n====', name, 'cycle', cid, '====')
    print('Handler:', handler_pred)
    print('Manual :', manual_pred)
    hmin = handler_pred.get('rul_min')
    mmin = manual_pred.get('rul_min')
    if hmin is None and mmin is None:
        print('Both returned None rul_min')
    else:
        print('Diff (handler - manual):', None if hmin is None or mmin is None else (float(hmin)-float(mmin)))

print('\nParity test complete')
