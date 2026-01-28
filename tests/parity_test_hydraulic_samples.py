"""Parity test using representative samples (rows) across RUL regimes.
For each selected row we build a sliding window of previous N samples from the same cycle and
compare handler.predict(...) with manual model pipeline results.
"""
import sys, os
sys.path.insert(0, 'src')
from system_nodes.prediction_handler import hydraulic_press_prediction_handler as h
import pandas as pd
import numpy as np

# Config
WINDOW = 50
np.set_printoptions(precision=6, suppress=True)

# Load models
h.reset_state()
h.load_models()

# Datasets
df_long = pd.read_csv('synthetic_hydraulic_press_data_long_life.csv')
df_short = pd.read_csv('synthetic_hydraulic_press_data_short_life.csv')

# Find representative rows
r_long = df_long[df_long['current_rul'] >= 5000].iloc[0]
r_near = df_long[(df_long['current_rul'] > 600) & (df_long['current_rul'] < 5000)].iloc[0]
r_crit = df_short[df_short['current_rul'] <= 100].iloc[0]

candidates = [
                ('long_sample', int(r_long['cycle_id']), r_long['time_min']),
                ('near_sample', int(r_near['cycle_id']), r_near['time_min']),
                ('critical_sample', int(r_crit['cycle_id']), r_crit['time_min'])
            ]

results = []
for name, cid, tmin in candidates:
    # get window of up to WINDOW samples from same cycle ending at tmin
    dfc = df_long[df_long['cycle_id']==cid] if name != 'critical_sample' else df_short[df_short['cycle_id']==cid]
    dfc = dfc.sort_values('time_min')
    # find index position of tmin
    mask = dfc['time_min'] <= tmin
    if not mask.any():
        print('No samples <= tmin for', name); continue
    df_window = dfc[mask].tail(WINDOW)
    sensor_data = []
    for _, r in df_window.iterrows():
        d = r.to_dict(); d['elapsed_minutes'] = float(d.get('time_min', 0)); sensor_data.append(d)

    # handler predict
    h.reset_state(); handler_pred = h.predict(sensor_data)

    # manual pipeline
    h.reset_state(); h.normalize_sensors_and_features(sensor_data); engineered = h.apply_feature_engineering(sensor_data); X_clf, X_base = h.prepare_feature_matrices(sensor_data, engineered)

    # stage0
    if h.models.get('stage0') is not None:
        try:
            probs = h.models['stage0'].predict_proba(X_clf); p_arr = np.asarray(probs)
            if p_arr.ndim==2 and p_arr.shape[1]>=2: s0_prob = float(p_arr[:,1].ravel()[0])
            else: s0_prob = float(np.asarray(probs).ravel()[0])
        except Exception:
            s0_prob = float(np.asarray(h.models['stage0'].predict(X_clf)).ravel()[0])
    else:
        s0_prob = 1.0
    s0_pred = int(s0_prob >= 0.7)

    manual = {'stage0_prob': s0_prob}
    if s0_pred==0:
        base_log = float(h.models['base'].predict(X_base).ravel()[0]); base_hr = float(np.expm1(base_log)); base_min = base_hr*60; base_min = h._clamp_rul_minutes('base', base_min); manual.update({'rul_min':float(base_min),'regime':'LONG_TERM'})
    else:
        # stage1
        if h.models.get('stage1') is not None:
            try:
                probs1 = h.models['stage1'].predict_proba(X_clf); p1 = np.asarray(probs1); stage1_prob = float(p1[:,1].ravel()[0])
            except Exception:
                stage1_prob = float(np.asarray(h.models['stage1'].predict(X_clf)).ravel()[0])
            thresh1 = float(h.stage1_config.get('optimal_prob_threshold', 0.5)) if h.stage1_config else 0.5
            stage1_pred = int(stage1_prob >= thresh1)
        else:
            stage1_prob = 0.0; stage1_pred = 0
        if stage1_pred==0:
            base_log = float(h.models['base'].predict(X_base).ravel()[0]); base_hr = float(np.expm1(base_log)); base_min = base_hr*60; base_min = h._clamp_rul_minutes('base', base_min)
            manual.update({'rul_min':float(base_min),'regime':'NEAR_TERM','stage1_prob':stage1_prob})
        else:
            if h.models.get('stage2a') is None:
                base_log = float(h.models['base'].predict(X_base).ravel()[0]); base_hr = float(np.expm1(base_log)); base_min = base_hr*60; base_min = h._clamp_rul_minutes('base', base_min); manual.update({'rul_min':float(base_min),'regime':'CRITICAL','stage1_prob':stage1_prob})
            else:
                plog = float(h.models['stage2a'].predict(X_clf).ravel()[0]); pmin = float(np.expm1(plog)); pmin = h._clamp_rul_minutes('stage2a', pmin); manual.update({'rul_min':float(pmin),'regime':'CRITICAL','stage1_prob':stage1_prob})

    results.append((name, cid, tmin, handler_pred, manual))

# Print
for name, cid, tmin, handler_pred, manual in results:
    print('\n===', name, 'cycle', cid, 'tmin', tmin, '===')
    print('Handler:', handler_pred)
    print('Manual :', manual)
    diff = None
    if handler_pred.get('rul_min') is not None and manual.get('rul_min') is not None:
        diff = float(handler_pred['rul_min']) - float(manual['rul_min'])
    print('Diff (handler - manual):', diff)

print('\nSample parity test done')
