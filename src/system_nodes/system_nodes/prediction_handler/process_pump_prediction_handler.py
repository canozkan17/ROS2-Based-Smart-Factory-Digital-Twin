import os
import json
import xgboost as xgb


current_dir = os.path.dirname(os.path.abspath(__file__))

# all .json files will be in the same dir in RPI deployment [Correct in Edge Deployment]
root_path = os.path.abspath(os.path.join(current_dir, "../../../.."))
sources_path = os.path.join(root_path, "models_and_features", "process_pump")

base_model = None
stg1_classifier = None
stg2a_regressor = None
stg2b_regressor = None
selected_features = []

def load_models_and_features():
    global base_model, stg1_classifier, stg2a_regressor, stg2b_regressor

    base_model = xgb.XGBRegressor()
    stg1_classifier = xgb.XGBClassifier()
    stg2a_regressor = xgb.XGBRegressor()
    stg2b_regressor = xgb.XGBRegressor()
    
    model = os.path.join(sources_path, "process_pump_base_model.json")
    base_model.load_model(model)

    model = os.path.join(sources_path, "process_pump_stage1_classifier.json")
    stg1_classifier.load_model(model)

    model = os.path.join(sources_path, "process_pump_stage2a_regressor.json")
    stg2a_regressor.load_model(model)

    model = os.path.join(sources_path, "process_pump_stage2b_regressor.json")
    stg2b_regressor.load_model(model)

    features_path = os.path.join(sources_path, "selected_process_pump_features.json")
    with open(features_path, "r") as f:
        global selected_features
        selected_features = json.load(f)
    print("Models and features loaded successfully.")


SENSORS = ['vibration', 'temp_motor', 'pressure', 'vib_motor']

def engineer_features(df, WINDOW_SHORT=5, WINDOW_LONG=20):
    df_feat = df.copy()
    df_feat = df_feat.sort_values(['cycle_id', 'time_min']).reset_index(drop=True)
    grouped = df_feat.groupby('cycle_id')
    
    # Original rolling features
    for s in SENSORS:
        df_feat[f"{s}_roll_mean_{WINDOW_SHORT}"] = (
                                                        grouped[s].rolling(window=WINDOW_SHORT, min_periods=1).mean()
                                                        .reset_index(level=0, drop=True)
                                                    )
        df_feat[f"{s}_roll_mean_{WINDOW_LONG}"] = (
                                                    grouped[s].rolling(window=WINDOW_LONG, min_periods=1).mean()
                                                    .reset_index(level=0, drop=True)
                                                )
        df_feat[f"{s}_roll_std_{WINDOW_LONG}"] = (
                                                    grouped[s].rolling(window=WINDOW_LONG, min_periods=2).std()
                                                    .reset_index(level=0, drop=True).fillna(0)
                                                )
        df_feat[f"{s}_dev_long"] = df_feat[s] - df_feat[f"{s}_roll_mean_{WINDOW_LONG}"]
        df_feat[f"{s}_slope_{WINDOW_LONG}"] = (
                                                    grouped[f"{s}_roll_mean_{WINDOW_LONG}"].diff(WINDOW_LONG).fillna(0)
                                                )
    
    # Original interaction features
    with np.errstate(divide='ignore', invalid='ignore'):
        coupling = (df_feat['vib_motor'] / df_feat['vibration']).replace([np.inf, -np.inf], np.nan)

    df_feat['motor_pump_coupling'] = coupling.fillna(1.0).clip(0, 10)
    
    df_feat['vibration_energy'] = (
                                        grouped['vibration'].rolling(window=WINDOW_SHORT, min_periods=1).mean()
                                        .reset_index(level=0, drop=True) ** 2
                                    )
    

    pressure_std = df_feat['pressure_roll_std_20'].replace(0, 1e-6)
    df_feat['pressure_stability'] = 1.0 / pressure_std
    
    # Critical aware features:
    # Acceleration features (rate of change of slope)
    for s in ['vibration', 'temp_motor']:
        df_feat[f"{s}_acceleration"] = grouped[f"{s}_slope_{WINDOW_LONG}"].diff(5).fillna(0)
    
    # Volatility surge (sudden std increase)
    for s in SENSORS:
        roll_std_short = grouped[s].rolling(window=5, min_periods=2).std().reset_index(level=0, drop=True)
        df_feat[f"{s}_volatility_surge"] = roll_std_short - df_feat[f"{s}_roll_std_{WINDOW_LONG}"]
    
    # Multi-sensor degradation indicator
    vib_norm = (df_feat['vibration'] - df_feat['vibration'].min()) / (df_feat['vibration'].max() - df_feat['vibration'].min() + 1e-6)
    temp_norm = (df_feat['temp_motor'] - df_feat['temp_motor'].min()) / (df_feat['temp_motor'].max() - df_feat['temp_motor'].min() + 1e-6)
    pressure_inv = 1 - (df_feat['pressure'] - df_feat['pressure'].min()) / (df_feat['pressure'].max() - df_feat['pressure'].min() + 1e-6)
    df_feat['degradation_index'] = (vib_norm + temp_norm + pressure_inv) / 3
    
    # Recent max features (captures spikes)
    for s in ['vibration', 'vib_motor']:
        df_feat[f"{s}_max_10"] = grouped[s].rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    
    # Clean up
    for col in df_feat.columns:
        if col in ['cycle_id', 'time_min', 'current_rul', 'total_rul', 'timestamp']:
            continue
        df_feat[col] = grouped[col].transform(lambda x: x.ffill().bfill())
    
    df_feat.replace([np.inf, -np.inf], 0, inplace=True)
    df_feat.fillna(0, inplace=True)
    
    return df_feat

