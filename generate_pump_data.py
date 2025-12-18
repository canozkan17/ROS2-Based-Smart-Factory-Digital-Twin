import numpy as np
import pandas as pd
import os


# GLOBAL SETTINGS
SEED = 42
RNG = np.random.default_rng(seed=SEED)

# Dataset 1: LONG LIFE (Base Model)
LONG_MIN_RUL = 30000
LONG_MAX_RUL = 48000
LONG_N_CYCLES = 60

# Dataset 2: SHORT LIFE (For Two-Stage Models)
SHORT_MIN_RUL = 300
SHORT_MAX_RUL = 600
SHORT_N_CYCLES = 120

# CYCLE GENERATOR 
def generate_cycle(cycle_id: int, total_rul: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a single pump cycle until failure.
    """

    t = np.arange(total_rul, dtype=float)
    current_rul = total_rul - t - 1
    fraction = t / total_rul

    base_vib = rng.uniform(0.02, 0.3)
    failure_vib = rng.uniform(2.0, 6.0)
    vib_noise_scale = rng.uniform(0.01, 0.1)

    base_temp = rng.uniform(40.0, 60.0)
    k_temp = rng.uniform(0.005, 0.02)
    temp_noise_scale = rng.uniform(0.05, 0.3)

    base_pressure = rng.uniform(7.6, 8.4)
    k_pressure = rng.uniform(0.0005, 0.003)
    pressure_noise_scale = rng.uniform(0.01, 0.08)

    coupling_factor = rng.uniform(0.4, 0.9)
    vib_motor_noise = rng.uniform(0.005, 0.05)

    # Critical region (last 100 minutes)
    critical_mask = current_rul <= 100
    critical_boost = np.where(
                                critical_mask,
                                1.0 + (100 - current_rul) / 100 * 0.5,
                                1.0
                            )

    vibration = (
                    base_vib
                    + (failure_vib - base_vib) * (fraction ** 2) * critical_boost
                    + rng.normal(0.0, vib_noise_scale * (1.0 + fraction), size=total_rul)
                )

    temp_motor = (
                    base_temp + k_temp * t * critical_boost
                    + rng.normal(0.0, temp_noise_scale, size=total_rul)
                )

    pressure = (
                    base_pressure - k_pressure * t * critical_boost
                    + rng.normal(0.0, pressure_noise_scale * (1.0 + 0.5 * fraction), size=total_rul)
                )

    vib_motor = (
                    vibration * coupling_factor
                    + rng.normal(0.0, vib_motor_noise * (1.0 + fraction), size=total_rul)
                )

    df = pd.DataFrame({
                        "cycle_id": cycle_id,
                        "time_min": t.astype(int),
                        "total_rul": int(total_rul),
                        "current_rul": current_rul.astype(int),
                        "vibration": vibration.astype(float),
                        "temp_motor": temp_motor.astype(float),
                        "pressure": pressure.astype(float),
                        "vib_motor": vib_motor.astype(float),
                    })

    return df



# DATASET GENERATOR

def generate_dataset(
                    n_cycles: int,
                    min_rul: int,
                    max_rul: int,
                    output_path: str
                ):
    dfs = []

    for cycle_id in range(1, n_cycles + 1):
        total_rul = int(RNG.integers(min_rul, max_rul + 1))
        df_cycle = generate_cycle(cycle_id, total_rul, RNG)
        dfs.append(df_cycle)

    df_all = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_all.to_csv(output_path, index=False)

    print(f"\nSaved dataset to {output_path}")
    print(f"Total cycles: {df_all['cycle_id'].nunique()}")
    print(f"Total samples: {len(df_all)}")
    print(f"RUL range: {df_all['current_rul'].min()} - {df_all['current_rul'].max()}")

    critical_ratio = (df_all["current_rul"] <= 20).mean() * 100
    print(f"Critical region (RUL ≤ 20): {critical_ratio:.2f}%")



# MAIN
if __name__ == "__main__":

    # Dataset for BASE MODEL
    generate_dataset(
                        n_cycles=LONG_N_CYCLES,
                        min_rul=LONG_MIN_RUL,
                        max_rul=LONG_MAX_RUL,
                        output_path="synthetic_pump_data_long_life.csv"
                    )

    # Dataset for TWO-STAGE MODELS
    generate_dataset(
                        n_cycles=SHORT_N_CYCLES,
                        min_rul=SHORT_MIN_RUL,
                        max_rul=SHORT_MAX_RUL,
                        output_path="synthetic_pump_data_short_life.csv"
                    )
