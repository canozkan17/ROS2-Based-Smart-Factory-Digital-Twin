import numpy as np
import pandas as pd
import os



# GLOBAL SETTINGS
SEED = 42
RNG = np.random.default_rng(seed=SEED)

# Dataset 1: LONG LIFE (Base Model)
LONG_MIN_RUL = 120_000
LONG_MAX_RUL = 720_000
LONG_N_CYCLES = 50  # Increased for better generalization

# Dataset 2: SHORT LIFE (Two-Stage Models)
SHORT_MIN_RUL = 600
SHORT_MAX_RUL = 3_000
SHORT_N_CYCLES = 150  # Increased for Stage-2A accuracy

# Critical region threshold for Hydraulic Press
CRITICAL_RUL_THRESHOLD = 600



# SINGLE CYCLE GENERATOR
def generate_cycle(cycle_id: int, total_rul: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a single Hydraulic Press cycle until failure.
    Dominant failure mode: Seal wear
    Background effects: Oil degradation + mild valve / misalignment effects
    """

    t = np.arange(total_rul, dtype=float)
    current_rul = total_rul - t - 1
    fraction = t / total_rul


    # CRITICAL REGION BOOST (RUL <= 600)
    critical_mask = current_rul <= CRITICAL_RUL_THRESHOLD
    critical_boost = np.where(
                                critical_mask,
                                1.0 + (CRITICAL_RUL_THRESHOLD - current_rul) / CRITICAL_RUL_THRESHOLD * 2.0,  # Stronger boost
                                1.0
                            )


    # HYDRAULIC PRESS SENSOR PARAMETERS
    # Hydraulic pressure (bar) - seal wear dominant
    base_pressure = rng.uniform(180.0, 220.0)
    pressure_drop_rate = rng.uniform(0.00008, 0.00025)  
    pressure_noise = rng.uniform(0.3, 1.5)  

    # Oil temperature (°C) - oil degradation
    base_oil_temp = rng.uniform(45.0, 65.0)
    oil_temp_rate = rng.uniform(0.0005, 0.0015)  
    oil_temp_noise = rng.uniform(0.08, 0.4)

    # Oil contamination index (dimensionless)
    base_contamination = rng.uniform(0.5, 2.0)
    contamination_growth = rng.uniform(0.00004, 0.00012)
    contamination_noise = rng.uniform(0.008, 0.03)

    # Ram position deviation (mm) - misalignment
    base_ram_dev = rng.uniform(0.01, 0.05)
    ram_dev_growth = rng.uniform(0.00002, 0.00008)
    ram_dev_noise = rng.uniform(0.0008, 0.003)

    # Press force / tonnage (tons)
    base_force = rng.uniform(80.0, 120.0)
    force_loss_rate = rng.uniform(0.00005, 0.00015)
    force_noise = rng.uniform(0.2, 1.0)

    # Frame / ram vibration (mm/s)
    base_vibration = rng.uniform(0.1, 0.5)
    failure_vibration = rng.uniform(4.0, 10.0)  # Higher failure vibration
    vibration_noise = rng.uniform(0.015, 0.07)

    # Hydraulic flow rate (L/min)
    base_flow = rng.uniform(90.0, 130.0)
    flow_loss_rate = rng.uniform(0.00006, 0.00018)
    flow_noise = rng.uniform(0.2, 0.8)

    # Motor current (A)
    base_current = rng.uniform(30.0, 55.0)
    current_growth_rate = rng.uniform(0.00008, 0.0003)
    current_noise = rng.uniform(0.08, 0.4)


    # SENSOR SIGNAL GENERATION
    hydraulic_pressure = (
                            base_pressure
                            - pressure_drop_rate * t * critical_boost
                            + rng.normal(0.0, pressure_noise * (1.0 + fraction), size=total_rul)
                        )

    oil_temperature = (
                        base_oil_temp
                        + oil_temp_rate * t * critical_boost
                        + rng.normal(0.0, oil_temp_noise, size=total_rul)
                    )

    oil_contamination = (
                            base_contamination
                            + contamination_growth * t * critical_boost
                            + rng.normal(0.0, contamination_noise * (1.0 + fraction), size=total_rul)
                        )

    ram_position_deviation = (
                                base_ram_dev
                                + ram_dev_growth * t * critical_boost
                                + rng.normal(0.0, ram_dev_noise, size=total_rul)
                            )

    press_force = (
                    base_force
                    - force_loss_rate * t * critical_boost
                    + rng.normal(0.0, force_noise * (1.0 + 0.5 * fraction), size=total_rul)
                )

    vibration = (
                    base_vibration
                    + (failure_vibration - base_vibration) * (fraction ** 1.5) * critical_boost  # Faster exponential growth
                    + rng.normal(0.0, vibration_noise * (1.0 + 0.5 * fraction), size=total_rul)
                )

    flow_rate = (
                    base_flow
                    - flow_loss_rate * t * critical_boost
                    + rng.normal(0.0, flow_noise * (1.0 + fraction), size=total_rul)
                )

    motor_current = (
                        base_current
                        + current_growth_rate * t * critical_boost
                        + rng.normal(0.0, current_noise * (1.0 + fraction), size=total_rul)
                    )


    # DATAFRAME
    df = pd.DataFrame({
                        "cycle_id": cycle_id,
                        "time_min": t.astype(int),
                        "total_rul": int(total_rul),
                        "current_rul": current_rul.astype(int),

                        "hydraulic_pressure": hydraulic_pressure.astype(float),
                        "oil_temperature": oil_temperature.astype(float),
                        "oil_contamination": oil_contamination.astype(float),
                        "ram_position_deviation": ram_position_deviation.astype(float),
                        "press_force": press_force.astype(float),
                        "vibration": vibration.astype(float),
                        "flow_rate": flow_rate.astype(float),
                        "motor_current": motor_current.astype(float),
                    })

    return df



# DATASET GENERATOR
def generate_dataset(n_cycles: int, min_rul: int, max_rul: int, output_path: str):
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
                        output_path="synthetic_hydraulic_press_data_long_life.csv"
                    )

    # Dataset for TWO-STAGE MODELS
    generate_dataset(
                        n_cycles=SHORT_N_CYCLES,
                        min_rul=SHORT_MIN_RUL,
                        max_rul=SHORT_MAX_RUL,
                        output_path="synthetic_hydraulic_press_data_short_life.csv"
                    )