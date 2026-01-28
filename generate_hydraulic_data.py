import numpy as np
import pandas as pd
import os



# GLOBAL SETTINGS
SEED = 42
RNG = np.random.default_rng(seed=SEED)

# Dataset 1: LONG LIFE (Base Model + Stage-0)
# Range: 5,000 - 720,000 minutes (~3.5 days to 500 days)
# This gives Stage-0 enough NEAR_TERM samples to learn regime transitions
LONG_MIN_RUL = 5_000      # Lowered from 120,000 to include transition zone
LONG_MAX_RUL = 720_000
LONG_N_CYCLES = 80        # Increased for better coverage across regimes

# Dataset 2: SHORT LIFE (Stage-1 + Stage-2A)
# Range: 100 - 5,000 minutes (~1.7 hours to 3.5 days)
# Overlaps with NEAR_TERM threshold for seamless model handoff
SHORT_MIN_RUL = 100       # Lowered for better critical region coverage
SHORT_MAX_RUL = 5_000     # Increased to match Stage-0 NEAR_TERM threshold
SHORT_N_CYCLES = 200      # Increased for Stage-2A accuracy

# Critical region thresholds (aligned with pipeline)
CRITICAL_RUL_THRESHOLD = 600    # Stage-1/2A boundary
STAGE0_NEAR_TERM_THRESHOLD = 5_000  # Stage-0 regime boundary



# SINGLE CYCLE GENERATOR
def generate_cycle(cycle_id: int, total_rul: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a single Hydraulic Press cycle until failure.
    Dominant failure mode: Seal wear
    Background effects: Oil degradation + mild valve / misalignment effects
    
    Physics-based degradation model:
    - Degradation follows non-linear profile with 3 distinct phases
    - Phase 1 (LONG_TERM): Slow, linear degradation (wear-in complete, stable operation)
    - Phase 2 (NEAR_TERM): Accelerating degradation (wear accumulation, coupling effects)
    - Phase 3 (CRITICAL): Rapid non-linear degradation (failure imminent)
    """

    t = np.arange(total_rul, dtype=float)
    current_rul = total_rul - t - 1
    fraction = t / total_rul

    # PHYSICS-BASED DEGRADATION PROFILE
    # Creates consistent sensor behavior across different total_rul values
    # Key insight: A machine at RUL=600 should look the same whether it started
    # with total_rul=3000 or total_rul=300000
    
    # Absolute RUL-based degradation (independent of total lifecycle)
    # This ensures consistent sensor signatures at same RUL regardless of total_rul
    
    # Define degradation zones based on ABSOLUTE RUL values
    rul_for_near_term = STAGE0_NEAR_TERM_THRESHOLD  # 5000 min
    rul_for_critical = CRITICAL_RUL_THRESHOLD       # 600 min
    
    # Base degradation: linear wear over lifecycle
    base_degradation = fraction
    
    # Near-term acceleration: starts when RUL drops below 5000 min
    # Use non-negative relative distances to avoid invalid values when raising to fractional powers.
    rel_near = np.maximum((rul_for_near_term - current_rul) / rul_for_near_term, 0.0)
    near_term_factor = 1.0 + (rel_near ** 1.5) * 2.0

    # Critical boost: exponential increase when RUL drops below 600 min
    rel_critical = np.maximum((rul_for_critical - current_rul) / rul_for_critical, 0.0)
    critical_boost = 1.0 + (rel_critical ** 2) * 5.0
    
    # Combined degradation factor (multiplicative)
    degradation_factor = base_degradation * near_term_factor * critical_boost


    # HYDRAULIC PRESS SENSOR PARAMETERS
    # Note: Parameters are randomized per-cycle for diversity but degradation
    # profiles are consistent based on absolute RUL values
    
    # Hydraulic pressure (bar) - seal wear dominant
    base_pressure = rng.uniform(180.0, 220.0)
    pressure_drop_rate = rng.uniform(0.00005, 0.00015)
    pressure_noise_base = rng.uniform(0.3, 1.0)

    # Oil temperature (°C) - oil degradation
    base_oil_temp = rng.uniform(45.0, 65.0)
    oil_temp_rate = rng.uniform(0.0003, 0.0010)
    oil_temp_noise = rng.uniform(0.08, 0.3)

    # Oil contamination index (dimensionless)
    base_contamination = rng.uniform(0.5, 2.0)
    contamination_growth = rng.uniform(0.00003, 0.00010)
    contamination_noise = rng.uniform(0.008, 0.025)

    # Ram position deviation (mm) - misalignment
    base_ram_dev = rng.uniform(0.01, 0.05)
    ram_dev_growth = rng.uniform(0.00001, 0.00006)
    ram_dev_noise = rng.uniform(0.0008, 0.002)

    # Press force / tonnage (tons)
    base_force = rng.uniform(80.0, 120.0)
    force_loss_rate = rng.uniform(0.00003, 0.00012)
    force_noise = rng.uniform(0.2, 0.8)

    # Frame / ram vibration (mm/s) - key failure indicator
    base_vibration = rng.uniform(0.1, 0.5)
    failure_vibration = rng.uniform(6.0, 12.0)
    vibration_noise_base = rng.uniform(0.015, 0.05)

    # Hydraulic flow rate (L/min)
    base_flow = rng.uniform(90.0, 130.0)
    flow_loss_rate = rng.uniform(0.00004, 0.00014)
    flow_noise = rng.uniform(0.2, 0.6)

    # Motor current (A)
    base_current = rng.uniform(30.0, 55.0)
    current_growth_rate = rng.uniform(0.00006, 0.00020)
    current_noise_base = rng.uniform(0.08, 0.3)


    # SENSOR SIGNAL GENERATION WITH PHYSICS-BASED DEGRADATION
    # Noise increases with degradation (heteroscedastic noise model)
    noise_multiplier = 1.0 + degradation_factor
    
    hydraulic_pressure = (
                            base_pressure
                            - pressure_drop_rate * t * near_term_factor * critical_boost
                            + rng.normal(0.0, pressure_noise_base, size=total_rul) * noise_multiplier
                        )

    oil_temperature = (
                        base_oil_temp
                        + oil_temp_rate * t * near_term_factor * critical_boost
                        + rng.normal(0.0, oil_temp_noise, size=total_rul) * (1.0 + 0.5 * degradation_factor)
                    )

    oil_contamination = (
                            base_contamination
                            + contamination_growth * t * near_term_factor * critical_boost
                            + rng.normal(0.0, contamination_noise, size=total_rul) * noise_multiplier
                        )

    ram_position_deviation = (
                                base_ram_dev
                                + ram_dev_growth * t * near_term_factor * critical_boost
                                + rng.normal(0.0, ram_dev_noise, size=total_rul) * noise_multiplier
                            )

    press_force = (
                    base_force
                    - force_loss_rate * t * near_term_factor * critical_boost
                    + rng.normal(0.0, force_noise, size=total_rul) * (1.0 + 0.3 * degradation_factor)
                )

    # Vibration: S-curve growth pattern (exponential near failure)
    # This creates distinct signatures for LONG_TERM vs NEAR_TERM vs CRITICAL
    vibration_profile = (failure_vibration - base_vibration) * (
                                                                    # Slow growth in early life
                                                                    0.1 * fraction +
                                                                    # Accelerated growth in near-term
                                                                    0.3 * (rel_near ** 2) +
                                                                    # Rapid growth in critical region
                                                                    0.6 * (rel_critical ** 1.5)
                                                                )
    vibration = (
                    base_vibration
                    + vibration_profile
                    + rng.normal(0.0, vibration_noise_base, size=total_rul) * noise_multiplier
                )

    flow_rate = (
                    base_flow
                    - flow_loss_rate * t * near_term_factor * critical_boost
                    + rng.normal(0.0, flow_noise, size=total_rul) * noise_multiplier
                )

    motor_current = (
                        base_current
                        + current_growth_rate * t * near_term_factor * critical_boost
                        + rng.normal(0.0, current_noise_base, size=total_rul) * (1.0 + 0.5 * degradation_factor)
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
def generate_dataset(n_cycles: int, min_rul: int, max_rul: int, output_path: str, sample_step: int = 1):
    """Generate dataset by streaming each cycle to CSV to avoid high memory usage.

    Args:
        n_cycles: number of cycles to generate
        min_rul, max_rul: range for total RUL per cycle
        output_path: path to write CSV
        sample_step: keep 1 out of every `sample_step` samples (downsampling). Default 1 (no downsampling).
    """

    # Prepare output file (remove if existing to avoid accidental appends)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    first_write = True

    # Incremental statistics to report at the end
    total_samples = 0
    total_cycles = 0
    min_current_rul = None
    max_current_rul = None

    near_term_count = 0
    critical_count = 0
    very_critical_count = 0

    for cycle_id in range(1, n_cycles + 1):
        total_rul = int(RNG.integers(min_rul, max_rul + 1))
        df_cycle = generate_cycle(cycle_id, total_rul, RNG)

        # Downsample if requested (useful to reduce rows for very long cycles)
        if sample_step > 1:
            df_cycle = df_cycle.iloc[::sample_step].reset_index(drop=True)

        # Update incremental stats
        total_cycles += 1
        samples = len(df_cycle)
        total_samples += samples

        if min_current_rul is None:
            min_current_rul = df_cycle["current_rul"].min()
            max_current_rul = df_cycle["current_rul"].max()
        else:
            min_current_rul = min(min_current_rul, df_cycle["current_rul"].min())
            max_current_rul = max(max_current_rul, df_cycle["current_rul"].max())

        near_term_count += int((df_cycle["current_rul"] <= STAGE0_NEAR_TERM_THRESHOLD).sum())
        critical_count += int((df_cycle["current_rul"] <= CRITICAL_RUL_THRESHOLD).sum())
        very_critical_count += int((df_cycle["current_rul"] <= 300).sum())

        # Append cycle to CSV (streaming)
        df_cycle.to_csv(output_path, mode="a", header=first_write, index=False)
        first_write = False

    # Final reporting
    print(f"\nSaved dataset to {output_path}")
    print(f"Total cycles: {total_cycles}")
    print(f"Total samples: {total_samples}")

    if min_current_rul is not None and max_current_rul is not None:
        print(f"RUL range: {min_current_rul} - {max_current_rul}")

    # Regime distribution analysis (from counts)
    if total_samples:
        near_term_ratio = near_term_count / total_samples * 100
        critical_ratio = critical_count / total_samples * 100
        very_critical_ratio = very_critical_count / total_samples * 100
    else:
        near_term_ratio = critical_ratio = very_critical_ratio = 0.0

    print(f"\nRegime Distribution:")
    print(f"  NEAR_TERM (RUL <= {STAGE0_NEAR_TERM_THRESHOLD}): {near_term_ratio:.2f}%")
    print(f"  CRITICAL (RUL <= {CRITICAL_RUL_THRESHOLD}): {critical_ratio:.2f}%")
    print(f"  Very Critical (RUL <= 300): {very_critical_ratio:.2f}%")



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