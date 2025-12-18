import numpy as np
import pandas as pd
import os

SEED        = 42
MIN_RUL     = 500
MAX_RUL     = 800
N_CYCLES    = 150
RNG = np.random.default_rng(seed=SEED)


def generate_cycle(cycle_id: int, total_rul: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a single pump cycle until failure.
    
    The model follows the project's technical plan:
    - vibration:    exponential-like (quadratic) increase as RUL decreases
    - temp_motor:   linear increase with time
    - pressure:     slight linear decrease with increasing noise
    - vib_motor:    coupled to pump vibration via a coupling factor
    """
    
    t = np.arange(total_rul, dtype=float)   # minutes since cycle start
    
    # make current_rul reach 0 at the fnal timestamp
    current_rul = total_rul - t - 1 
    fraction = t / total_rul                # 0 -> 1 as we approach failure

    # Per-cycle randomized base parameters (small variations make cycles realistic)
    base_vib = rng.uniform(0.02, 0.3)
    failure_vib = rng.uniform(2.0, 6.0)
    vib_noise_scale = rng.uniform(0.01, 0.1)

    base_temp = rng.uniform(40.0, 60.0)             # degrees Celsius
    k_temp = rng.uniform(0.005, 0.02)               # linear temp growth per minute
    temp_noise_scale = rng.uniform(0.05, 0.3)

    base_pressure = rng.uniform(7.6, 8.4)           # bar
    k_pressure = rng.uniform(0.0005, 0.003)         # small pressure decline per minute
    pressure_noise_scale = rng.uniform(0.01, 0.08)

    coupling_factor = rng.uniform(0.4, 0.9)
    vib_motor_noise = rng.uniform(0.005, 0.05)

    # Enhanced degradation in critical region (last 100 minutes)
    critical_mask = current_rul <= 100
    critical_boost = np.where(
                                    critical_mask,
                                    1.0 + (100 - current_rul) / 100 * 0.5,   # Up to 1.5x boost
                                    1.0
                                )

    # Sensor signal generation
    vibration = (
                    base_vib
                    + (failure_vib - base_vib) * (fraction ** 2) * critical_boost
                    + rng.normal(loc=0.0, scale=vib_noise_scale * (1.0 + fraction), size=total_rul)
                )

    temp_motor = (
                    base_temp + k_temp * t * critical_boost 
                    + rng.normal(loc=0.0, scale=temp_noise_scale, size=total_rul)
                )

    pressure = (
                base_pressure - k_pressure * t * critical_boost
                + rng.normal(loc=0.0, scale=pressure_noise_scale * (1.0 + 0.5 * fraction), size=total_rul)
            )

    vib_motor = vibration * coupling_factor + rng.normal(loc=0.0, scale=vib_motor_noise * (1.0 + fraction), size=total_rul)

    df = pd.DataFrame(
                        {
                            "cycle_id": cycle_id,
                            "time_min": t.astype(int),
                            "total_rul": int(total_rul),
                            "current_rul": current_rul.astype(int),
                            "vibration": vibration.astype(float),
                            "temp_motor": temp_motor.astype(float),
                            "pressure": pressure.astype(float),
                            "vib_motor": vib_motor.astype(float),
                        }
                    )
    return df

dfs = []
for cycle_id in range(1, N_CYCLES + 1):
    total_rul = int(RNG.integers(MIN_RUL, MAX_RUL + 1))
    df_cycle = generate_cycle(cycle_id=cycle_id, total_rul=total_rul, rng=RNG)
    dfs.append(df_cycle)

df_all = pd.concat(dfs, ignore_index=True)

out_path = "synthetic_pump_data.csv"
out_dir = os.path.dirname(out_path) or "."
os.makedirs(out_dir, exist_ok=True)

df_all.to_csv(out_path, index=False)
print(f"\nSaved synthetic dataset to {out_path} ({len(df_all)} rows)")

def print_statistics(df):
    print(f"\nDataset Statistics:")
    print('-' * 30)
    print(f"Total cycles:            {df['cycle_id'].nunique()}")
    print(f"Total samples:           {len(df)}")
    print(f"Average cycle length:    {df.groupby('cycle_id').size().mean():.1f}")
    
    # RUL distribution
    rul_stats = df['current_rul'].describe()
    print(f"\nRUL Statistics:")
    print('-' * 30)
    print(f"Min:    {rul_stats['min']}")
    print(f"Max:    {rul_stats['max']}")
    print(f"Mean:   {rul_stats['mean']:.1f}")
    print(f"Std:    {rul_stats['std']:.1f}")
    
    # Critical region analysis
    critical = (df['current_rul'] <= 20).mean() * 100
    print('\n' + ('-' * 30))
    print(f"Critical region (RUL ≤ 20): {critical:.1f}%")
    print('-' * 30)

print_statistics(df_all)
