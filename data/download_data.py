"""
Data Download Script
India Space Academy — AI & ML in Space Exploration
Student: Nirav Singh Dabhi | Roll: 13101980

Downloads and prepares all required datasets:
1. NASA SMAP/MSL labelled anomaly dataset
2. Simulated spacecraft subsystem telemetry (generated)
3. Solar storm / OMNI space weather data
"""

import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """Stream-download a file with progress bar."""
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=desc, total=total, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [WARN] Could not download {desc}: {e}")
        print(f"  Manual download from: {url}")
        return False


def generate_spacecraft_telemetry(n_nominal: int = 8000,
                                   n_anomaly: int = 800,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Generate simulated spacecraft subsystem telemetry.

    Subsystems modelled:
      - Power bus voltage (V)
      - Solar panel current (A)
      - Thermal sensor — thruster cluster (degC)
      - Attitude control gyro drift (deg/s)
      - Communication signal strength (dBm)
      - Battery state of charge (%)
      - Propellant tank pressure (bar)

    Anomaly types injected:
      - Point anomalies: sudden spike/drop in single sensor
      - Contextual anomalies: normal value but wrong context (e.g., high
        current during eclipse)
      - Collective anomalies: correlated drift across multiple sensors
    """
    rng = np.random.default_rng(seed)
    n_total = n_nominal + n_anomaly

    # --- Nominal telemetry ---
    t = np.linspace(0, 1, n_total)
    orbit_phase = np.sin(2 * np.pi * t * 16)  # ~16 orbits across dataset

    voltage       = 28.0 + 0.5 * orbit_phase + rng.normal(0, 0.1, n_total)
    solar_current = 4.2  + 0.8 * np.clip(orbit_phase, 0, 1) + rng.normal(0, 0.05, n_total)
    thermal       = 45.0 + 12.0 * np.abs(orbit_phase) + rng.normal(0, 1.5, n_total)
    gyro_drift    = rng.normal(0, 0.02, n_total)
    signal_str    = -72.0 + 8.0 * orbit_phase + rng.normal(0, 1.0, n_total)
    battery_soc   = 85.0 - 5.0 * np.abs(orbit_phase) + rng.normal(0, 0.5, n_total)
    tank_pressure = 210.0 + rng.normal(0, 0.8, n_total)

    labels = np.zeros(n_total, dtype=int)

    # --- Inject anomalies ---
    anomaly_indices = rng.choice(np.arange(n_nominal, n_total), size=n_anomaly, replace=False)

    # Type 1: Power bus voltage spike (HPC degradation analogue)
    spike_idx = anomaly_indices[:n_anomaly // 4]
    voltage[spike_idx] += rng.uniform(3.5, 7.0, len(spike_idx))

    # Type 2: Thermal runaway — thruster overheat
    thermal_idx = anomaly_indices[n_anomaly // 4: n_anomaly // 2]
    thermal[thermal_idx] += rng.uniform(30, 80, len(thermal_idx))

    # Type 3: Gyro drift — attitude control failure precursor
    gyro_idx = anomaly_indices[n_anomaly // 2: 3 * n_anomaly // 4]
    gyro_drift[gyro_idx] += rng.uniform(0.15, 0.4, len(gyro_idx))

    # Type 4: Collective — battery + solar current anomaly (eclipse entry failure)
    collective_idx = anomaly_indices[3 * n_anomaly // 4:]
    battery_soc[collective_idx] -= rng.uniform(15, 30, len(collective_idx))
    solar_current[collective_idx] -= rng.uniform(2.0, 3.5, len(collective_idx))

    labels[anomaly_indices] = 1

    df = pd.DataFrame({
        "timestamp":       pd.date_range("2024-01-01", periods=n_total, freq="1min"),
        "orbit_phase":     orbit_phase,
        "voltage_bus":     voltage,
        "solar_current":   solar_current,
        "thermal_thruster":thermal,
        "gyro_drift":      gyro_drift,
        "signal_strength": signal_str,
        "battery_soc":     battery_soc,
        "tank_pressure":   tank_pressure,
        "anomaly_label":   labels,
    })

    anomaly_type = np.full(n_total, "nominal", dtype=object)
    anomaly_type[spike_idx]      = "power_spike"
    anomaly_type[thermal_idx]    = "thermal_runaway"
    anomaly_type[gyro_idx]       = "gyro_drift"
    anomaly_type[collective_idx] = "eclipse_failure"
    df["anomaly_type"] = anomaly_type

    return df


def generate_solar_storm_data(n_points: int = 5000, seed: int = 7) -> pd.DataFrame:
    """
    Generate simulated solar wind / space weather telemetry
    (representative of OMNI / Aditya-L1 L1 point data).

    Channels:
      - Solar wind speed (km/s)
      - Proton density (n/cc)
      - Magnetic field Bz (nT)  — southward = geomagnetic storm trigger
      - Solar proton flux (pfu)
      - Kp geomagnetic index
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    sw_speed      = 400  + 80  * np.sin(2 * np.pi * t / 27)  + rng.normal(0, 20, n_points)
    proton_density= 6    + 3   * np.cos(2 * np.pi * t / 13)  + rng.exponential(1, n_points)
    bz            = rng.normal(0, 3, n_points)
    proton_flux   = 10   ** rng.uniform(-1, 2, n_points)
    kp_index      = np.clip(rng.exponential(1.5, n_points), 0, 9)

    # Inject CME (Coronal Mass Ejection) events
    cme_starts = [800, 2200, 3900]
    cme_label  = np.zeros(n_points, dtype=int)
    for cs in cme_starts:
        duration = rng.integers(48, 96)
        ce = min(cs + duration, n_points)
        sw_speed[cs:ce]       += rng.uniform(300, 700, ce - cs)
        proton_density[cs:ce] += rng.uniform(20, 60, ce - cs)
        bz[cs:ce]             -= rng.uniform(10, 40, ce - cs)  # southward Bz
        kp_index[cs:ce]       += rng.uniform(4, 8, ce - cs)
        cme_label[cs:ce]       = 1

    df = pd.DataFrame({
        "timestamp":     pd.date_range("2024-01-01", periods=n_points, freq="1H"),
        "sw_speed_kmps": sw_speed,
        "proton_density":proton_density,
        "bz_nt":         bz,
        "proton_flux":   proton_flux,
        "kp_index":      np.clip(kp_index, 0, 9),
        "cme_event":     cme_label,
    })
    return df


def generate_degradation_telemetry(n_units: int = 50,
                                    max_cycles: int = 300,
                                    seed: int = 99) -> pd.DataFrame:
    """
    Generate multivariate spacecraft subsystem degradation data
    for RUL prediction (analogous to NASA CMAPSS FD003 — multi-fault).
    Each 'unit' is a spacecraft subsystem run to failure.
    """
    rng = np.random.default_rng(seed)
    records = []
    for unit in range(1, n_units + 1):
        life = rng.integers(150, max_cycles)
        for cyc in range(1, life + 1):
            deg = cyc / life  # degradation progress 0→1
            records.append({
                "unit_id":         unit,
                "cycle":           cyc,
                "rul":             life - cyc,
                "op_setting_1":    rng.uniform(0, 1),
                "op_setting_2":    rng.uniform(0, 1),
                "sensor_temp":     500 + 120 * deg + rng.normal(0, 5),
                "sensor_pressure": 140 - 30  * deg + rng.normal(0, 2),
                "sensor_rpm":      9000 - 1500 * deg + rng.normal(0, 50),
                "sensor_vibration":0.02 + 0.15 * deg ** 2 + rng.exponential(0.005),
                "sensor_current":  8.5  + 3.5  * deg + rng.normal(0, 0.3),
            })
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("ISA Space Anomaly Detection — Data Preparation")
    print("Student: Nirav Singh Dabhi | Roll: 13101980")
    print("=" * 60)

    # 1. Simulated spacecraft telemetry
    print("\n[1/3] Generating simulated spacecraft telemetry...")
    df_telemetry = generate_spacecraft_telemetry()
    path = os.path.join(RAW_DIR, "spacecraft_telemetry.csv")
    df_telemetry.to_csv(path, index=False)
    print(f"  Saved {len(df_telemetry)} records → {path}")
    print(f"  Anomaly rate: {df_telemetry['anomaly_label'].mean()*100:.1f}%")

    # 2. Solar storm data
    print("\n[2/3] Generating solar wind / storm data...")
    df_solar = generate_solar_storm_data()
    path = os.path.join(RAW_DIR, "solar_storm_data.csv")
    df_solar.to_csv(path, index=False)
    print(f"  Saved {len(df_solar)} records → {path}")
    print(f"  CME events: {df_solar['cme_event'].sum()} hours flagged")

    # 3. Degradation telemetry for RUL
    print("\n[3/3] Generating subsystem degradation data for RUL prediction...")
    df_deg = generate_degradation_telemetry()
    path = os.path.join(RAW_DIR, "degradation_telemetry.csv")
    df_deg.to_csv(path, index=False)
    print(f"  Saved {len(df_deg)} records → {path}")
    print(f"  Units: {df_deg['unit_id'].nunique()} | Avg life: {df_deg.groupby('unit_id')['cycle'].max().mean():.0f} cycles")

    print("\nAll datasets ready. Run notebooks in order: 01 → 08")
    print("=" * 60)


if __name__ == "__main__":
    main()
