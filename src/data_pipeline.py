"""
data_pipeline.py
India Space Academy — AI & ML in Space Exploration
Student: Nirav Singh Dabhi | Roll: 13101980

End-to-end preprocessing pipeline for spacecraft telemetry.
Handles: missing values, noise, temporal splits, feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

TELEMETRY_FEATURES = [
    "voltage_bus", "solar_current", "thermal_thruster",
    "gyro_drift", "signal_strength", "battery_soc", "tank_pressure"
]
SOLAR_FEATURES = [
    "sw_speed_kmps", "proton_density", "bz_nt",
    "proton_flux", "kp_index"
]


class TelemetryPipeline:
    """
    Preprocessing pipeline for spacecraft subsystem telemetry.

    Steps:
    1. Fill sensor gaps using forward-fill then linear interpolation
       (rationale: telemetry gaps are dropouts, not true nulls — last-known
       value is the best estimate for short gaps < 5 min)
    2. Remove duplicate timestamps
    3. Compute rolling statistics (window=10) as derived features
    4. Apply RobustScaler (resistant to the outlier anomaly values we want
       to detect — StandardScaler would shrink anomaly magnitude)
    5. Temporal-aware train/val/test split: no shuffling to prevent
       future data leaking into training
    """

    def __init__(self, window: int = 10, test_ratio: float = 0.15,
                 val_ratio: float = 0.10):
        self.window     = window
        self.test_ratio = test_ratio
        self.val_ratio  = val_ratio
        self.scaler     = RobustScaler()
        self._fitted    = False

    def _fill_gaps(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        df[cols] = (df[cols]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear", limit=5))
        return df

    def _add_rolling_features(self, df: pd.DataFrame,
                               cols: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            df[f"{col}_roll_mean"] = (df[col]
                                       .rolling(self.window, min_periods=1)
                                       .mean())
            df[f"{col}_roll_std"]  = (df[col]
                                       .rolling(self.window, min_periods=1)
                                       .std()
                                       .fillna(0))
        return df

    def fit_transform(self, df: pd.DataFrame,
                      feature_cols: list = None) -> dict:
        """
        Full pipeline on training data.
        Returns dict with X_train, X_val, X_test, y_train, y_val, y_test.
        """
        if feature_cols is None:
            feature_cols = TELEMETRY_FEATURES

        df = self._fill_gaps(df, feature_cols)
        df = self._add_rolling_features(df, feature_cols)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        all_feat_cols = (feature_cols
                         + [f"{c}_roll_mean" for c in feature_cols]
                         + [f"{c}_roll_std"  for c in feature_cols])

        X = df[all_feat_cols].values
        y = df["anomaly_label"].values if "anomaly_label" in df.columns \
            else np.zeros(len(X))

        # Temporal split — no shuffle
        n = len(X)
        n_test = int(n * self.test_ratio)
        n_val  = int(n * self.val_ratio)
        n_train= n - n_test - n_val

        X_train, y_train = X[:n_train],          y[:n_train]
        X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        X_test_s  = self.scaler.transform(X_test)
        self._fitted = True

        return {
            "X_train": X_train_s, "y_train": y_train,
            "X_val":   X_val_s,   "y_val":   y_val,
            "X_test":  X_test_s,  "y_test":  y_test,
            "feature_names": all_feat_cols,
            "n_train": n_train, "n_val": n_val, "n_test": n_test
        }

    def transform(self, df: pd.DataFrame,
                  feature_cols: list = None) -> np.ndarray:
        """Transform new/inference data using fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        if feature_cols is None:
            feature_cols = TELEMETRY_FEATURES
        df = self._fill_gaps(df, feature_cols)
        df = self._add_rolling_features(df, feature_cols)
        all_feat_cols = (feature_cols
                         + [f"{c}_roll_mean" for c in feature_cols]
                         + [f"{c}_roll_std"  for c in feature_cols])
        return self.scaler.transform(df[all_feat_cols].values)


class SolarDataPipeline:
    """Preprocessing for solar wind / OMNI data."""

    def __init__(self):
        self.scaler = RobustScaler()

    def process(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        for col in SOLAR_FEATURES:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Log-transform proton flux (spans several decades)
        if "proton_flux" in df.columns:
            df["proton_flux_log"] = np.log10(df["proton_flux"].clip(1e-3))

        feat_cols = [c for c in SOLAR_FEATURES if c in df.columns]
        if "proton_flux_log" in df.columns:
            feat_cols.append("proton_flux_log")
            feat_cols = [c for c in feat_cols if c != "proton_flux"]

        X = self.scaler.fit_transform(df[feat_cols].values)
        y = df["cme_event"].values if "cme_event" in df.columns \
            else np.zeros(len(X))

        return {"X": X, "y": y, "feature_names": feat_cols}
