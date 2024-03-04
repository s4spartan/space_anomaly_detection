"""
rul_predictor.py
India Space Academy — AI & ML in Space Exploration
Student: Nirav Singh Dabhi | Roll: 13101980

LSTM-based Remaining Useful Life (RUL) predictor for spacecraft subsystems.
Trained on multi-variate degradation telemetry (analogous to NASA CMAPSS FD003).

Mission context: RUL prediction enables mission planners to schedule
maintenance or redundancy switches before a subsystem fails — particularly
critical when the next communication window is 14 minutes away (Mars scenario).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

SENSOR_COLS = [
    "sensor_temp", "sensor_pressure", "sensor_rpm",
    "sensor_vibration", "sensor_current",
    "op_setting_1", "op_setting_2"
]


def create_sequences(X: np.ndarray, y: np.ndarray,
                     seq_len: int = 50) -> tuple:
    """
    Slide a window of length seq_len over the time series.
    Returns (X_seq, y_seq) where X_seq[i] is the 50-step window
    and y_seq[i] is the RUL at the end of that window.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


class LSTMRULPredictor:
    """
    Stacked LSTM network for RUL prediction.

    Architecture:
      LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)

    Sequence length of 50 cycles chosen after ablation study showing that
    shorter windows miss slow degradation trends and longer windows add
    computational cost without accuracy gain.

    RUL is clipped at a maximum of 130 cycles to focus prediction accuracy
    on the critical near-failure region (practise from NASA CMAPSS literature).
    """

    MAX_RUL = 130

    def __init__(self, seq_len: int = 50, epochs: int = 60,
                 batch_size: int = 128, lr: float = 1e-3):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required for LSTM predictor.")
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.scaler     = MinMaxScaler()
        self.model      = None
        self.history    = None

    def _build_model(self, n_features: int):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_len, n_features)),
            keras.layers.LSTM(64, return_sequences=True, name="lstm_1"),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False, name="lstm_2"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="linear", name="rul_output")
        ], name="LSTM_RUL_Predictor")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="mse",
            metrics=["mae"]
        )
        return model

    def _prepare(self, df: pd.DataFrame) -> tuple:
        """Prepare sequences from degradation dataframe."""
        all_X, all_y = [], []
        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("cycle")
            rul   = np.clip(group["rul"].values, 0, self.MAX_RUL)
            feats = self.scaler.transform(group[SENSOR_COLS].values)
            if len(feats) > self.seq_len:
                X_s, y_s = create_sequences(feats, rul, self.seq_len)
                all_X.append(X_s)
                all_y.append(y_s)
        return np.vstack(all_X), np.hstack(all_y)

    def fit(self, df_train: pd.DataFrame,
            df_val: pd.DataFrame = None) -> "LSTMRULPredictor":
        self.scaler.fit(df_train[SENSOR_COLS].values)
        X_train, y_train = self._prepare(df_train)

        val_data = None
        if df_val is not None:
            X_val, y_val = self._prepare(df_val)
            val_data = (X_val, y_val)

        self.model = self._build_model(len(SENSOR_COLS))
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True, monitor="val_loss"
                if val_data else "loss"),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=4, min_lr=1e-5)
        ]
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0
        )
        return self

    def predict_unit(self, df_unit: pd.DataFrame) -> np.ndarray:
        """Predict RUL for a single spacecraft subsystem unit."""
        df_unit = df_unit.sort_values("cycle")
        feats   = self.scaler.transform(df_unit[SENSOR_COLS].values)
        X_s, _  = create_sequences(feats,
                                    np.zeros(len(feats)), self.seq_len)
        if len(X_s) == 0:
            return np.array([])
        preds = self.model.predict(X_s, verbose=0).flatten()
        return np.clip(preds, 0, self.MAX_RUL)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        all_true, all_pred = [], []
        unit_results = []
        for unit_id, group in df_test.groupby("unit_id"):
            preds = self.predict_unit(group)
            if len(preds) == 0:
                continue
            true_rul = np.clip(
                group.sort_values("cycle")["rul"].values[self.seq_len:],
                0, self.MAX_RUL
            )
            n = min(len(preds), len(true_rul))
            all_pred.extend(preds[:n])
            all_true.extend(true_rul[:n])
            unit_results.append({
                "unit_id": unit_id,
                "mae":     round(mean_absolute_error(true_rul[:n], preds[:n]), 2),
                "rmse":    round(np.sqrt(mean_squared_error(true_rul[:n], preds[:n])), 2)
            })

        mae  = mean_absolute_error(all_true, all_pred)
        rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        return {
            "model": "LSTM RUL Predictor",
            "overall_mae":  round(mae, 2),
            "overall_rmse": round(rmse, 2),
            "per_unit":     unit_results
        }

    def save(self, path: str):
        self.model.save(path.replace(".pkl", ".h5"))
        joblib.dump({"scaler": self.scaler, "seq_len": self.seq_len,
                     "max_rul": self.MAX_RUL}, path)

    @classmethod
    def load(cls, path: str) -> "LSTMRULPredictor":
        obj = cls()
        meta = joblib.load(path)
        obj.scaler  = meta["scaler"]
        obj.seq_len = meta["seq_len"]
        obj.model   = keras.models.load_model(path.replace(".pkl", ".h5"))
        return obj
