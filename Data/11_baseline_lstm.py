"""
11_baseline_lstm.py

Person 2 — Baselines & Co-Modeling
Plain LSTM baseline (no attention) — the first sequence-aware model in the
project, giving a comparison point for the later encoder+attention model.

Uses the same train_test_split.npz as 10_baseline_classical.py and the same
evaluate_predictions() from 09_metrics_pipeline.py, so all three baselines
(LogReg, RF, LSTM) are directly comparable.

Run from inside the `Data/` folder (uses bare filenames, matching scripts 01-09).
"""

import importlib.util
import os
import numpy as np
import os

# Silence TensorFlow's start-up chatter (must be set BEFORE importing tensorflow).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # hide info/warning logs from the C++ backend
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # hide the oneDNN notice
import tensorflow as tf
tf.get_logger().setLevel("ERROR")   # hide retracing / deprecation warnings
from tensorflow import keras
from tensorflow.keras import layers


# Always run relative to this file's folder (Data/), wherever it's launched from
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _load_metrics_pipeline():
    spec = importlib.util.spec_from_file_location(
        "metrics_pipeline", "09_metrics_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics_pipeline = _load_metrics_pipeline()
evaluate_predictions = metrics_pipeline.evaluate_predictions

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)



def load_countries(ids, csv_path="input_locations.csv"):
    """Look up each location's country via input_locations.csv. Returns None if it can't."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        country_col = next((cols[c] for c in cols if "country" in c), None)
        id_col = next((cols[c] for c in ("location_id", "id", "loc_id", "site_id", "name") if c in cols), None)
        if country_col is None or id_col is None:
            print(f"[countries] Couldn't find id/country columns in {csv_path}. Columns are: {list(df.columns)}")
            return None
        lookup = dict(zip(df[id_col].astype(str), df[country_col]))
        out = np.array([lookup.get(str(i), "unknown") for i in ids])
        if (out == "unknown").all():
            print("[countries] No ids matched input_locations.csv - skipping per-country breakdown.")
            return None
        return out
    except Exception as e:
        print(f"[countries] Skipping per-country breakdown: {e}")
        return None


def to_sequences(X, n_features):
    """Return X as (n_locations, n_months, n_features), whether it was saved flat or not."""
    return X.reshape(X.shape[0], -1, n_features) if X.ndim == 2 else X

def load_split(path="train_test_split.npz"):
    data = np.load(path, allow_pickle=True)
    print(f"Keys found in {path}: {list(data.files)}")
    return data


def build_model(n_timesteps, n_features):
    """
    Deliberately plain: a single LSTM layer + dropout + dense output.
    No attention, no bidirectionality, no stacking — that's the point of
    this baseline. The encoder+attention model is where those go.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(n_timesteps, n_features)),
            layers.Masking(mask_value=0.0),  # harmless if there's no padding
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    data = load_split()

    # --- Adjust these to match your actual saved key names ---
    n_features_total = len(data["feature_names"])
    X_train = to_sequences(data["X_train"], n_features_total).astype("float32")
    X_test = to_sequences(data["X_test"], n_features_total).astype("float32")
    y_train = data["y_train"].astype("float32")
    y_test = data["y_test"].astype("float32")
    countries_test = load_countries(data["ids_test"])

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = build_model(n_timesteps, n_features)
    model.summary()

    # Class weights since the split is 105/108 and 27/27 — close to balanced,
    # but this keeps it robust if you rerun with a different split later.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,  # carved out of train, not from the held-out test set
        epochs=100,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    proba = model.predict(X_test).ravel()

    print("\n=== Plain LSTM ===")
    results = evaluate_predictions(
        y_true=y_test.astype(int),
        y_pred_proba=proba,
        model_name="Plain LSTM",
        countries=countries_test,
    )

    return model, history, results


if __name__ == "__main__":
    main()