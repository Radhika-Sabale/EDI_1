"""
Model/01_attention_fusion_model.py

Main model: LSTM encoder + temporal attention + classifier.
This is the model the whole project is building toward — everything in
Data/ (scripts 01-11) exists to feed this one.

Folder layout assumed:

    Illegal Mining Detection/
      Data/
        train_test_split.npz
        09_metrics_pipeline.py
        10_baseline_classical.py
        11_baseline_lstm.py
      Model/
        01_attention_fusion_model.py   <- this file

Run from inside `Model/` (paths below are relative to that, going up into
Data/ for both the split and the shared metrics evaluator — adjust the
DATA_DIR constant if your actual layout differs).
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

# Paths are resolved relative to this file, so it works whether you run it
# from Model/, from the project root, or via the VS Code run button.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data")


def _load_metrics_pipeline():
    path = os.path.join(DATA_DIR, "09_metrics_pipeline.py")
    spec = importlib.util.spec_from_file_location("metrics_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics_pipeline = _load_metrics_pipeline()
evaluate_predictions = metrics_pipeline.evaluate_predictions

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Temporal attention layer
#
# Fusion here is "early fusion": each month's feature vector already carries
# both NDVI and VH-derived values side by side (from Data/07), so the LSTM
# reads one fused signal per month rather than two separate streams. What
# this attention layer adds is the *interpretability* piece from the
# proposal — a weight per month, summing to 1, showing which months the
# classifier actually leaned on.
# ---------------------------------------------------------------------------
class TemporalAttention(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(units, activation="tanh")
        self.score_out = layers.Dense(1)

    def call(self, encoder_outputs):
        # encoder_outputs: (batch, timesteps, hidden)
        scores = self.score_out(self.score_dense(encoder_outputs))  # (batch, T, 1)
        weights = tf.nn.softmax(scores, axis=1)                     # (batch, T, 1)
        context = tf.reduce_sum(weights * encoder_outputs, axis=1)  # (batch, hidden)
        return context, tf.squeeze(weights, axis=-1)                # weights: (batch, T)


def build_model(n_timesteps, n_features, lstm_units=32, attn_units=32):
    inputs = keras.Input(shape=(n_timesteps, n_features), name="monthly_features")
    encoded = layers.LSTM(lstm_units, return_sequences=True, name="encoder")(inputs)
    context, attn_weights = TemporalAttention(attn_units, name="temporal_attention")(encoded)
    x = layers.Dropout(0.3)(context)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="verdict")(x)

    train_model = keras.Model(inputs=inputs, outputs=output, name="attention_fusion_model")
    # Second model sharing the same weights, for pulling out attention weights
    # at inference time without touching how training works.
    attention_model = keras.Model(inputs=inputs, outputs=[output, attn_weights])

    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return train_model, attention_model



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

def load_split(path=None):
    path = path or os.path.join(DATA_DIR, "train_test_split.npz")
    data = np.load(path, allow_pickle=True)
    print(f"Keys found in {path}: {list(data.files)}")
    return data


def main():
    data = load_split()

    # --- Adjust these to match your actual saved key names ---
    n_features_total = len(data["feature_names"])
    X_train = to_sequences(data["X_train"], n_features_total).astype("float32")
    X_test = to_sequences(data["X_test"], n_features_total).astype("float32")
    y_train = data["y_train"].astype("float32")
    y_test = data["y_test"].astype("float32")
    countries_test = load_countries(
        data["ids_test"], csv_path=os.path.join(DATA_DIR, "input_locations.csv")
    )

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    train_model, attention_model = build_model(n_timesteps, n_features)
    train_model.summary()

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=12, restore_best_weights=True
    )

    train_model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=150,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    proba, attn_weights_test = attention_model.predict(X_test)
    proba = proba.ravel()
    pred = (proba >= 0.5).astype(int)  # only used for the saved attention file

    print("\n=== Attention Fusion Model ===")
    results = evaluate_predictions(
        y_true=y_test.astype(int),
        y_pred_proba=proba,
        model_name="Attention Fusion Model",
        countries=countries_test,
    )

    # Save attention weights for every test location so Person 4's demo (or a
    # paper figure) can plot "which months drove this verdict" without
    # retraining. Shape: (n_test, n_timesteps).
    np.savez(
        os.path.join(SCRIPT_DIR, "attention_weights_test.npz"),
        attention_weights=attn_weights_test,
        y_true=y_test,
        y_pred=pred,
        proba=proba,
    )
    print(f"\nSaved attention weights for {attn_weights_test.shape[0]} test locations "
          f"to attention_weights_test.npz")

    return train_model, attention_model, results


if __name__ == "__main__":
    main()