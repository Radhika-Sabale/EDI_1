"""
Model/04_tune_attention_model.py

Tunes the temporal attention model using stratified 5-fold cross-validation on
the 213 ORIGINAL TRAINING locations only (zero test-set leakage).

Search Space:
  - LSTM units: {16, 32, 64}
  - Dropout: {0.2, 0.3, 0.5}
  - Learning rate: {3e-4, 1e-3, 3e-3}
  - Transformer-encoder variants (positional encoding, 1-2 layers, d_model=32,
    2 heads, temporal attention pooling + classifier head)

Evaluation Workflow:
  1. Stratified 5-fold CV on original 213 training locations (stratified by country + label).
  2. Ranks configurations by out-of-fold validation ROC-AUC and F1.
  3. Evaluates best configuration on the held-out 54-location test set (Data/train_test_split.npz).
  4. Runs the winning configuration through the 10-seed multi-seed protocol from Model/02.
     (Note: Because tuning was performed on the 213 training locations from split 0,
      subsequent multi-seed splits may slightly overlap with the tuning set, which is noted in output).

How to run:
  python Model/04_tune_attention_model.py
  (or via the VS Code run button from anywhere)
"""

import contextlib
import importlib.util
import io
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Silence TensorFlow startup chatter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data")


# --------------------------------------------------------------------------
# Load helpers from Model/02_multiseed_evaluation.py
# --------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m02 = _load_module("multiseed_eval", os.path.join(SCRIPT_DIR, "02_multiseed_evaluation.py"))
evaluate_predictions = m02.evaluate_predictions
load_countries = m02.load_countries
load_pooled_data = m02.load_pooled_data
make_split = m02.make_split
rescale = m02.rescale
BASE_SEED = m02.BASE_SEED
N_REPEATS = int(os.environ.get("MULTISEED_REPEATS", 10))


# --------------------------------------------------------------------------
# Temporal Attention Layer
# --------------------------------------------------------------------------
class TemporalAttention(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(units, activation="tanh")
        self.score_out = layers.Dense(1)

    def call(self, encoder_outputs):
        scores = self.score_out(self.score_dense(encoder_outputs))
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(weights * encoder_outputs, axis=1)
        return context, tf.squeeze(weights, axis=-1)


# --------------------------------------------------------------------------
# Model Builders
# --------------------------------------------------------------------------
def build_lstm_attention(t=12, f=6, units=32, dropout=0.3, lr=1e-3):
    inputs = keras.Input(shape=(t, f), name="monthly_inputs")
    encoded = layers.LSTM(units, return_sequences=True, name="lstm_encoder")(inputs)
    context, attn_w = TemporalAttention(units, name="temporal_attention")(encoded)
    x = layers.Dropout(dropout)(context)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="verdict")(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_transformer_attention(t=12, f=6, d_model=32, num_heads=2, num_layers=1, dropout=0.2, lr=1e-3):
    inputs = keras.Input(shape=(t, f), name="monthly_inputs")
    x = layers.Dense(d_model, name="projection")(inputs)
    pos_emb = layers.Embedding(input_dim=t, output_dim=d_model, name="pos_embedding")(tf.range(t))
    x = x + pos_emb
    for i in range(num_layers):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=f"mha_{i}")(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dropout)(attn))
        ffn = layers.Dense(d_model * 2, activation="relu")(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    context, attn_w = TemporalAttention(d_model, name="temporal_attention")(x)
    h = layers.Dropout(dropout)(context)
    h = layers.Dense(16, activation="relu")(h)
    output = layers.Dense(1, activation="sigmoid", name="verdict")(h)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# --------------------------------------------------------------------------
# Candidate Search Space
# --------------------------------------------------------------------------
def get_candidate_configs():
    """Returns candidate hyperparameter configurations spanning the search space."""
    return [
        # Baseline Reference
        {"id": "LSTM_u32_d0.3_lr1e-3 (Default)", "arch": "lstm", "units": 32, "dropout": 0.3, "lr": 1e-3},
        # Varying LSTM units {16, 32, 64}
        {"id": "LSTM_u16_d0.3_lr1e-3", "arch": "lstm", "units": 16, "dropout": 0.3, "lr": 1e-3},
        {"id": "LSTM_u64_d0.3_lr1e-3", "arch": "lstm", "units": 64, "dropout": 0.3, "lr": 1e-3},
        # Varying Dropout {0.2, 0.5}
        {"id": "LSTM_u32_d0.2_lr1e-3", "arch": "lstm", "units": 32, "dropout": 0.2, "lr": 1e-3},
        {"id": "LSTM_u32_d0.5_lr1e-3", "arch": "lstm", "units": 32, "dropout": 0.5, "lr": 1e-3},
        # Varying Learning Rate {3e-4, 3e-3}
        {"id": "LSTM_u32_d0.3_lr3e-4", "arch": "lstm", "units": 32, "dropout": 0.3, "lr": 3e-4},
        {"id": "LSTM_u32_d0.3_lr3e-3", "arch": "lstm", "units": 32, "dropout": 0.3, "lr": 3e-3},
        # Promising Cross-Combinations
        {"id": "LSTM_u16_d0.2_lr1e-3", "arch": "lstm", "units": 16, "dropout": 0.2, "lr": 1e-3},
        {"id": "LSTM_u64_d0.5_lr1e-3", "arch": "lstm", "units": 64, "dropout": 0.5, "lr": 1e-3},
        {"id": "LSTM_u64_d0.3_lr3e-4", "arch": "lstm", "units": 64, "dropout": 0.3, "lr": 3e-4},
        # Transformer-Encoder Variants
        {"id": "Transformer_L1_d32_h2_drop0.2_lr1e-3", "arch": "transformer", "layers": 1, "d_model": 32, "heads": 2, "dropout": 0.2, "lr": 1e-3},
        {"id": "Transformer_L2_d32_h2_drop0.3_lr1e-3", "arch": "transformer", "layers": 2, "d_model": 32, "heads": 2, "dropout": 0.3, "lr": 1e-3},
        {"id": "Transformer_L1_d32_h2_drop0.3_lr3e-4", "arch": "transformer", "layers": 1, "d_model": 32, "heads": 2, "dropout": 0.3, "lr": 3e-4},
    ]


def build_candidate_model(cfg, t=12, f=6):
    if cfg["arch"] == "lstm":
        return build_lstm_attention(
            t=t, f=f, units=cfg["units"], dropout=cfg["dropout"], lr=cfg["lr"]
        )
    else:
        return build_transformer_attention(
            t=t, f=f, d_model=cfg["d_model"], num_heads=cfg["heads"],
            num_layers=cfg["layers"], dropout=cfg["dropout"], lr=cfg["lr"]
        )


# --------------------------------------------------------------------------
# Main Tuning Routine
# --------------------------------------------------------------------------
def main():
    split_path = os.path.join(DATA_DIR, "train_test_split.npz")
    data = np.load(split_path, allow_pickle=True)
    X_train = data["X_train"].astype("float32")  # (213, 12, 6)
    y_train = data["y_train"].astype(int)
    ids_train = data["ids_train"]
    X_test = data["X_test"].astype("float32")    # (54, 12, 6)
    y_test = data["y_test"].astype(int)
    ids_test = data["ids_test"]

    countries_train = load_countries(ids_train, os.path.join(DATA_DIR, "input_locations.csv"))
    countries_test = load_countries(ids_test, os.path.join(DATA_DIR, "input_locations.csv"))

    strata_train = (
        np.array([f"{c}_{l}" for c, l in zip(countries_train, y_train)])
        if countries_train is not None else y_train
    )

    configs = get_candidate_configs()
    n_configs = len(configs)
    n_splits = 5

    print("=" * 80)
    print("TASK 2: Attention Model Hyperparameter Tuning & Architecture Search")
    print("=" * 80)
    print(f"Data: {len(y_train)} original training locations (no test leakage).")
    print(f"Stratified {n_splits}-fold CV across {n_configs} candidate architectures/hyperparameters.")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_records = []
    total_start = time.time()

    for idx, cfg in enumerate(configs, 1):
        cfg_start = time.time()
        oof_proba = np.zeros(len(y_train), dtype="float32")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, strata_train), 1):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            keras.backend.clear_session()
            keras.utils.set_random_seed(42 + fold)

            model = build_candidate_model(cfg, t=X_train.shape[1], f=X_train.shape[2])
            n_pos = y_tr.sum()
            n_neg = len(y_tr) - n_pos
            class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=7, restore_best_weights=True
            )

            model.fit(
                X_tr,
                y_tr.astype("float32"),
                validation_data=(X_val, y_val.astype("float32")),
                epochs=45,
                batch_size=32,
                class_weight=class_weight,
                callbacks=[early_stop],
                verbose=0,
            )

            oof_proba[val_idx] = model.predict(X_val, verbose=0).ravel()

        with contextlib.redirect_stdout(io.StringIO()):
            res = evaluate_predictions(
                y_true=y_train,
                y_pred_proba=oof_proba,
                model_name=cfg["id"],
                countries=countries_train,
            )

        elapsed = time.time() - cfg_start
        cv_records.append({
            "config_id": cfg["id"],
            "arch": cfg["arch"],
            "val_roc_auc": res["roc_auc"],
            "val_f1": res["f1"],
            "val_precision": res["precision"],
            "val_recall": res["recall"],
            "time_sec": elapsed,
        })

        print(f"[{idx:2d}/{n_configs}] {cfg['id']:<38} -> "
              f"Val ROC-AUC: {res['roc_auc']:.3f} | Val F1: {res['f1']:.3f} ({elapsed:.1f}s)")

    cv_df = pd.DataFrame(cv_records).sort_values(by="val_roc_auc", ascending=False).reset_index(drop=True)
    cv_csv_path = os.path.join(SCRIPT_DIR, "tuning_cv_results.csv")
    cv_df.to_csv(cv_csv_path, index=False)

    print(f"\nSaved 5-fold CV results to {cv_csv_path}")
    print("\nTop 5 Architectures by 5-Fold Validation ROC-AUC:")
    print(cv_df[["config_id", "arch", "val_roc_auc", "val_f1", "val_precision", "val_recall", "time_sec"]].head().to_string(index=False))

    # Identify winning configuration
    best_cfg_id = cv_df.iloc[0]["config_id"]
    best_cfg = next(c for c in configs if c["id"] == best_cfg_id)
    print(f"\nWinning Configuration: {best_cfg_id}")

    # --------------------------------------------------------------------------
    # Step 3: Single Evaluation on Held-Out Test Set (Original 54 locations)
    # --------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Evaluating Winning Configuration on Held-Out Test Set (54 locations)")
    print(f"{'=' * 80}")

    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    final_model = build_candidate_model(best_cfg, t=X_train.shape[1], f=X_train.shape[2])

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=8, restore_best_weights=True
    )

    final_model.fit(
        X_train,
        y_train.astype("float32"),
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )

    test_proba = final_model.predict(X_test, verbose=0).ravel()
    test_eval = evaluate_predictions(
        y_true=y_test,
        y_pred_proba=test_proba,
        model_name=f"Tuned: {best_cfg_id}",
        countries=countries_test,
    )

    eval_summary_df = pd.DataFrame([{
        "configuration": best_cfg_id,
        "test_precision": test_eval["precision"],
        "test_recall": test_eval["recall"],
        "test_f1": test_eval["f1"],
        "test_roc_auc": test_eval["roc_auc"],
    }])
    eval_csv_path = os.path.join(SCRIPT_DIR, "tuned_model_evaluation.csv")
    eval_summary_df.to_csv(eval_csv_path, index=False)

    # --------------------------------------------------------------------------
    # Step 4: Multi-Seed 10-Repeat Evaluation (Same protocol as Model/02)
    # --------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"Running 10-Seed Repeat Protocol on Winning Config: {best_cfg_id}")
    print(f"{'=' * 80}")
    print("NOTE: This configuration was selected using the 213 training locations.")
    print("In the multi-seed protocol, those locations can appear in random test folds,")
    print("so this multi-seed estimate may be slightly optimistic compared to nested CV.")
    print("=" * 80)

    X_pool, y_pool, countries_pool = load_pooled_data()
    strata_pool = (
        np.array([f"{c}_{l}" for c, l in zip(countries_pool, y_pool)])
        if countries_pool is not None else y_pool
    )

    ms_rows = []
    for r in range(N_REPEATS):
        seed = BASE_SEED + r
        tr_idx, te_idx = make_split(y_pool, strata_pool, seed)
        X_tr, X_te = rescale(X_pool[tr_idx], X_pool[te_idx])
        y_tr, y_te = y_pool[tr_idx], y_pool[te_idx]
        c_te = countries_pool[te_idx] if countries_pool is not None else None

        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        m = build_candidate_model(best_cfg, t=X_tr.shape[1], f=X_tr.shape[2])

        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        cw = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=8, restore_best_weights=True
        )

        m.fit(
            X_tr,
            y_tr.astype("float32"),
            validation_split=0.15,
            epochs=60,
            batch_size=32,
            class_weight=cw,
            callbacks=[stop],
            verbose=0,
        )

        proba = m.predict(X_te, verbose=0).ravel()
        with contextlib.redirect_stdout(io.StringIO()):
            res = evaluate_predictions(
                y_true=y_te, y_pred_proba=proba, model_name=best_cfg_id, countries=c_te
            )

        ms_rows.append({
            "repeat": r,
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"],
        })
        print(f"  Repeat {r + 1}/{N_REPEATS}: ROC-AUC={res['roc_auc']:.3f}, F1={res['f1']:.3f}")

    ms_df = pd.DataFrame(ms_rows)
    ms_summary = pd.DataFrame({
        "metric": ["precision", "recall", "f1", "roc_auc"],
        "mean": [ms_df[m].mean() for m in ["precision", "recall", "f1", "roc_auc"]],
        "std": [ms_df[m].std() for m in ["precision", "recall", "f1", "roc_auc"]],
    })
    ms_summary["mean_pm_std"] = ms_summary.apply(lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f}", axis=1)

    print(f"\nTuned Model 10-Seed Performance Summary ({best_cfg_id}):")
    print(ms_summary[["metric", "mean_pm_std"]].to_string(index=False))

    ms_summary_path = os.path.join(SCRIPT_DIR, "tuned_multiseed_summary.csv")
    ms_summary.to_csv(ms_summary_path, index=False)
    print(f"\nSaved tuned multi-seed summary to {ms_summary_path}")
    print(f"Total time elapsed: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
