"""
Model/05_seed_ensemble.py

Stabilizes the final model predictions by building a seed ensemble:
For each split, trains N=5 independent copies of the best attention configuration
using different initialization seeds, and averages their predicted probabilities.

Architecture:
  Uses the top-performing configuration identified in Model/04:
  Transformer-Encoder (1 layer, d_model=32, 2 heads, dropout=0.2, lr=1e-3)
  coupled with Temporal Attention pooling.

Evaluation Protocol:
  - Runs on the same 10 stratified splits as Model/02_multiseed_evaluation.py.
  - Compares Single Model (seed 0) vs. 5-Seed Ensemble on identical test sets.
  - Reports Mean +/- Std across splits for ROC-AUC, F1, Precision, and Recall.
  - Computes paired differences and win/loss/tie tallies.
  - Saves results to ensemble_results.csv and ensemble_summary.csv.

How to run:
  python Model/05_seed_ensemble.py
  (or via the VS Code run button from anywhere)
"""

import contextlib
import importlib.util
import io
import os
import time

import numpy as np
import pandas as pd

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
load_pooled_data = m02.load_pooled_data
make_split = m02.make_split
rescale = m02.rescale
BASE_SEED = m02.BASE_SEED
N_REPEATS = int(os.environ.get("MULTISEED_REPEATS", 10))
N_ENSEMBLE = 5


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
# Best Architecture Builder (from Model/04 tuning)
# --------------------------------------------------------------------------
def build_best_model(t=12, f=6, d_model=32, num_heads=2, num_layers=1, dropout=0.2, lr=1e-3):
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
    context, _ = TemporalAttention(d_model, name="temporal_attention")(x)
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


def train_single_instance(X_tr, y_tr, X_te, seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_best_model(t=X_tr.shape[1], f=X_tr.shape[2])

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    cw = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=8, restore_best_weights=True
    )

    model.fit(
        X_tr,
        y_tr.astype("float32"),
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        class_weight=cw,
        callbacks=[stop],
        verbose=0,
    )
    return model.predict(X_te, verbose=0).ravel()


# --------------------------------------------------------------------------
# Main Ensemble Evaluation
# --------------------------------------------------------------------------
def main():
    X, y, countries = load_pooled_data()
    strata = (
        np.array([f"{c}_{l}" for c, l in zip(countries, y)])
        if countries is not None else y
    )

    print("=" * 80)
    print("TASK 3: Seed Ensemble Evaluation (N=5 copies per split)")
    print("=" * 80)
    print(f"Data: {len(y)} locations | {N_REPEATS} repeated splits")
    print(f"Architecture: Tuned Transformer + Temporal Attention")
    print(f"Ensemble size per split: N={N_ENSEMBLE} models trained with seeds [seed + 100*i]")
    print("=" * 80)

    rows = []
    start = time.time()

    for r in range(N_REPEATS):
        split_seed = BASE_SEED + r
        tr_idx, te_idx = make_split(y, strata, split_seed)
        X_tr, X_te = rescale(X[tr_idx], X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]
        c_te = countries[te_idx] if countries is not None else None

        # Train N models with different initialization seeds
        member_probas = []
        for m_idx in range(N_ENSEMBLE):
            init_seed = split_seed * 100 + m_idx
            proba_m = train_single_instance(X_tr, y_tr, X_te, seed=init_seed)
            member_probas.append(proba_m)

        member_probas = np.array(member_probas)  # (N_ENSEMBLE, n_test)
        single_proba = member_probas[0]          # First model (individual seed)
        ensemble_proba = member_probas.mean(axis=0)  # Averaged probabilities

        # Evaluate Single Model
        with contextlib.redirect_stdout(io.StringIO()):
            res_single = evaluate_predictions(
                y_true=y_te, y_pred_proba=single_proba, model_name="Single Model", countries=c_te
            )
            res_ens = evaluate_predictions(
                y_true=y_te, y_pred_proba=ensemble_proba, model_name="5-Seed Ensemble", countries=c_te
            )

        rows.append({
            "repeat": r,
            "single_precision": res_single["precision"],
            "single_recall": res_single["recall"],
            "single_f1": res_single["f1"],
            "single_roc_auc": res_single["roc_auc"],
            "ens_precision": res_ens["precision"],
            "ens_recall": res_ens["recall"],
            "ens_f1": res_ens["f1"],
            "ens_roc_auc": res_ens["roc_auc"],
        })

        diff_auc = res_ens["roc_auc"] - res_single["roc_auc"]
        diff_f1 = res_ens["f1"] - res_single["f1"]
        print(f"  Split {r + 1:2d}/{N_REPEATS} -> "
              f"Single: [AUC={res_single['roc_auc']:.3f}, F1={res_single['f1']:.3f}] | "
              f"Ensemble: [AUC={res_ens['roc_auc']:.3f}, F1={res_ens['f1']:.3f}] | "
              f"Diff AUC: {diff_auc:+.3f}, Diff F1: {diff_f1:+.3f}")

    df_results = pd.DataFrame(rows)
    res_path = os.path.join(SCRIPT_DIR, "ensemble_results.csv")
    df_results.to_csv(res_path, index=False)

    # Summary table
    metrics = ["precision", "recall", "f1", "roc_auc"]
    summary_data = []
    for m in metrics:
        s_m = df_results[f"single_{m}"]
        e_m = df_results[f"ens_{m}"]
        diff = e_m - s_m
        summary_data.append({
            "metric": m,
            "single_mean": s_m.mean(),
            "single_std": s_m.std(),
            "ens_mean": e_m.mean(),
            "ens_std": e_m.std(),
            "mean_diff": diff.mean(),
            "std_diff": diff.std(),
            "ens_wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "single_wins": int((diff < 0).sum()),
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary["Single (Mean +/- Std)"] = df_summary.apply(
        lambda r: f"{r['single_mean']:.3f} +/- {r['single_std']:.3f}", axis=1
    )
    df_summary["Ensemble (Mean +/- Std)"] = df_summary.apply(
        lambda r: f"{r['ens_mean']:.3f} +/- {r['ens_std']:.3f}", axis=1
    )
    df_summary["Mean Diff (Ens - Single)"] = df_summary.apply(
        lambda r: f"{r['mean_diff']:+.3f} (std {r['std_diff']:.3f})", axis=1
    )
    df_summary["Win Record (W-T-L)"] = df_summary.apply(
        lambda r: f"{r['ens_wins']}-{r['ties']}-{r['single_wins']}", axis=1
    )

    print("\n" + "=" * 85)
    print(f"Single Model vs. 5-Seed Ensemble ({N_REPEATS} Stratified Splits)")
    print("=" * 85)
    disp_cols = ["metric", "Single (Mean +/- Std)", "Ensemble (Mean +/- Std)", "Mean Diff (Ens - Single)", "Win Record (W-T-L)"]
    print(df_summary[disp_cols].to_string(index=False))

    sum_path = os.path.join(SCRIPT_DIR, "ensemble_summary.csv")
    df_summary.to_csv(sum_path, index=False)
    print(f"\nSaved detailed results to {res_path}")
    print(f"Saved summary table to {sum_path}")
    print(f"Total time elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
