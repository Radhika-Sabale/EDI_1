"""
Model/02_multiseed_evaluation.py

Runs all four models (Logistic Regression, Random Forest, plain LSTM,
Attention Fusion) over many random train/test splits and reports
mean +/- standard deviation, instead of trusting a single 54-sample test set.

How it works
  1. Pools the original train + test locations back together (267 locations).
  2. For each repeat: makes a new stratified split (by country + label, like
     Data/08), re-fits the feature scaler on that split's TRAIN part only
     (so nothing leaks from test), trains every model, and scores it with the
     shared evaluate_predictions() from Data/09_metrics_pipeline.py.
  3. Prints a mean +/- std table, a paired attention-vs-baseline comparison,
     and a per-country F1 table. Saves everything to CSV next to this script.

Run time: a few minutes on CPU with the default 10 repeats.
Change the number of repeats with the N_REPEATS constant below (or set the
MULTISEED_REPEATS environment variable).

Folder layout assumed:  <project>/Data/...   and   <project>/Model/<this file>
Paths are resolved relative to this file, so the VS Code run button works.
"""

import contextlib
import importlib.util
import io
import os
import time

import numpy as np
import pandas as pd
import os

# Silence TensorFlow's start-up chatter (must be set BEFORE importing tensorflow).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # hide info/warning logs from the C++ backend
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # hide the oneDNN notice
import tensorflow as tf
tf.get_logger().setLevel("ERROR")   # hide retracing / deprecation warnings
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

N_REPEATS = int(os.environ.get("MULTISEED_REPEATS", 10))
TEST_FRACTION = 0.2          # ~54 of 267 locations, same size as the original test set
BASE_SEED = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data")


# --------------------------------------------------------------------------
# Shared metrics function (same one every other script uses)
# --------------------------------------------------------------------------
def _load_metrics_pipeline():
    path = os.path.join(DATA_DIR, "09_metrics_pipeline.py")
    spec = importlib.util.spec_from_file_location("metrics_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate_predictions = _load_metrics_pipeline().evaluate_predictions


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def load_countries(ids, csv_path):
    """Look up each location's country via input_locations.csv. None if it can't."""
    try:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        country_col = next((cols[c] for c in cols if "country" in c), None)
        id_col = next((cols[c] for c in ("location_id", "id", "loc_id", "site_id", "name") if c in cols), None)
        if country_col is None or id_col is None:
            print(f"[countries] Couldn't find id/country columns in {csv_path}. Columns: {list(df.columns)}")
            return None
        lookup = dict(zip(df[id_col].astype(str), df[country_col]))
        out = np.array([lookup.get(str(i), "unknown") for i in ids])
        if (out == "unknown").all():
            print("[countries] No ids matched input_locations.csv - skipping country stratification/breakdown.")
            return None
        return out
    except Exception as e:
        print(f"[countries] Skipping country stratification/breakdown: {e}")
        return None


def to_sequences(X, n_features):
    """(n, months, features), whether it was saved flat or not."""
    return X.reshape(X.shape[0], -1, n_features) if X.ndim == 2 else X


def load_pooled_data():
    data = np.load(os.path.join(DATA_DIR, "train_test_split.npz"), allow_pickle=True)
    n_feat = len(data["feature_names"])
    X = np.concatenate([to_sequences(data["X_train"], n_feat),
                        to_sequences(data["X_test"], n_feat)]).astype("float32")
    y = np.concatenate([data["y_train"], data["y_test"]]).astype(int)
    ids = np.concatenate([data["ids_train"], data["ids_test"]])
    countries = load_countries(ids, os.path.join(DATA_DIR, "input_locations.csv"))
    return X, y, countries


def make_split(y, strata, seed):
    idx = np.arange(len(y))
    try:
        return train_test_split(idx, test_size=TEST_FRACTION, stratify=strata, random_state=seed)
    except ValueError:  # a country+label group too small to split -> fall back to label only
        return train_test_split(idx, test_size=TEST_FRACTION, stratify=y, random_state=seed)


def rescale(X_tr, X_te):
    """Standardise each feature using TRAIN statistics only (leak-free per split)."""
    f = X_tr.shape[2]
    sc = StandardScaler().fit(X_tr.reshape(-1, f))
    def apply(X):
        return sc.transform(X.reshape(-1, f)).reshape(X.shape).astype("float32")
    return apply(X_tr), apply(X_te)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class TemporalAttention(layers.Layer):
    """One weight per month (softmax, sums to 1) -> weighted sum of LSTM outputs."""
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(units, activation="tanh")
        self.score_out = layers.Dense(1)

    def call(self, encoder_outputs):
        scores = self.score_out(self.score_dense(encoder_outputs))
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(weights * encoder_outputs, axis=1)
        return context, tf.squeeze(weights, axis=-1)


def _compile(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def build_plain_lstm(t, f):
    return _compile(keras.Sequential([
        layers.Input(shape=(t, f)),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]))


def build_attention_model(t, f):
    inputs = keras.Input(shape=(t, f))
    encoded = layers.LSTM(32, return_sequences=True)(inputs)
    context, _ = TemporalAttention(32)(encoded)
    x = layers.Dropout(0.3)(context)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, out))


def fit_and_predict_keras(builder, X_tr, y_tr, X_te, seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = builder(X_tr.shape[1], X_tr.shape[2])
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}
    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=12, restore_best_weights=True)
    model.fit(X_tr, y_tr.astype("float32"), validation_split=0.15, epochs=150,
              batch_size=16, class_weight=class_weight, callbacks=[stop], verbose=0)
    return model.predict(X_te, verbose=0).ravel()


def run_logreg(X_tr, y_tr, X_te, seed):
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    return m.predict_proba(X_te.reshape(len(X_te), -1))[:, 1]


def run_rf(X_tr, y_tr, X_te, seed):
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=seed, n_jobs=-1)
    m.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    return m.predict_proba(X_te.reshape(len(X_te), -1))[:, 1]


MODELS = {
    "Logistic Regression": run_logreg,
    "Random Forest": run_rf,
    "Plain LSTM": lambda a, b, c, s: fit_and_predict_keras(build_plain_lstm, a, b, c, s),
    "Attention Fusion": lambda a, b, c, s: fit_and_predict_keras(build_attention_model, a, b, c, s),
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    X, y, countries = load_pooled_data()
    strata = np.array([f"{c}_{l}" for c, l in zip(countries, y)]) if countries is not None else y
    print(f"Pooled data: {X.shape[0]} locations, sequence shape {X.shape[1:]}, "
          f"{N_REPEATS} repeats x {len(MODELS)} models\n")

    rows, country_rows = [], []
    start = time.time()
    for r in range(N_REPEATS):
        seed = BASE_SEED + r
        tr, te = make_split(y, strata, seed)
        X_tr, X_te = rescale(X[tr], X[te])
        for name, run in MODELS.items():
            proba = run(X_tr, y[tr], X_te, seed)
            with contextlib.redirect_stdout(io.StringIO()):   # keep the console readable
                res = evaluate_predictions(
                    y_true=y[te], y_pred_proba=proba, model_name=name,
                    countries=countries[te] if countries is not None else None)
            rows.append({"repeat": r, "model": name, "precision": res["precision"],
                         "recall": res["recall"], "f1": res["f1"], "roc_auc": res["roc_auc"]})
            for country, m in res.get("per_country", {}).items():
                country_rows.append({"repeat": r, "model": name, "country": country,
                                     "n": m["n"], "f1": m["f1"]})
        print(f"  repeat {r + 1}/{N_REPEATS} done  ({time.time() - start:.0f}s elapsed)")

    results = pd.DataFrame(rows)
    order = list(MODELS)
    metrics = ["precision", "recall", "f1", "roc_auc"]

    # --- mean +/- std table ---
    summary = results.groupby("model")[metrics].agg(["mean", "std"]).reindex(order)
    table = pd.DataFrame({m: [f"{summary[(m, 'mean')][k]:.3f} +/- {summary[(m, 'std')][k]:.3f}"
                              for k in order] for m in metrics}, index=order)
    print(f"\n{'=' * 78}\nMean +/- std over {N_REPEATS} random splits "
          f"(~{int(round(len(y) * TEST_FRACTION))} test locations each)\n{'=' * 78}")
    print(table.to_string())

    # --- paired comparison: attention vs each other model ---
    print(f"\nPaired ROC-AUC difference (Attention Fusion minus other), same splits:")
    auc = results.pivot(index="repeat", columns="model", values="roc_auc")
    for other in order[:-1]:
        d = auc["Attention Fusion"] - auc[other]
        print(f"  vs {other:<20} mean diff {d.mean():+.3f}  (std {d.std():.3f}), "
              f"attention better in {(d > 0).sum()}/{len(d)} repeats")

    # --- per-country F1 ---
    if country_rows:
        pc = pd.DataFrame(country_rows).groupby(["country", "model"])["f1"].mean().unstack()[order]
        print("\nMean F1 by country (countries with few test locations are noisy):")
        print(pc.round(3).to_string())

    results.to_csv(os.path.join(SCRIPT_DIR, "multiseed_results.csv"), index=False)
    table.to_csv(os.path.join(SCRIPT_DIR, "multiseed_summary.csv"))
    print(f"\nSaved multiseed_results.csv and multiseed_summary.csv in {SCRIPT_DIR}")
    print("Note: repeats share locations, so they are not fully independent - the std is a\n"
          "rough guide to split-to-split variation, not a formal confidence interval.")


if __name__ == "__main__":
    main()