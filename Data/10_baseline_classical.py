"""
10_baseline_classical.py

Person 2 — Baselines & Co-Modeling
Classical (non-sequential) baselines: Logistic Regression and Random Forest.

Loads the leak-free train/test split produced by 08_train_test_split.py and
evaluates both models with the shared evaluate_predictions() from
09_metrics_pipeline.py, so results are directly comparable to every later
model (plain LSTM, encoder+attention).

Run from inside the `Data/` folder (uses bare filenames, matching scripts 01-09).
"""

import importlib.util
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Always run relative to this file's folder (Data/), wherever it's launched from
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# --- import 09_metrics_pipeline.py without relying on Python identifier rules
# (module names can't start with a digit, so we load it by file path instead
# of `import 09_metrics_pipeline`)
def _load_metrics_pipeline():
    spec = importlib.util.spec_from_file_location(
        "metrics_pipeline", "09_metrics_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics_pipeline = _load_metrics_pipeline()
evaluate_predictions = metrics_pipeline.evaluate_predictions



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
    """
    Loads the split and prints the available keys first, since the exact
    key names depend on how 08_train_test_split.py saved them — adjust the
    variable names below if they don't match what gets printed.
    """
    data = np.load(path, allow_pickle=True)
    print(f"Keys found in {path}: {list(data.files)}")
    return data


def main():
    data = load_split()

    # --- Adjust these four lines to match your actual saved key names ---
    n_features = len(data["feature_names"])
    X_train = to_sequences(data["X_train"], n_features)   # (n, months, features)
    X_test = to_sequences(data["X_test"], n_features)
    y_train = data["y_train"]
    y_test = data["y_test"]
    countries_test = load_countries(data["ids_test"])
    print(f"Sequence shape: {X_train.shape}  features: {list(data['feature_names'])}")

    # Flatten the (months, features) sequence into a single feature vector
    # per location — classical models have no notion of time order, so this
    # just gives them every (month, feature) value as an independent column.
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    print(f"\nTrain shape: {X_train_flat.shape}, Test shape: {X_test_flat.shape}")

    results = {}

    # --- Logistic Regression -------------------------------------------------
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_flat, y_train)
    logreg_proba = logreg.predict_proba(X_test_flat)[:, 1]

    print("\n=== Logistic Regression ===")
    results["logistic_regression"] = evaluate_predictions(
        y_true=y_test,
        y_pred_proba=logreg_proba,
        model_name="Logistic Regression",
        countries=countries_test,
    )

    # --- Random Forest --------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_flat, y_train)
    rf_proba = rf.predict_proba(X_test_flat)[:, 1]

    print("\n=== Random Forest ===")
    results["random_forest"] = evaluate_predictions(
        y_true=y_test,
        y_pred_proba=rf_proba,
        model_name="Random Forest",
        countries=countries_test,
    )

    # Quick feature-importance sanity check for RF: which (month, feature)
    # positions matter most. Useful sanity check / paper figure candidate.
    importances = rf.feature_importances_
    n_months, n_features = X_train.shape[1], X_train.shape[2]
    importances_grid = importances.reshape(n_months, n_features)
    print("\nRF feature importance by month (rows) x feature (cols):")
    print(np.round(importances_grid, 4))

    return results


if __name__ == "__main__":
    main()