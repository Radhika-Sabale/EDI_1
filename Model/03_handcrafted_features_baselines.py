"""
Model/03_handcrafted_features_baselines.py

Adds domain-specific hand-crafted summary features per location computed from
each location's 12-month series for NDVI and VH (and their month-to-month deltas).

Features extracted per location:
  NDVI & VH Backscatter:
    - mean, std, min, max, range (max - min)
    - month of minimum (1-indexed, 1..12)
    - first-half mean minus second-half mean (months 1-6 minus months 7-12)
    - linear trend slope (degree-1 polynomial fit across the 12 months)
    - largest single-month drop (maximum month-over-month decrease)
  Deltas:
    - delta_ndvi std, delta_ndvi min
    - delta_vh std, delta_vh min

Evaluates Logistic Regression and Random Forest on:
  (a) Raw flattened features only (72 features: 12 months x 6 channels)
  (b) Hand-crafted features only (22 summary features)
  (c) Combined features (94 features: hand-crafted + raw flattened)

Uses the same repeated stratified splits (country + label) and leak-free per-split
StandardScaler as Model/02_multiseed_evaluation.py.

How to run:
  python Model/03_handcrafted_features_baselines.py
  (or via the VS Code run button from anywhere)
"""

import contextlib
import importlib.util
import io
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
make_split = m02.make_split

N_REPEATS = int(os.environ.get("MULTISEED_REPEATS", 10))
BASE_SEED = m02.BASE_SEED


# --------------------------------------------------------------------------
# Load pooled data with physical unscaled features
# --------------------------------------------------------------------------
def load_all_data():
    """
    Loads pooled locations preserving the exact same ordering as Model/02,
    returning:
      X_raw: (267, 72) flattened sequence features (12 x 6)
      X_hc:  (267, 22) unscaled hand-crafted summary features
      y:     (267,) labels
      countries: (267,) country names
    """
    s = np.load(os.path.join(DATA_DIR, "train_test_split.npz"), allow_pickle=True)
    f = np.load(os.path.join(DATA_DIR, "features_sequences.npz"), allow_pickle=True)

    pooled_ids = np.concatenate([s["ids_train"], s["ids_test"]])
    y = np.concatenate([s["y_train"], s["y_test"]]).astype(int)
    countries = load_countries(pooled_ids, os.path.join(DATA_DIR, "input_locations.csv"))

    # Map location_id to index in features_sequences.npz
    f_map = {lid: i for i, lid in enumerate(f["location_ids"])}
    reorder_idx = [f_map[lid] for lid in pooled_ids]
    X_seq = f["X"][reorder_idx].astype("float32")  # (267, 12, 6)

    # Raw flattened features: (267, 72)
    X_raw = X_seq.reshape(len(X_seq), -1)

    # Compute handcrafted summary features
    hc_features, feat_names = extract_handcrafted_features(X_seq)

    return X_raw, hc_features, y, countries, feat_names


def extract_handcrafted_features(X_seq):
    """
    Computes summary features from each location's 12-month series.
    X_seq: (N, 12, 6)
    Feature channels:
      0: ndvi, 1: vh_backscatter, 2: delta_ndvi, 3: delta_vh, 4: roll3_ndvi, 5: roll3_vh
    """
    t = np.arange(12)
    feats = []
    feat_names = [
        # NDVI features
        "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "ndvi_range",
        "ndvi_month_of_min", "ndvi_half_diff", "ndvi_slope", "ndvi_largest_drop",
        # VH features
        "vh_mean", "vh_std", "vh_min", "vh_max", "vh_range",
        "vh_month_of_min", "vh_half_diff", "vh_slope", "vh_largest_drop",
        # Delta features
        "delta_ndvi_std", "delta_ndvi_min", "delta_vh_std", "delta_vh_min"
    ]

    for i in range(len(X_seq)):
        ndvi = X_seq[i, :, 0]
        vh = X_seq[i, :, 1]
        d_ndvi = X_seq[i, :, 2]
        d_vh = X_seq[i, :, 3]

        row = [
            # NDVI summary
            float(ndvi.mean()),
            float(ndvi.std()),
            float(ndvi.min()),
            float(ndvi.max()),
            float(ndvi.max() - ndvi.min()),
            float(np.argmin(ndvi) + 1),                      # Month of minimum (1..12)
            float(ndvi[:6].mean() - ndvi[6:].mean()),        # First half mean minus second half mean
            float(np.polyfit(t, ndvi, 1)[0]),                # Linear trend slope
            float(np.max(ndvi[:-1] - ndvi[1:])),             # Largest single-month drop

            # VH summary
            float(vh.mean()),
            float(vh.std()),
            float(vh.min()),
            float(vh.max()),
            float(vh.max() - vh.min()),
            float(np.argmin(vh) + 1),
            float(vh[:6].mean() - vh[6:].mean()),
            float(np.polyfit(t, vh, 1)[0]),
            float(np.max(vh[:-1] - vh[1:])),

            # Delta statistics
            float(d_ndvi[1:].std()),
            float(d_ndvi[1:].min()),
            float(d_vh[1:].std()),
            float(d_vh[1:].min()),
        ]
        feats.append(row)

    return np.array(feats, dtype="float32"), feat_names


# --------------------------------------------------------------------------
# Main multi-seed benchmark
# --------------------------------------------------------------------------
def main():
    X_raw, X_hc, y, countries, feat_names = load_all_data()
    strata = np.array([f"{c}_{l}" for c, l in zip(countries, y)]) if countries is not None else y

    print(f"Loaded {len(y)} locations:")
    print(f"  Raw flattened features: {X_raw.shape[1]}")
    print(f"  Hand-crafted features:  {X_hc.shape[1]} ({', '.join(feat_names[:6])}...)")
    print(f"  Combined features:      {X_raw.shape[1] + X_hc.shape[1]}")
    print(f"Running {N_REPEATS} repeats across 6 baseline configurations...\n")

    models_config = {
        "LR (Raw)": lambda tr, te, s: (
            LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr["raw"], y_tr),
            tr["raw"], te["raw"]
        ),
        "LR (Handcrafted)": lambda tr, te, s: (
            LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr["hc"], y_tr),
            tr["hc"], te["hc"]
        ),
        "LR (Handcrafted+Raw)": lambda tr, te, s: (
            LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr["comb"], y_tr),
            tr["comb"], te["comb"]
        ),
        "RF (Raw)": lambda tr, te, s: (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=s, n_jobs=-1).fit(tr["raw"], y_tr),
            tr["raw"], te["raw"]
        ),
        "RF (Handcrafted)": lambda tr, te, s: (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=s, n_jobs=-1).fit(tr["hc"], y_tr),
            tr["hc"], te["hc"]
        ),
        "RF (Handcrafted+Raw)": lambda tr, te, s: (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=s, n_jobs=-1).fit(tr["comb"], y_tr),
            tr["comb"], te["comb"]
        ),
    }

    rows = []
    start = time.time()

    for r in range(N_REPEATS):
        seed = BASE_SEED + r
        tr_idx, te_idx = make_split(y, strata, seed)
        y_tr, y_te = y[tr_idx], y[te_idx]
        c_te = countries[te_idx] if countries is not None else None

        # Leak-free scaling fitted strictly on training data
        sc_raw = StandardScaler().fit(X_raw[tr_idx])
        raw_tr = sc_raw.transform(X_raw[tr_idx])
        raw_te = sc_raw.transform(X_raw[te_idx])

        sc_hc = StandardScaler().fit(X_hc[tr_idx])
        hc_tr = sc_hc.transform(X_hc[tr_idx])
        hc_te = sc_hc.transform(X_hc[te_idx])

        comb_tr = np.hstack([hc_tr, raw_tr])
        comb_te = np.hstack([hc_te, raw_te])

        data_tr = {"raw": raw_tr, "hc": hc_tr, "comb": comb_tr}
        data_te = {"raw": raw_te, "hc": hc_te, "comb": comb_te}

        for name, run_fn in models_config.items():
            clf, _, X_test_feat = run_fn(data_tr, data_te, seed)
            proba = clf.predict_proba(X_test_feat)[:, 1]

            with contextlib.redirect_stdout(io.StringIO()):
                res = evaluate_predictions(
                    y_true=y_te,
                    y_pred_proba=proba,
                    model_name=name,
                    countries=c_te,
                )

            rows.append({
                "repeat": r,
                "model": name,
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "roc_auc": res["roc_auc"],
            })

        print(f"  Repeat {r + 1}/{N_REPEATS} completed ({time.time() - start:.1f}s elapsed)")

    results = pd.DataFrame(rows)
    metrics = ["precision", "recall", "f1", "roc_auc"]
    order = list(models_config.keys())

    summary = results.groupby("model")[metrics].agg(["mean", "std"]).reindex(order)
    table = pd.DataFrame(
        {m: [f"{summary[(m, 'mean')][k]:.3f} +/- {summary[(m, 'std')][k]:.3f}" for k in order] for m in metrics},
        index=order,
    )

    print(f"\n{'=' * 85}")
    print(f"Handcrafted vs Raw Features: Mean +/- Std over {N_REPEATS} Repeated Splits")
    print(f"{'=' * 85}")
    print(table.to_string())

    # Paired comparisons
    auc_pivot = results.pivot(index="repeat", columns="model", values="roc_auc")
    print(f"\n{'=' * 85}")
    print("Paired ROC-AUC Comparisons (Same Splits):")
    print(f"{'=' * 85}")

    comparisons = [
        ("LR (Handcrafted)", "LR (Raw)"),
        ("LR (Handcrafted+Raw)", "LR (Raw)"),
        ("RF (Handcrafted)", "RF (Raw)"),
        ("RF (Handcrafted+Raw)", "RF (Raw)"),
    ]
    for m_cand, m_base in comparisons:
        diff = auc_pivot[m_cand] - auc_pivot[m_base]
        wins = (diff > 0).sum()
        ties = (diff == 0).sum()
        print(f"  {m_cand:<22} minus {m_base:<12} -> "
              f"Mean diff: {diff.mean():+.3f} (std: {diff.std():.3f}) | Better in {wins}/{N_REPEATS} splits (ties: {ties})")

    # Save outputs
    res_path = os.path.join(SCRIPT_DIR, "handcrafted_results.csv")
    sum_path = os.path.join(SCRIPT_DIR, "handcrafted_summary.csv")
    results.to_csv(res_path, index=False)
    table.to_csv(sum_path)
    print(f"\nSaved detailed results to {res_path}")
    print(f"Saved summary table to {sum_path}")


if __name__ == "__main__":
    main()
