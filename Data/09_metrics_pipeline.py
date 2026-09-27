"""
09_metrics_pipeline.py
--------------------------
The shared evaluation function every model in the project runs
through -- baselines, plain LSTM, the attention model, and the
NDVI-only/VH-only/fused ablation study. Having ONE function for this
means every model's numbers are computed the same way and are
directly comparable, which matters a lot for the ablation study and
for the final baseline-vs-main-model comparison in your results
section.

USAGE:
Import evaluate_predictions() from this file in any model script:

    from importlib import import_module
    metrics_module = import_module("09_metrics_pipeline")
    # (or rename this file to metrics_pipeline.py and import normally
    #  once you're past the numbered-script stage -- see note below)

    results = metrics_module.evaluate_predictions(
        y_true=y_test, y_pred_proba=model_predict_proba(X_test),
        model_name="Random Forest", countries=countries_test,
    )

NOTE ON THE FILENAME: Python can't import a module whose name starts
with a digit (e.g. "09_metrics_pipeline") using a normal `import`
statement -- that's why the usage example above uses import_module.
Once you're done with the numbered pipeline scripts and start writing
model scripts, consider copying this file to metrics_pipeline.py (no
number) so you can just write `from metrics_pipeline import
evaluate_predictions` normally. Both files can coexist; the numbered
one documents pipeline order, the renamed one is what you'd actually
import from.

WHAT IT COMPUTES:
  - Precision, recall, F1 (at a chosen probability threshold, default 0.5)
  - ROC-AUC (threshold-independent, uses the raw probabilities)
  - Confusion matrix
  - Optional per-country breakdown, if you pass `countries` -- this is
    what lets you check whether a model does well overall but poorly
    on one specific country, which a single aggregate number would hide.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)


def evaluate_predictions(y_true, y_pred_proba, model_name="model", threshold=0.5, countries=None):
    """
    y_true: array of 0/1 true labels
    y_pred_proba: array of predicted probabilities of class 1 (mining)
    model_name: label for the printed report
    threshold: probability cutoff for turning probabilities into 0/1 predictions
    countries: optional array (same length as y_true) for a per-country breakdown

    Returns a dict of the computed metrics, so results can be collected
    across multiple models for a comparison table later.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC needs both classes present in y_true, and is undefined otherwise
    if len(set(y_true)) < 2:
        roc_auc = float("nan")
        print("WARNING: only one class present in y_true -- ROC-AUC is undefined, reporting NaN.")
    else:
        roc_auc = roc_auc_score(y_true, y_pred_proba)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"Results: {model_name}  (threshold={threshold})")
    print(f"{'='*60}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC:   NaN")
    print(f"\nConfusion matrix (rows=true, cols=predicted, order=[non-mining, mining]):")
    print(cm)
    print(f"\n{classification_report(y_true, y_pred, target_names=['non-mining', 'mining'], zero_division=0)}")

    results = {
        "model_name": model_name, "threshold": threshold,
        "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true), "n_positive": int(y_true.sum()),
    }

    if countries is not None:
        countries = np.asarray(countries)
        print(f"\nPer-country breakdown:")
        per_country = {}
        for country in sorted(set(countries)):
            mask = countries == country
            n = mask.sum()
            if n == 0:
                continue
            c_true, c_pred, c_proba = y_true[mask], y_pred[mask], y_pred_proba[mask]
            c_precision = precision_score(c_true, c_pred, zero_division=0)
            c_recall = recall_score(c_true, c_pred, zero_division=0)
            c_f1 = f1_score(c_true, c_pred, zero_division=0)
            print(f"  {country} (n={n}): precision={c_precision:.4f}, recall={c_recall:.4f}, f1={c_f1:.4f}")
            per_country[country] = {
                "n": int(n), "precision": c_precision, "recall": c_recall, "f1": c_f1,
            }
        results["per_country"] = per_country

    return results


if __name__ == "__main__":
    print("This is a library module -- import evaluate_predictions() from your model scripts.")
    print("Running a quick self-test with random dummy data:\n")
    rng = np.random.default_rng(0)
    dummy_true = rng.integers(0, 2, size=50)
    dummy_proba = rng.random(50)
    dummy_countries = rng.choice(["India", "Brazil", "Australia"], size=50)
    evaluate_predictions(dummy_true, dummy_proba, model_name="dummy self-test", countries=dummy_countries)