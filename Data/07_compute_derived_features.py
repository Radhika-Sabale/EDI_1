"""
07_compute_derived_features.py
---------------------------------
Adds derived features on top of the raw monthly NDVI/VH sequences in
cleaned_sequences.npz: month-to-month deltas (ΔNDVI, ΔVH) and rolling
averages. These are the features Person 1's plan calls for beyond the
raw values, and they matter for a mining-detection model specifically
because disturbance shows up as a CHANGE, not just an absolute level --
a location with low NDVI year-round (e.g. naturally arid/sparse
vegetation) looks very different from one where NDVI recently dropped.
Deltas and rolling averages give the model that "recently changed"
signal directly, rather than making it infer it from raw levels alone.

FEATURES ADDED (per month, per location):
  - delta_ndvi, delta_vh: this month's value minus last month's value.
    The first month in each sequence has no prior month to compare
    against, so its delta is set to 0 (interpreted as "no observed
    change yet" rather than a missing value -- keeps the array
    numeric and avoids NaNs feeding into the model).
  - roll_ndvi, roll_vh: a trailing 3-month rolling average (current
    month + the 2 before it). For the first two months, where fewer
    than 3 prior months exist, the average is taken over whatever
    months ARE available (months 1 and 2 use 1- and 2-month windows
    respectively) rather than padding with zeros, so the rolling
    average is never artificially dragged down at the start of a
    sequence.

Output feature order per timestep: [ndvi, vh, delta_ndvi, delta_vh,
roll_ndvi, roll_vh] -- 6 features per month instead of the original 2.

Run this AFTER 05_clean_and_reshape.py.
Input:  cleaned_sequences.npz       (X: N x 12 x 2, y, location_ids)
Output: features_sequences.npz      (X: N x 12 x 6, y, location_ids, feature_names)
"""

import numpy as np

INPUT_NPZ = "cleaned_sequences.npz"
OUTPUT_NPZ = "features_sequences.npz"

ROLLING_WINDOW = 3

data = np.load(INPUT_NPZ, allow_pickle=True)
X = data["X"]              # shape (N, 12, 2) -- [ndvi, vh]
y = data["y"]
location_ids = data["location_ids"]
base_features = list(data["features"])  # ["ndvi", "vh"]

n_locations, n_months, n_base = X.shape
print(f"Loaded {n_locations} locations x {n_months} months x {n_base} base features "
      f"({base_features}) from {INPUT_NPZ}")


def compute_deltas(seq):
    """seq: (n_months, n_base). Returns (n_months, n_base), first row = 0."""
    deltas = np.zeros_like(seq)
    deltas[1:] = seq[1:] - seq[:-1]
    return deltas


def compute_rolling_mean(seq, window):
    """seq: (n_months, n_base). Trailing rolling mean, expanding at the start."""
    n = seq.shape[0]
    out = np.zeros_like(seq)
    for t in range(n):
        start = max(0, t - window + 1)
        out[t] = seq[start:t + 1].mean(axis=0)
    return out


all_features = np.zeros((n_locations, n_months, n_base * 3))

for i in range(n_locations):
    seq = X[i]  # (12, n_base)
    deltas = compute_deltas(seq)
    rolling = compute_rolling_mean(seq, ROLLING_WINDOW)
    all_features[i] = np.concatenate([seq, deltas, rolling], axis=1)

feature_names = (
    base_features
    + [f"delta_{f}" for f in base_features]
    + [f"roll{ROLLING_WINDOW}_{f}" for f in base_features]
)

np.savez(
    OUTPUT_NPZ,
    X=all_features,
    y=y,
    location_ids=location_ids,
    feature_names=np.array(feature_names),
)

print(f"Output shape: {all_features.shape} (features: {feature_names})")
print(f"Saved to {OUTPUT_NPZ}")

# Quick sanity check on one location so you can see the values are sane
print(f"\nExample -- {location_ids[0]}:")
for month in range(n_months):
    row = all_features[0, month]
    print(f"  month {month}: " + ", ".join(f"{name}={val:.4f}" for name, val in zip(feature_names, row)))