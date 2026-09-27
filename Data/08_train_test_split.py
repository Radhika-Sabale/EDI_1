"""
08_train_test_split.py
-------------------------
Builds the train/test split and applies feature scaling -- the last
step before any model (baseline or LSTM) can be trained.

SPLIT STRATEGY:
Stratified by (country, label) rather than a full country holdout.
A full holdout (train on 4 countries, test entirely on the 5th) is the
"purest" test of geographic generalization, but with only 5 countries
and 267 locations total, it would leave the test set tiny and
single-country, and training would be missing an entire geography's
patterns. Stratifying by country+label instead ensures every country
and both classes are proportionally represented in both train and
test -- so no single country or class can dominate either side, and
performance can still be broken down by country afterward to see how
well the model generalizes across geographies.
LIMITATION TO DISCLOSE IN YOUR PAPER: this is not a full geographic
holdout, so cross-country generalization is not being strictly tested
here. A leave-one-country-out cross-validation (each country held out
as test in turn, results averaged) would be the rigorous version --
worth doing in weeks 5-6 as an additional robustness check once you
have a working model, not required to start training.

SCALING:
StandardScaler is fit ONLY on the training set, then applied to both
train and test. Fitting on the full dataset (including test) would
leak test-set statistics into training and inflate results.

Run this AFTER 07_compute_derived_features.py.
Inputs:  features_sequences.npz, mining_sites.csv, negative_sites.csv
Outputs: train_test_split.npz (X_train, y_train, ids_train, X_test,
         y_test, ids_test -- all scaled)
         scaler_params.npz (mean, scale -- so new data can be scaled
         identically later)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_NPZ = "features_sequences.npz"
MINING_CSV = "mining_sites.csv"
NEGATIVE_CSV = "negative_sites.csv"
OUTPUT_SPLIT_NPZ = "train_test_split.npz"
OUTPUT_SCALER_NPZ = "scaler_params.npz"

TEST_FRACTION = 0.2
RANDOM_SEED = 42

data = np.load(FEATURES_NPZ, allow_pickle=True)
X = data["X"]                      # (N, 12, n_features)
y = data["y"]
location_ids = data["location_ids"]
feature_names = data["feature_names"]

n_locations, n_months, n_features = X.shape
print(f"Loaded {n_locations} locations x {n_months} months x {n_features} features")

# ---- Attach country to each location_id ----
mining_df = pd.read_csv(MINING_CSV)[["location_id", "country"]]
negative_df = pd.read_csv(NEGATIVE_CSV)[["location_id", "country"]]
country_lookup = pd.concat([mining_df, negative_df]).set_index("location_id")["country"]

countries = np.array([country_lookup.get(lid, "UNKNOWN") for lid in location_ids])
if "UNKNOWN" in countries:
    n_unknown = (countries == "UNKNOWN").sum()
    print(f"WARNING: {n_unknown} locations had no matching country -- check mining_sites.csv/negative_sites.csv")

# ---- Stratify on combined (country, label) key ----
strat_key = np.array([f"{c}_{l}" for c, l in zip(countries, y)])

indices = np.arange(n_locations)
train_idx, test_idx = train_test_split(
    indices, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=strat_key
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
ids_train, ids_test = location_ids[train_idx], location_ids[test_idx]
countries_train, countries_test = countries[train_idx], countries[test_idx]

# ---- Scale: fit on train only ----
# Flatten to (N * months, features) to fit the scaler across all timesteps,
# then reshape back -- StandardScaler expects 2D input.
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, n_features)
scaler.fit(X_train_flat)

X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

np.savez(
    OUTPUT_SPLIT_NPZ,
    X_train=X_train_scaled, y_train=y_train, ids_train=ids_train,
    X_test=X_test_scaled, y_test=y_test, ids_test=ids_test,
    feature_names=feature_names,
)
np.savez(OUTPUT_SCALER_NPZ, mean=scaler.mean_, scale=scaler.scale_, feature_names=feature_names)

# ---- Report ----
print(f"\nTrain: {len(train_idx)} locations, Test: {len(test_idx)} locations")
print(f"\nClass balance -- Train: {int(y_train.sum())} mining / {len(y_train) - int(y_train.sum())} non-mining")
print(f"Class balance -- Test:  {int(y_test.sum())} mining / {len(y_test) - int(y_test.sum())} non-mining")

print("\nCountry breakdown:")
for country in sorted(set(countries)):
    train_count = (countries_train == country).sum()
    test_count = (countries_test == country).sum()
    print(f"  {country}: train={train_count}, test={test_count}")

print(f"\nSplit saved to {OUTPUT_SPLIT_NPZ}")
print(f"Scaler params saved to {OUTPUT_SCALER_NPZ}")