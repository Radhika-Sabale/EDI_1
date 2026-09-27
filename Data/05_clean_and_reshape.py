"""
05_clean_and_reshape.py
--------------------------
Cleans timeseries_data.csv and reshapes it from long format (one row
per location x month) into per-location sequences ready for the LSTM
encoder: an array of shape (num_locations, 12, num_features).

WHAT COUNTS AS "MISSING":
A location can have a gap in two different ways, and this script
checks for both:
  1. The (location_id, year_month) row is missing entirely -- e.g. a
     batch/month that failed extraction and was never retried.
  2. The row exists, but ndvi or vh_backscatter is NaN/null -- e.g. a
     month where cloud cover masked out every pixel in the buffer, so
     Sentinel-2's median composite had nothing valid to average.

RULE APPLIED (see MAX_MISSING_MONTHS below):
  - If a location is missing MAX_MISSING_MONTHS or fewer months (per
    feature), the gap(s) are filled by linear interpolation using the
    surrounding months. Gaps at the very start/end of the 12-month
    window (which can't be linearly interpolated -- there's nothing
    on one side) are filled with the nearest available value instead.
  - If a location is missing MORE than MAX_MISSING_MONTHS, it is
    dropped entirely rather than guessed at.
Every decision is written to cleaning_report.csv so you can see
exactly what happened to each location, not have it hidden.

Run this FIFTH, after 04_gee_extract_timeseries.py.
Inputs:  timeseries_data.csv
Outputs: cleaned_sequences.npz   (X, y, location_ids -- ready for modeling)
         cleaning_report.csv     (per-location audit trail)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_CSV = "timeseries_data.csv"
SEQUENCES_OUT = "cleaned_sequences.npz"
REPORT_OUT = "cleaning_report.csv"

FEATURES = ["ndvi", "vh_backscatter"]
EXPECTED_MONTHS = 12
# Per-feature gap tolerance, not one shared threshold. NDVI (Sentinel-2, optical) is
# far more cloud-prone in some regions -- e.g. Indonesia's near-constant cloud cover --
# so a location shouldn't be dropped just because its optical channel had a rough year
# while its SAR channel (vh_backscatter, weather-independent) is perfectly complete.
MAX_MISSING_MONTHS = {
    "ndvi": 4,
    "vh_backscatter": 2,
}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV} "
          f"({df['location_id'].nunique()} unique locations)")

    # Full expected month index, taken from whatever months actually appear in the data
    all_months = sorted(df["year_month"].unique())
    if len(all_months) != EXPECTED_MONTHS:
        print(f"NOTE: found {len(all_months)} distinct months in the data, "
              f"expected {EXPECTED_MONTHS}. Using the months present: "
              f"{all_months[0]} to {all_months[-1]}.")

    report_rows = []
    kept_X, kept_y, kept_ids = [], [], []

    for location_id, group in df.groupby("location_id"):
        # Reindex onto the full month list -- rows for genuinely missing
        # (location, month) pairs become NaN rows here
        group = group.set_index("year_month").reindex(all_months)
        label = df.loc[df["location_id"] == location_id, "label"].iloc[0]

        missing_counts = {feat: int(group[feat].isna().sum()) for feat in FEATURES}
        exceeded = {feat: missing_counts[feat] > MAX_MISSING_MONTHS[feat] for feat in FEATURES}

        if any(exceeded.values()):
            over_budget = [
                f"{feat} missing {missing_counts[feat]} (max {MAX_MISSING_MONTHS[feat]})"
                for feat in FEATURES if exceeded[feat]
            ]
            report_rows.append({
                "location_id": location_id,
                "missing_ndvi": missing_counts["ndvi"],
                "missing_vh": missing_counts["vh_backscatter"],
                "action": "DROPPED",
                "reason": "; ".join(over_budget),
            })
            continue

        # Interpolate small gaps; limit_direction="both" also handles gaps
        # at the very start/end by filling with the nearest real value,
        # since linear interpolation alone can't extrapolate past an edge.
        filled = group[FEATURES].interpolate(method="linear", limit_direction="both")

        worst_gap = max(missing_counts.values())
        action = "COMPLETE" if worst_gap == 0 else f"INTERPOLATED (ndvi:{missing_counts['ndvi']}, vh:{missing_counts['vh_backscatter']})"
        report_rows.append({
            "location_id": location_id,
            "missing_ndvi": missing_counts["ndvi"],
            "missing_vh": missing_counts["vh_backscatter"],
            "action": action,
            "reason": "",
        })

        kept_X.append(filled[FEATURES].to_numpy())  # shape (12, 2)
        kept_y.append(int(label))
        kept_ids.append(location_id)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_OUT, index=False)

    X = np.stack(kept_X)                # shape (N, 12, len(FEATURES))
    y = np.array(kept_y)
    ids = np.array(kept_ids)

    np.savez(SEQUENCES_OUT, X=X, y=y, location_ids=ids, features=np.array(FEATURES))

    # ---- Summary ----
    n_total = len(report_df)
    n_dropped = (report_df["action"] == "DROPPED").sum()
    n_interp = report_df["action"].str.startswith("INTERPOLATED").sum()
    n_complete = (report_df["action"] == "COMPLETE").sum()

    print(f"\n{'='*60}")
    print(f"Locations processed: {n_total}")
    print(f"  Complete (no gaps):        {n_complete}")
    print(f"  Interpolated (small gaps): {n_interp}")
    print(f"  Dropped (too many gaps):   {n_dropped}")
    print(f"{'='*60}")
    print(f"Final dataset: X shape {X.shape}, y shape {y.shape}")
    print(f"Class balance: {int(y.sum())} mining / {len(y) - int(y.sum())} non-mining")
    print(f"\nFull per-location breakdown saved to {REPORT_OUT}")
    print(f"Model-ready sequences saved to {SEQUENCES_OUT}")

    if n_dropped > 0:
        print(f"\nDropped location_ids: {report_df[report_df['action']=='DROPPED']['location_id'].tolist()}")


if __name__ == "__main__":
    main()