"""
03_combine_locations.py
-------------------------
Combines mining_sites.csv and negative_sites.csv into a
single input_locations.csv, ready for the GEE time-series extraction
script (04_gee_extract_timeseries.py).

Run this THIRD, after both 01 and 02 have finished.

Output: input_locations.csv
    columns: location_id, latitude, longitude, label
"""

import pandas as pd

MINING_CSV = "mining_sites.csv"
NEGATIVE_CSV = "negative_sites.csv"
OUTPUT_CSV = "input_locations.csv"

mining_df = pd.read_csv(MINING_CSV)
negative_df = pd.read_csv(NEGATIVE_CSV)

combined = pd.concat([mining_df, negative_df], ignore_index=True)

# Basic sanity checks before saving
assert combined["location_id"].is_unique, "Duplicate location_id found!"
assert combined["label"].isin([0, 1]).all(), "Unexpected label values found!"

combined.to_csv(OUTPUT_CSV, index=False)

print(f"Combined dataset saved to {OUTPUT_CSV}")
print(f"  Mining sites:     {len(mining_df)}")
print(f"  Non-mining sites: {len(negative_df)}")
print(f"  Total:            {len(combined)}")