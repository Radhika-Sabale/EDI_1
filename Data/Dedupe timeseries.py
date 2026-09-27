"""
dedupe_timeseries.py
-----------------------
One-off repair. Removes duplicate rows from timeseries_data.csv --
i.e. more than one row for the same (location_id, year_month) pair.

This was caused by a bug in 04_gee_extract_timeseries.py: it only
skipped reprocessing a batch if EVERY location in that batch already
had a given month saved. After purge_incomplete_rows.py cleared a few
locations, most batches had at least one incomplete location, so the
whole batch (up to 20 locations) got reprocessed and re-appended --
including locations that were already complete, creating duplicate
rows for them. That bug is now fixed in 04, but the duplicates it
already created need to be cleaned out once.

Keeps the LAST occurrence of each (location_id, year_month) pair (the
most recently written one), drops the rest.

Run this ONCE, then rerun 05_clean_and_reshape.py.
"""

import pandas as pd

INPUT_CSV = "timeseries_data.csv"

df = pd.read_csv(INPUT_CSV)
before = len(df)

dupe_mask = df.duplicated(subset=["location_id", "year_month"], keep=False)
n_dupe_rows = dupe_mask.sum()

deduped = df.drop_duplicates(subset=["location_id", "year_month"], keep="last")

deduped.to_csv(INPUT_CSV, index=False)

print(f"Before: {before} rows")
print(f"Found {n_dupe_rows} rows involved in duplicate (location_id, year_month) pairs")
print(f"After dedupe: {len(deduped)} rows")
print(f"Removed: {before - len(deduped)} duplicate rows")

if n_dupe_rows > 0:
    affected = sorted(df[dupe_mask]["location_id"].unique())
    print(f"\n{len(affected)} locations had duplicates: {affected}")