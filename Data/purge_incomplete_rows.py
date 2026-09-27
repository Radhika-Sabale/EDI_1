"""
purge_incomplete_rows.py
---------------------------
One-off repair utility. Removes rows from timeseries_data.csv where
vh_backscatter is null, so script 04 will reprocess them with the new
VV fallback in get_monthly_vh().

IMPORTANT -- this deliberately does NOT touch rows where only ndvi is
null. Those are genuine Sentinel-2 cloud-cover gaps (common in
monsoon-affected regions like Indonesia) -- the historical imagery for
that month really doesn't have a clear view of the ground, so
re-running the extraction queries the exact same fixed imagery and
gets the same null result. There's no equivalent "fallback band" for
optical data the way VV backstops VH, so purging and reprocessing
those rows would just waste GEE calls for no benefit. They're left
for 05_clean_and_reshape.py's existing interpolate-or-drop logic to
handle as-is.

Run this ONCE, right after updating get_monthly_vh(), then rerun
04_gee_extract_timeseries.py -- it will see the purged rows as "not
done yet" and refill them with the corrected logic.

Prints a before/after count and which locations had rows removed, so
you can see exactly what will be reprocessed.
"""

import pandas as pd

INPUT_CSV = "timeseries_data.csv"

df = pd.read_csv(INPUT_CSV)
before = len(df)

vh_missing_mask = df["vh_backscatter"].isna()
removed = df[vh_missing_mask]
kept = df[~vh_missing_mask]

if len(removed) == 0:
    print("No rows with missing vh_backscatter found -- nothing to purge.")
else:
    kept.to_csv(INPUT_CSV, index=False)
    print(f"Removed {len(removed)} rows with missing vh_backscatter out of {before}.")
    print(f"{len(kept)} rows remain in {INPUT_CSV} (including any with only ndvi missing -- left untouched).")
    affected_locations = sorted(removed["location_id"].unique())
    print(f"\n{len(affected_locations)} locations affected, will be reprocessed on next run of script 04:")
    print(affected_locations)