"""
04_gee_extract_timeseries.py
------------------------------
Extracts a 12-month time series of monthly NDVI (Sentinel-2) and
VH backscatter (Sentinel-1) for every location in input_locations.csv.

WHY THE ORIGINAL APPROACH TIMES OUT:
Pulling one reduceRegion + getInfo() per location per month means
280 locations x 12 months = 3,360 separate round-trips to GEE's
synchronous API, which has a hard per-request compute timeout. Doing
that in a tight Python loop is exactly the pattern that stalls out.

WHAT THIS SCRIPT DOES INSTEAD:
1. Splits the 280 locations into small batches (BATCH_SIZE each).
2. For each (batch, month) pair, builds ONE server-side monthly
   composite image and reduces it over the WHOLE batch in a single
   reduceRegions() call -- not one call per point. This cuts total
   getInfo() calls from 3,360 down to (280/BATCH_SIZE) x 12.
3. Writes results to disk immediately after every batch/month, so a
   crash or timeout partway through never loses progress already
   pulled -- and the script can simply be re-run; it detects and
   skips (location, month) pairs already saved.
4. Retries each GEE call with exponential backoff before giving up
   on a batch/month (and logs it so you know exactly what to re-run).

Run this FOURTH, after 03_combine_locations.py.
Output: timeseries_data.csv (long format)
    columns: location_id, year_month, ndvi, vh_backscatter, label
"""

import time
import os
import ee
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- adjust these to match your actual study window
# ---------------------------------------------------------------------------
PROJECT_ID = "stable-matter-508305-h2"
INPUT_CSV = "input_locations.csv"
OUTPUT_CSV = "timeseries_data.csv"

BATCH_SIZE = 20            # locations per batch -- lower this further if you still see timeouts
START_DATE = "2024-01-01"  # <-- SET THIS to the actual start of your 12-month window
NUM_MONTHS = 12
SCALE = 10                 # meters/pixel -- native S1/S2 resolution
BUFFER_M = 30              # radius (meters) averaged around each point -- suppresses SAR speckle
                            # and absorbs centroid/geolocation slop. Narrow it if your sites are small.
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 5       # backoff doubles each retry: 5s, 10s, 20s, 40s
SLEEP_BETWEEN_CALLS_SEC = 1  # small pause to stay comfortably under GEE rate limits

ee.Initialize(project=PROJECT_ID)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def month_ranges(start_date_str, n_months):
    """Yield (year_month_label, ee.Date start, ee.Date end) for n consecutive months."""
    start = ee.Date(start_date_str)
    for i in range(n_months):
        m_start = start.advance(i, "month")
        m_end = m_start.advance(1, "month")
        label = m_start.format("YYYY-MM").getInfo()
        yield label, m_start, m_end


def mask_s2_clouds(image):
    """Cloud-mask Sentinel-2 SR using the SCL band -- keeps clear veg/soil/water/unclassified/snow."""
    scl = image.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(good)


def get_monthly_ndvi(m_start, m_end):
    """Cloud-masked monthly median NDVI composite (Sentinel-2 SR harmonized)."""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(m_start, m_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(mask_s2_clouds)
    )
    composite = s2.median()
    return composite.normalizedDifference(["B8", "B4"]).rename("ndvi")


def get_monthly_vh(m_start, m_end):
    """Monthly mean SAR backscatter (Sentinel-1 GRD, IW mode) -- prefers VH, falls
    back to VV where VH is unavailable.

    Sentinel-1's acquisition plan does not guarantee dual-polarization (VH) coverage
    everywhere -- it's denser over Europe and sparser in many other regions (notably
    much of inland Australia), independent of cloud cover or anything in this script.
    Requiring VH strictly leaves real, systematic gaps concentrated in those regions.
    VV is far more consistently acquired, so pixels/months where VH is masked or
    missing are filled with VV instead of being left null.
    """
    s1_all = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(m_start, m_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )
    vh = (
        s1_all.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select("VH")
        .mean()
        .rename("vh_backscatter")
    )
    vv = (
        s1_all.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
        .mean()
        .rename("vh_backscatter")
    )
    # unmask() fills only the pixels/regions where vh has no data, using vv there instead
    return vh.unmask(vv)


def call_with_retry(fn, *args, **kwargs):
    """Call a GEE-triggering function, retrying transient failures with exponential backoff."""
    delay = RETRY_BACKOFF_SEC
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
    raise last_err


def extract_batch_month(batch_fc, ndvi_img, vh_img):
    """Server-side: reduce NDVI + VH over every buffered point in the batch in ONE call.

    Averaging over BUFFER_M rather than sampling a single pixel suppresses Sentinel-1
    speckle noise and absorbs small geolocation/grid-alignment slop between the mining
    centroid coordinates and the S1/S2 pixel grids.
    """
    combined = ndvi_img.addBands(vh_img)
    reduced = combined.reduceRegions(
        collection=batch_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE,
    )
    return reduced.getInfo()["features"]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    locations_df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(locations_df)} locations from {INPUT_CSV}")

    # Resume support: skip (location_id, year_month) pairs already saved from a prior run
    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_pairs = set(zip(done_df["location_id"], done_df["year_month"]))
        print(f"Found existing {OUTPUT_CSV} with {len(done_df)} rows -- resuming, will skip completed pairs.")
    else:
        done_pairs = set()
        pd.DataFrame(columns=["location_id", "year_month", "ndvi", "vh_backscatter", "label"]).to_csv(
            OUTPUT_CSV, index=False
        )

    batches = [
        locations_df.iloc[i : i + BATCH_SIZE] for i in range(0, len(locations_df), BATCH_SIZE)
    ]
    print(f"Split into {len(batches)} batches of up to {BATCH_SIZE} locations each.")

    months = list(month_ranges(START_DATE, NUM_MONTHS))
    print(f"Extracting {len(months)} months: {months[0][0]} to {months[-1][0]}")

    total_calls = len(batches) * len(months)
    call_num = 0

    for batch_idx, batch_df in enumerate(batches):
        # Build the ee.FeatureCollection once per batch, reused across all 12 months.
        # Each point is buffered to a small disc so reduceRegions averages over an
        # area (suppresses SAR speckle) instead of sampling one raw pixel.
        ee_points = [
            ee.Feature(
                ee.Geometry.Point([row.longitude, row.latitude]).buffer(BUFFER_M),
                {"location_id": row.location_id, "label": int(row.label)},
            )
            for row in batch_df.itertuples()
        ]
        batch_fc = ee.FeatureCollection(ee_points)
        batch_ids = set(batch_df["location_id"])

        for month_label, m_start, m_end in months:
            call_num += 1

            done_this_month = {lid for lid, ym in done_pairs if ym == month_label}
            if batch_ids.issubset(done_this_month):
                print(f"[{call_num}/{total_calls}] Batch {batch_idx+1}, {month_label}: already done, skipping.")
                continue

            print(f"[{call_num}/{total_calls}] Batch {batch_idx+1}/{len(batches)} "
                  f"({len(batch_df)} locations), month {month_label}...")

            ndvi_img = get_monthly_ndvi(m_start, m_end)
            vh_img = get_monthly_vh(m_start, m_end)

            try:
                features = call_with_retry(extract_batch_month, batch_fc, ndvi_img, vh_img)
            except Exception as e:
                print(f"    GAVE UP on batch {batch_idx+1}, {month_label} after {MAX_RETRIES} attempts: {e}")
                print(f"    Skipping -- rerun the script later to fill just this gap.")
                continue

            rows = []
            for f in features:
                props = f["properties"]
                loc_id = props.get("location_id")
                # Only append pairs not already saved -- a partially-done batch (e.g.
                # after purge_incomplete_rows.py cleared just a few locations) still
                # triggers a full-batch GEE call, but this stops already-complete
                # locations in that same batch from getting duplicate rows appended.
                if (loc_id, month_label) in done_pairs:
                    continue
                rows.append({
                    "location_id": loc_id,
                    "year_month": month_label,
                    "ndvi": props.get("ndvi"),
                    "vh_backscatter": props.get("vh_backscatter"),
                    "label": props.get("label"),
                })

            if not rows:
                print(f"    All locations in this batch already had {month_label} -- nothing new to append.")
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                continue

            # Append immediately -- progress is never lost even if the NEXT call times out
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            done_pairs.update((r["location_id"], r["year_month"]) for r in rows)

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print(f"\nDone. Results in {OUTPUT_CSV}")
    final_df = pd.read_csv(OUTPUT_CSV)
    expected = len(locations_df) * len(months)
    print(f"Total rows: {len(final_df)} (expected up to {expected})")
    missing = expected - len(final_df)
    if missing > 0:
        print(f"WARNING: {missing} (location, month) pairs are missing -- rerun this script to fill gaps.")


if __name__ == "__main__":
    main()