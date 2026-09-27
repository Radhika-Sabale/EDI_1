"""
diagnose_vh_gap.py
---------------------
Checks whether Sentinel-1 genuinely has ZERO imagery (any polarization)
for a handful of the locations that are still showing missing VH after
the VV-fallback fix in 04_gee_extract_timeseries.py.

If both vh_count and vv_count come back 0 for a location/month, that
confirms there's no Sentinel-1 pass over that point at all that month
-- a real coverage gap, not a bug in the fallback logic. If either
count is > 0, something in the extraction script itself needs another
look, since data does exist and should have been picked up.

Edit CHECK_LOCATIONS below to test specific location_ids.
"""

import ee
import pandas as pd

PROJECT_ID = "stable-matter-508305-h2"
INPUT_CSV = "input_locations.csv"
CHECK_LOCATIONS = ["mine_083", "mine_096", "mine_101", "neg_046", "neg_070"]
CHECK_MONTHS = ["2024-01", "2024-06"]  # spot-check a couple months per location

ee.Initialize(project=PROJECT_ID)

locations_df = pd.read_csv(INPUT_CSV)

for loc_id in CHECK_LOCATIONS:
    row = locations_df[locations_df["location_id"] == loc_id]
    if row.empty:
        print(f"{loc_id}: not found in {INPUT_CSV}")
        continue
    lat = row.iloc[0]["latitude"]
    lon = row.iloc[0]["longitude"]
    print(f"\n{loc_id} ({lat:.4f}, {lon:.4f}):")

    point = ee.Geometry.Point([lon, lat]).buffer(30)

    for month_label in CHECK_MONTHS:
        m_start = ee.Date(month_label + "-01")
        m_end = m_start.advance(1, "month")

        s1_all = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(point)
            .filterDate(m_start, m_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        total_count = s1_all.size().getInfo()

        vh_count = s1_all.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VH")
        ).size().getInfo()
        vv_count = s1_all.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
        ).size().getInfo()

        print(f"  {month_label}: total S1 IW images={total_count}, "
              f"with VH={vh_count}, with VV={vv_count}")