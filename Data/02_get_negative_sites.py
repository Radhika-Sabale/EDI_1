"""
02_get_negative_sites.py
--------------------------
Generates non-mining ("negative") coordinates spread across the SAME
5 countries used in 01_get_mining_sites.py, sampled across MULTIPLE
land cover types (forest, cropland, grassland, built-up, bare/sparse
land) using the ESA WorldCover dataset.

WHY MULTIPLE COUNTRIES AND LAND TYPES:
- Same countries as the mining set: keeps geography/climate balanced
  between classes, so the model can't just learn "this region = mining,
  that region = not."
- Multiple land cover types: forces the model to learn the actual
  mining disturbance pattern, not just "vegetation loss vs. no loss"
  (which would also fire on logging, wildfire, or drought).

Mining exclusion zones (2km buffer around known mines) are enforced
using spatial indexing (cKDTree in 3D cartesian coordinates) against all
known mine locations in the country -- avoiding costly GEE geometric unions
that cause server timeouts.

Run this SECOND (independent of script 01, can run in parallel).
Output feeds into 03_combine_locations.py.

Output: negative_sites.csv
    columns: location_id, latitude, longitude, label, land_cover_class, country, maps_link
"""

import time
import ee
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ID = "stable-matter-508305-h2"
RANDOM_SEED = 42
MINING_EXCLUSION_BUFFER_M = 2000
OUTPUT_CSV = "negative_sites.csv"

# Final 5-country list (matches 01_get_mining_sites.py).
COUNTRIES = {
    "IND": ("India", "India", 45),
    "BRA": ("Brazil", "Brazil", 35),
    "AUS": ("Australia", "Australia", 25),
    "ZAF": ("South Africa", "South Africa", 20),
    "IDN": ("Indonesia", "Indonesia", 15),
}
# 45+35+25+20+15 = 140

LAND_COVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
}
CLASS_WEIGHTS = {10: 0.20, 20: 0.10, 30: 0.20, 40: 0.30, 50: 0.10, 60: 0.10}

def latlon_to_cartesian(lat_deg, lon_deg):
    """Converts (lat, lon) degrees to 3D Cartesian coordinates on Earth sphere in meters."""
    R = 6371000.0  # Earth radius in meters
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack([x, y, z])

# ---------------------------------------------------------------------------
ee.Initialize(project=PROJECT_ID)

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
gaul = ee.FeatureCollection("FAO/GAUL/2015/level0")
mining_all = ee.FeatureCollection(
    "projects/sat-io/open-datasets/global-mining/global_mining_polygons"
)

all_rows = []

for iso3, (country_name, gaul_name, total_count) in COUNTRIES.items():
    print(f"\n=== {country_name} (target: {total_count}) ===")
    t_start = time.time()

    country_geom_fc = gaul.filter(ee.Filter.eq("ADM0_NAME", gaul_name))
    if country_geom_fc.size().getInfo() == 0:
        print(f"  WARNING: '{gaul_name}' not found in GAUL boundaries, skipping.")
        continue

    # 1. Fetch mining locations for exclusion
    country_mines = mining_all.filter(ee.Filter.eq("ISO3_CODE", iso3))
    mine_centroids = country_mines.map(lambda f: f.centroid()).getInfo()["features"]

    mine_tree = None
    if mine_centroids:
        mine_coords = np.array([f["geometry"]["coordinates"] for f in mine_centroids])
        # mine_coords are [lon, lat]
        mine_cartesian = latlon_to_cartesian(mine_coords[:, 1], mine_coords[:, 0])
        mine_tree = cKDTree(mine_cartesian)
        print(f"  Loaded {len(mine_centroids)} mapped mines for 2km exclusion filter.")
    else:
        print(f"  No mapped mining sites found for {country_name}, skipping exclusion.")

    # 2. Determine target count per class
    class_targets = {}
    remaining = total_count
    class_items = list(CLASS_WEIGHTS.items())
    for idx, (code, weight) in enumerate(class_items):
        if idx == len(class_items) - 1:
            class_targets[code] = remaining
        else:
            n = round(total_count * weight)
            class_targets[code] = n
            remaining -= n

    # 3. Stratified sampling across land cover classes within country boundaries
    bounds = country_geom_fc.geometry().bounds()
    country_mask = ee.Image().byte().paint(country_geom_fc, 1)
    masked_wc = worldcover.updateMask(country_mask)

    selected_by_class = {code: [] for code in class_targets}
    seed_offset = 0

    while any(len(selected_by_class[c]) < class_targets[c] for c in class_targets) and seed_offset < 3:
        samples = masked_wc.stratifiedSample(
            numPoints=20,
            classBand="Map",
            region=bounds,
            scale=150,
            geometries=True,
            seed=RANDOM_SEED + seed_offset,
        )
        features = samples.getInfo()["features"]

        for f in features:
            code = f["properties"]["Map"]
            if code not in selected_by_class:
                continue
            if len(selected_by_class[code]) >= class_targets[code]:
                continue

            lon, lat = f["geometry"]["coordinates"]

            # Exclude points within 2km of any known mine in the country
            if mine_tree is not None:
                pt_cart = latlon_to_cartesian(np.array([lat]), np.array([lon]))
                nearby = mine_tree.query_ball_point(pt_cart[0], r=MINING_EXCLUSION_BUFFER_M)
                if len(nearby) > 0:
                    continue

            selected_by_class[code].append((lat, lon))

        seed_offset += 1

    for code, target in class_targets.items():
        class_name = LAND_COVER_CLASSES[code]
        pts = selected_by_class[code][:target]
        print(f"  {class_name}: {len(pts)}/{target} found")
        for lat, lon in pts:
            all_rows.append({
                "latitude": lat,
                "longitude": lon,
                "land_cover_class": class_name,
                "country": country_name,
            })

    print(f"  Total collected for {country_name}: {sum(len(selected_by_class[c][:class_targets[c]]) for c in class_targets)}/{total_count} (took {time.time()-t_start:.1f}s)")

print(f"\nTotal negative sites collected: {len(all_rows)} (target 140)")

rows = []
for i, site in enumerate(all_rows):
    lat, lon = site["latitude"], site["longitude"]
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    rows.append({
        "location_id": f"neg_{i+1:03d}",
        "latitude": lat,
        "longitude": lon,
        "label": 0,
        "land_cover_class": site["land_cover_class"],
        "country": site["country"],
        "maps_link": maps_link,
    })

negative_df = pd.DataFrame(rows)
negative_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(negative_df)} non-mining sites to {OUTPUT_CSV}")
print("\nCountry breakdown:")
print(negative_df["country"].value_counts())
print("\nLand cover breakdown:")
print(negative_df["land_cover_class"].value_counts())