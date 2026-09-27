"""
01_get_mining_sites.py
------------------------
Pulls 140 mining site coordinates spread across 5 countries (India,
Brazil, Australia, South Africa, Indonesia) chosen to span different
continents, climates, and mineral types -- so the model doesn't learn
region-specific quirks instead of genuine mining disturbance patterns.

IMPORTANT -- ABOUT VERIFICATION:
This script pulls coordinates from a third-party mapped dataset (Maus
et al. global mining polygons). It CANNOT independently confirm that
every point is actually a currently active mining site -- datasets
like this can contain mapping errors, closed/abandoned sites, or
polygons whose centroid falls slightly off the actual disturbed area.

To help you manually verify, this script prints a Google Maps link for
EVERY site, and saves them in the output CSV as a `maps_link` column.
Before running the full 12-month GEE extraction (script 04), open a
random sample of ~10-15 of these links yourself and confirm they show
visible mining activity (pits, roads, disturbed earth) -- catching a
bad coordinate now is far cheaper than discovering it after training.

Run this FIRST. Output feeds into 03_combine_locations.py.

Output: mining_sites.csv
    columns: location_id, latitude, longitude, label, country, maps_link
"""

import ee
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ID = "stable-matter-508305-h2"
RANDOM_SEED = 42
OUTPUT_CSV = "mining_sites.csv"

# Final 5-country list (see PROJECT_HANDOFF.md section 4 for reasoning)
COUNTRIES = {
    "IND": ("India", 45),
    "BRA": ("Brazil", 35),
    "AUS": ("Australia", 25),
    "ZAF": ("South Africa", 20),
    "IDN": ("Indonesia", 15),
}
# 45+35+25+20+15 = 140

# ---------------------------------------------------------------------------
ee.Initialize(project=PROJECT_ID)

mining = ee.FeatureCollection(
    "projects/sat-io/open-datasets/global-mining/global_mining_polygons"
)

all_rows = []

for iso3, (country_name, target_count) in COUNTRIES.items():
    country_mining = mining.filter(ee.Filter.eq("ISO3_CODE", iso3))
    available = country_mining.size().getInfo()
    print(f"{country_name} ({iso3}): {available} polygons available, "
          f"requesting {target_count}")

    if available == 0:
        print(f"  WARNING: no mining polygons found for {country_name}, skipping.")
        continue

    use_count = min(target_count, available)
    if available < target_count:
        print(f"  WARNING: only {available} available, using all of them.")

    sampled = country_mining.randomColumn("rand", seed=RANDOM_SEED) \
                             .sort("rand") \
                             .limit(use_count)

    def get_centroid(feature):
        centroid = feature.geometry().centroid()
        return feature.set({
            "lon": centroid.coordinates().get(0),
            "lat": centroid.coordinates().get(1),
        })

    sampled_with_centroids = sampled.map(get_centroid)
    features_info = sampled_with_centroids.getInfo()["features"]

    for f in features_info:
        all_rows.append({
            "latitude": f["properties"]["lat"],
            "longitude": f["properties"]["lon"],
            "country": country_name,
        })

print(f"\nTotal mining sites collected: {len(all_rows)} (target 140)")

rows = []
for i, site in enumerate(all_rows):
    lat, lon = site["latitude"], site["longitude"]
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    rows.append({
        "location_id": f"mine_{i+1:03d}",
        "latitude": lat,
        "longitude": lon,
        "label": 1,
        "country": site["country"],
        "maps_link": maps_link,
    })

mining_df = pd.DataFrame(rows)
mining_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(mining_df)} mining sites to {OUTPUT_CSV}")
print("\nCountry breakdown:")
print(mining_df["country"].value_counts())

# Print a random sample of Maps links per country for quick manual spot-checking
print("\n" + "=" * 70)
print("SPOT-CHECK THESE BEFORE RUNNING THE FULL EXTRACTION (script 04):")
print("=" * 70)
sample_for_check = mining_df.groupby("country").apply(
    lambda g: g.sample(min(3, len(g)), random_state=RANDOM_SEED)
).reset_index(drop=True)
for _, row in sample_for_check.iterrows():
    print(f"  [{row['country']}] {row['location_id']}: {row['maps_link']}")
print("=" * 70)
print("Open a few of these links -- you should see visible pits, cleared")
print("earth, or mining infrastructure, not intact forest/farmland/water.")
