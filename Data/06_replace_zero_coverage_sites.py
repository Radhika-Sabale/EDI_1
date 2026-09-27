"""
06_replace_zero_coverage_sites.py
------------------------------------
Finds and swaps in replacement coordinates for any location confirmed
to have ZERO Sentinel-1 imagery of any polarization (see
diagnose_vh_gap.py) -- a real satellite-coverage gap that no amount of
code fixing in 04_gee_extract_timeseries.py can solve.

WHAT IT DOES:
1. Reads cleaning_report.csv, pulls out every location dropped for a
   vh_backscatter reason (these are the confirmed/likely zero-coverage
   sites).
2. For each one, searches for a replacement:
   - Mining sites: draws new candidate points from the same Maus et
     al. global mining polygon dataset, same country, excluding all
     coordinates already in use.
   - Negative sites: draws new candidate points from the same
     ESA WorldCover land-cover class, same country, respecting the
     2km mining-exclusion buffer (mirrors 02_get_negative_sites.py).
3. Each candidate is checked against live Sentinel-1 availability --
   a quick 3-month spot-check first (fast, filters out most bad
   candidates cheaply), then a full 12-month check on anything that
   passes, before it's accepted. Only candidates with imagery in
   EVERY month of the study window are accepted.
4. Updates mining_sites.csv / negative_sites.csv / input_locations.csv
   with the new coordinates for the replaced location_ids (location_id
   is KEPT THE SAME so downstream scripts don't need to change).
   Originals are backed up first.
5. Removes any existing rows for the replaced location_ids from
   timeseries_data.csv, since their old coordinates' data is no longer
   relevant -- so 04_gee_extract_timeseries.py will naturally refetch
   them under the new coordinates on its next run.
6. Prints a before/after table with Maps links for the new coordinates
   so you can eyeball them the same way you did in script 01, before
   trusting them.

Run this AFTER seeing which locations are dropped for a vh_backscatter
reason in cleaning_report.csv. Then rerun 04 and 05 as usual.
"""

import time
import random
import ee
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ID = "stable-matter-508305-h2"
CLEANING_REPORT_CSV = "cleaning_report.csv"
MINING_CSV = "mining_sites.csv"
NEGATIVE_CSV = "negative_sites.csv"
INPUT_LOCATIONS_CSV = "input_locations.csv"
TIMESERIES_CSV = "timeseries_data.csv"

STUDY_START = "2024-01-01"
NUM_MONTHS = 12
QUICK_CHECK_MONTHS = [0, 5, 10]   # month indices (0-based) for the fast pre-filter
MAX_CANDIDATES_PER_SITE = 15
MINING_EXCLUSION_BUFFER_M = 2000
RANDOM_SEED = 123  # different from the original 42, so we don't redraw the same points

ee.Initialize(project=PROJECT_ID)
random.seed(RANDOM_SEED)

mining_polygons = ee.FeatureCollection(
    "projects/sat-io/open-datasets/global-mining/global_mining_polygons"
)
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
gaul = ee.FeatureCollection("FAO/GAUL/2015/level0")

GAUL_NAME_BY_ISO3 = {
    "IND": "India", "BRA": "Brazil", "AUS": "Australia",
    "ZAF": "South Africa", "IDN": "Indonesia",
}
LAND_COVER_CODE_BY_NAME = {
    "tree_cover": 10, "shrubland": 20, "grassland": 30,
    "cropland": 40, "built_up": 50, "bare_sparse_vegetation": 60,
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def month_bounds(i):
    start = ee.Date(STUDY_START).advance(i, "month")
    return start, start.advance(1, "month")


def has_s1_coverage(lat, lon, month_indices):
    """True only if Sentinel-1 IW imagery (any polarization) exists in EVERY given month."""
    point = ee.Geometry.Point([lon, lat]).buffer(30)
    for i in month_indices:
        m_start, m_end = month_bounds(i)
        count = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(point)
            .filterDate(m_start, m_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .size()
            .getInfo()
        )
        if count == 0:
            return False
    return True


def find_mining_replacement(country_iso3, excluded_coords):
    country_mining = mining_polygons.filter(ee.Filter.eq("ISO3_CODE", country_iso3))
    available = country_mining.size().getInfo()
    if available == 0:
        return None

    sample_size = min(available, MAX_CANDIDATES_PER_SITE * 3)
    seed = RANDOM_SEED + random.randint(0, 100000)
    sampled = (
        country_mining.randomColumn("rand", seed=seed).sort("rand").limit(sample_size)
    )

    def get_centroid(feature):
        c = feature.geometry().centroid()
        return feature.set({"lon": c.coordinates().get(0), "lat": c.coordinates().get(1)})

    features_info = sampled.map(get_centroid).getInfo()["features"]

    tried = 0
    for f in features_info:
        if tried >= MAX_CANDIDATES_PER_SITE:
            break
        lat, lon = f["properties"]["lat"], f["properties"]["lon"]
        if (round(lat, 4), round(lon, 4)) in excluded_coords:
            continue
        tried += 1
        if not has_s1_coverage(lat, lon, QUICK_CHECK_MONTHS):
            continue
        if not has_s1_coverage(lat, lon, list(range(NUM_MONTHS))):
            continue
        return lat, lon
    return None


def find_negative_replacement(country_iso3, land_cover_class, mine_tree_coords, excluded_coords):
    import numpy as np
    from scipy.spatial import cKDTree

    def latlon_to_cartesian(lat_deg, lon_deg):
        R = 6371000.0
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)
        return np.column_stack([x, y, z])

    mine_tree = None
    if mine_tree_coords:
        mine_cart = latlon_to_cartesian(
            np.array([c[0] for c in mine_tree_coords]),
            np.array([c[1] for c in mine_tree_coords]),
        )
        mine_tree = cKDTree(mine_cart)

    gaul_name = GAUL_NAME_BY_ISO3[country_iso3]
    country_geom_fc = gaul.filter(ee.Filter.eq("ADM0_NAME", gaul_name))
    bounds = country_geom_fc.geometry().bounds()
    country_mask = ee.Image().byte().paint(country_geom_fc, 1)
    masked_wc = worldcover.updateMask(country_mask)
    code = LAND_COVER_CODE_BY_NAME[land_cover_class]

    tried = 0
    seed_offset = 0
    while tried < MAX_CANDIDATES_PER_SITE and seed_offset < 6:
        samples = masked_wc.stratifiedSample(
            numPoints=10, classBand="Map", region=bounds, scale=150,
            geometries=True, seed=RANDOM_SEED + random.randint(0, 100000) + seed_offset,
        )
        features = samples.getInfo()["features"]
        seed_offset += 1

        for f in features:
            if f["properties"]["Map"] != code:
                continue
            if tried >= MAX_CANDIDATES_PER_SITE:
                break
            lon, lat = f["geometry"]["coordinates"]
            if (round(lat, 4), round(lon, 4)) in excluded_coords:
                continue
            if mine_tree is not None:
                pt_cart = latlon_to_cartesian(np.array([lat]), np.array([lon]))
                nearby = mine_tree.query_ball_point(pt_cart[0], r=MINING_EXCLUSION_BUFFER_M)
                if len(nearby) > 0:
                    continue
            tried += 1
            if not has_s1_coverage(lat, lon, QUICK_CHECK_MONTHS):
                continue
            if not has_s1_coverage(lat, lon, list(range(NUM_MONTHS))):
                continue
            return lat, lon
    return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    report = pd.read_csv(CLEANING_REPORT_CSV)
    to_replace = report[
        report["reason"].astype(str).str.startswith("vh_backscatter missing")
    ]["location_id"].tolist()

    if not to_replace:
        print("No locations flagged for a vh_backscatter coverage gap -- nothing to replace.")
        return

    print(f"{len(to_replace)} locations need replacement due to zero Sentinel-1 coverage:")
    print(to_replace)

    mining_df = pd.read_csv(MINING_CSV)
    negative_df = pd.read_csv(NEGATIVE_CSV)
    input_df = pd.read_csv(INPUT_LOCATIONS_CSV)

    mining_df.to_csv(MINING_CSV + ".bak", index=False)
    negative_df.to_csv(NEGATIVE_CSV + ".bak", index=False)
    input_df.to_csv(INPUT_LOCATIONS_CSV + ".bak", index=False)

    excluded_coords = set(
        zip(mining_df["latitude"].round(4), mining_df["longitude"].round(4))
    ) | set(
        zip(negative_df["latitude"].round(4), negative_df["longitude"].round(4))
    )
    mine_coords = list(zip(mining_df["latitude"], mining_df["longitude"]))

    results = []

    for loc_id in to_replace:
        is_mining = loc_id.startswith("mine_")
        source_df = mining_df if is_mining else negative_df
        row = source_df[source_df["location_id"] == loc_id]
        if row.empty:
            print(f"  {loc_id}: not found in source CSV, skipping.")
            continue
        old_lat, old_lon = row.iloc[0]["latitude"], row.iloc[0]["longitude"]
        country_name = row.iloc[0]["country"]
        iso3 = {v: k for k, v in GAUL_NAME_BY_ISO3.items()}.get(country_name)
        if iso3 is None:
            print(f"  {loc_id}: unrecognized country '{country_name}', skipping.")
            continue

        print(f"\nSearching replacement for {loc_id} ({country_name})...")

        if is_mining:
            found = find_mining_replacement(iso3, excluded_coords)
        else:
            land_cover_class = row.iloc[0]["land_cover_class"]
            found = find_negative_replacement(iso3, land_cover_class, mine_coords, excluded_coords)

        if found is None:
            print(f"  FAILED to find a valid replacement for {loc_id} after "
                  f"{MAX_CANDIDATES_PER_SITE} candidates. Leave it dropped for now, "
                  f"or rerun with a higher MAX_CANDIDATES_PER_SITE.")
            continue

        new_lat, new_lon = found
        excluded_coords.add((round(new_lat, 4), round(new_lon, 4)))
        if is_mining:
            mine_coords.append((new_lat, new_lon))

        maps_link = f"https://www.google.com/maps?q={new_lat},{new_lon}"
        print(f"  Found replacement: ({new_lat:.4f}, {new_lon:.4f}) -- {maps_link}")

        source_df.loc[source_df["location_id"] == loc_id, ["latitude", "longitude", "maps_link"]] = [
            new_lat, new_lon, maps_link,
        ]
        input_df.loc[input_df["location_id"] == loc_id, ["latitude", "longitude"]] = [
            new_lat, new_lon,
        ]

        results.append({
            "location_id": loc_id, "country": country_name,
            "old_lat": old_lat, "old_lon": old_lon,
            "new_lat": new_lat, "new_lon": new_lon, "maps_link": maps_link,
        })

        time.sleep(1)

    mining_df.to_csv(MINING_CSV, index=False)
    negative_df.to_csv(NEGATIVE_CSV, index=False)
    input_df.to_csv(INPUT_LOCATIONS_CSV, index=False)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("replacement_sites.csv", index=False)
        print(f"\n{len(results)} locations replaced. Details saved to replacement_sites.csv")
        print("SPOT-CHECK a few of these Maps links before trusting them, same as script 01.")

        replaced_ids = results_df["location_id"].tolist()
        if TIMESERIES_CSV:
            try:
                ts_df = pd.read_csv(TIMESERIES_CSV)
                before = len(ts_df)
                ts_df = ts_df[~ts_df["location_id"].isin(replaced_ids)]
                ts_df.to_csv(TIMESERIES_CSV, index=False)
                print(f"Removed {before - len(ts_df)} stale rows from {TIMESERIES_CSV} "
                      f"for the replaced locations -- rerun 04 to refetch them fresh.")
            except FileNotFoundError:
                pass
    else:
        print("\nNo replacements were successfully found.")


if __name__ == "__main__":
    main()