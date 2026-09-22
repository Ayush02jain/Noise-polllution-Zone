"""
Multi-City Noise Pollution — Merge & JSON Export
==================================================
Merges Delhi (41), Chennai (10), Mumbai (18) = 69 total locations
Includes 2020-2024 historical data + 2025 ML forecasts.
Exports all_cities_locations_geo.json to data/geo/, frontend/, and public/.
"""

import pandas as pd
import json
import shutil
import os

print("=" * 70)
print("  MULTI-CITY NOISE POLLUTION — MERGE & JSON EXPORT")
print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

GEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'geo')
delhi_geo_file   = os.path.join(GEO_DIR, 'delhi_locations_geo.json')
chennai_geo_file = os.path.join(GEO_DIR, 'chennai_locations_geo.json')
mumbai_geo_file  = os.path.join(GEO_DIR, 'mumbai_locations_geo.json')

all_locations = []

for city, path in [('Delhi', delhi_geo_file), ('Chennai', chennai_geo_file), ('Mumbai', mumbai_geo_file)]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            locs = data.get('locations', [])
            all_locations.extend(locs)
            print(f"  Loaded {len(locs)} locations from {city} ({os.path.basename(path)})")
    else:
        print(f"  WARNING: {path} not found!")

print(f"\n  Total Merged Locations: {len(all_locations)}")

output = {'locations': all_locations}

geo_out = os.path.join(PROJECT_ROOT, 'data', 'geo', 'all_cities_locations_geo.json')
fe_out  = os.path.join(PROJECT_ROOT, 'frontend', 'all_cities_locations_geo.json')
pub_out = os.path.join(PROJECT_ROOT, 'public', 'all_cities_locations_geo.json')

with open(geo_out, 'w') as f:
    json.dump(output, f, indent=2)

shutil.copy(geo_out, fe_out)

if os.path.isdir(os.path.dirname(pub_out)):
    shutil.copy(geo_out, pub_out)

fsize = os.path.getsize(fe_out)
print(f"  Saved: {fe_out} ({fsize:,} bytes)")
print("=" * 70)
print("  MERGE COMPLETE!")
print("=" * 70)
