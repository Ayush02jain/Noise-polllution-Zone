"""
Multi-City Noise Pollution — Merge, Geocode & JSON Export
==========================================================
Merges Delhi (41 locations) + Chennai (10 locations) datasets.
Geocodes all 51 locations and exports all_cities_locations_geo.json.
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import json
import os

BASE_DIR = r"d:\Noise polllution Zone"
# # os.chdir(BASE_DIR)

print("=" * 70)
print("  MULTI-CITY NOISE POLLUTION — MERGE & GEOCODE")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD & MERGE
# ============================================================================
print("\n  Loading datasets...")

df_delhi = pd.read_csv('../../data/processed/delhi_noise_2020_2024.csv')
if 'City' not in df_delhi.columns:
    df_delhi['City'] = 'Delhi'
print(f"    Delhi:   {df_delhi.shape[0]} rows, {df_delhi['Location'].nunique()} locations")

df_chennai = pd.read_csv('../../data/processed/chennai_noise_2020_2024.csv')
print(f"    Chennai: {df_chennai.shape[0]} rows, {df_chennai['Location'].nunique()} locations")

# Keep only common columns
common_cols = ['S_No', 'Location', 'City', 'Zone_Type', 'Year', 'Month',
               'Month_Name', 'Noise_Day_dB', 'Noise_Night_dB', 'Zone_Category']
df_all = pd.concat([df_delhi[common_cols], df_chennai[common_cols]], ignore_index=True)
print(f"    Merged:  {df_all.shape[0]} rows, {df_all['Location'].nunique()} locations")

# Per-location overall aggregates
loc_overall = df_all.groupby(['Location', 'City']).agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

# Per-location+year aggregates
loc_yearly = df_all.groupby(['Location', 'City', 'Year']).agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0])
).reset_index()

# Per-city yearly trend
city_yearly = df_all.groupby(['City', 'Year'])[['Noise_Day_dB', 'Noise_Night_dB']].mean().reset_index()

print(f"    Locations: {len(loc_overall)} ({(loc_overall['City']=='Delhi').sum()} Delhi, "
      f"{(loc_overall['City']=='Chennai').sum()} Chennai)")

# ============================================================================
# STEP 2: GEOCODING
# ============================================================================
print("\n  Geocoding 51 locations...")

MANUAL_COORDS = {
    # Delhi Fallback
    "Adarsh Nagar": (28.7041, 77.1756), "Anand Vihar": (28.6469, 77.3152),
    "Ashok Vihar": (28.6921, 77.1840), "Karol Bagh": (28.6519, 77.1909),
    "Rohini": (28.7495, 77.0672), "Pahar Ganj": (28.6448, 77.2167),
    "Lajpat Nagar": (28.5700, 77.2435), "Dwaraka": (28.5921, 77.0460),
    "Vasant Kunj": (28.5244, 77.1565), "Sarita Vihar": (28.5318, 77.2907),
    "Mehrauli": (28.5245, 77.1855), "Tughlakabad": (28.4780, 77.2570),
    "Janak Puri": (28.6280, 77.0840), "Tilak Nagar": (28.6405, 77.0940),
    "Shalimar Bagh": (28.7166, 77.1613), "Pitampura": (28.7006, 77.1306),
    "Badli": (28.7478, 77.1540), "Mangol Puri": (28.7001, 77.0869),
    "Moti Nagar": (28.6567, 77.1380), "Patel Nagar": (28.6538, 77.1629),
    # Chennai Fallback
    "T Nagar": (13.0358, 80.2333), "Perambur": (13.1116, 80.2329),
    "Guindy": (13.0067, 80.2206), "Triplicane": (13.0569, 80.2762),
    "Velachery": (12.9815, 80.2180), "Washermanpet": (13.1155, 80.2874),
    "Pallikaranai": (12.9370, 80.2131), "Anna Nagar": (13.0850, 80.2101),
    "Sowcarpet": (13.0916, 80.2784), "Egmore Eye Hospital": (13.0732, 80.2609),
}

CITY_DEFAULTS = {'Delhi': (28.6139, 77.2090), 'Chennai': (13.0827, 80.2707)}
CITY_BOUNDS = {
    'Delhi': (28.4, 28.9, 76.8, 77.5),
    'Chennai': (12.8, 13.3, 80.0, 80.4)
}

geolocator = Nominatim(user_agent="multi_city_noise_map", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

latitudes, longitudes = [], []
stats = {'Geocoded': 0, 'Manual': 0, 'Default': 0}

for _, row in loc_overall.iterrows():
    loc = row['Location'].strip()
    city = row['City']
    bounds = CITY_BOUNDS.get(city, (0, 90, 0, 180))

    try:
        result = geocode(f"{loc}, {city}, India")
        if result and bounds[0] < result.latitude < bounds[1] and bounds[2] < result.longitude < bounds[3]:
            latitudes.append(round(result.latitude, 6))
            longitudes.append(round(result.longitude, 6))
            stats['Geocoded'] += 1
            print(f"    [{city:7s}] {loc}: ({result.latitude:.4f}, {result.longitude:.4f}) [Geocoded]")
            continue
    except Exception:
        pass

    if loc in MANUAL_COORDS:
        lat, lon = MANUAL_COORDS[loc]
        latitudes.append(lat)
        longitudes.append(lon)
        stats['Manual'] += 1
        print(f"    [{city:7s}] {loc}: ({lat:.4f}, {lon:.4f}) [Manual]")
    else:
        lat, lon = CITY_DEFAULTS.get(city, (20.5, 79.0))
        latitudes.append(lat)
        longitudes.append(lon)
        stats['Default'] += 1
        print(f"    [{city:7s}] {loc}: ({lat:.4f}, {lon:.4f}) [Default]")

loc_overall['Latitude'] = latitudes
loc_overall['Longitude'] = longitudes

# Step 3: Check for null/zero coordinates
failed_geocoding = loc_overall[(loc_overall['Latitude'].isnull()) | (loc_overall['Latitude'] == 0)]
if not failed_geocoding.empty:
    print("\n  WARNING: The following locations have NULL or 0 coordinates:")
    print(failed_geocoding[['Location', 'City']])
else:
    print("\n  SUCCESS: All locations have valid coordinates.")

print(f"\n  Geocoding: {stats['Geocoded']} geocoded, {stats['Manual']} manual, {stats['Default']} default")

# ============================================================================
# BUILD JSON
# ============================================================================
print("\n  Building JSON...")

geo_records = []
for _, ov in loc_overall.iterrows():
    loc = ov['Location']
    city = ov['City']
    yearly_rows = loc_yearly[(loc_yearly['Location'] == loc) & (loc_yearly['City'] == city)]
    yearly_clean = [{'Year': int(yr['Year']), 'Avg_Day': round(yr['Avg_Day'], 2),
                     'Avg_Night': round(yr['Avg_Night'], 2),
                     'Zone_Category': yr['Zone_Category']}
                    for _, yr in yearly_rows.iterrows()]
    geo_records.append({
        'Location': loc,
        'City': city,
        'Latitude': ov['Latitude'],
        'Longitude': ov['Longitude'],
        'Zone_Type': ov['Zone_Type'],
        'Avg_Day': round(ov['Avg_Day'], 2),
        'Avg_Night': round(ov['Avg_Night'], 2),
        'Zone_Category': ov['Zone_Category'],
        'Yearly': yearly_clean
    })

# City-level yearly trends
trends = {}
for _, tr in city_yearly.iterrows():
    city = tr['City']
    if city not in trends:
        trends[city] = []
    trends[city].append({
        'Year': int(tr['Year']),
        'Avg_Day': round(tr['Noise_Day_dB'], 2),
        'Avg_Night': round(tr['Noise_Night_dB'], 2)
    })

output = {
    'locations': geo_records,
    'trends': trends
}

with open('../../data/geo/all_cities_locations_geo.json', 'w') as f:
    json.dump(output, f, indent=2)

fsize = os.path.getsize('../../data/geo/all_cities_locations_geo.json')
print(f"  Saved: all_cities_locations_geo.json ({fsize:,} bytes, {len(geo_records)} locations)")

print("\n" + "=" * 70)
print("  DONE! Now serve frontend with: python -m http.server 8080")
print("  Open: http://localhost:8080/index.html")
print("=" * 70)
