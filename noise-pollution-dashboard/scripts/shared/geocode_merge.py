"""
Multi-City Noise Pollution — Merge & JSON Export
==================================================
Merges Delhi (41), Chennai (10), Mumbai (18) = 69 total locations.
Uses hardcoded coordinates — no geocoding API calls.
"""

import pandas as pd
import json
import shutil
import os

print("=" * 70)
print("  MULTI-CITY NOISE POLLUTION — MERGE & JSON EXPORT")
print("=" * 70)

# ============================================================================
# HARDCODED COORDINATES
# ============================================================================
DELHI_COORDS = {
    'Adarsh Nagar': (28.7165, 77.1709), 'Anand Vihar': (28.6469, 77.3156),
    'Ashok Vihar': (28.6957, 77.1770), 'Badli': (28.7353, 77.1331),
    'Braham Puri': (28.6750, 77.2700), 'Daryaganj': (28.6411, 77.2388),
    'Defence Colony': (28.5744, 77.2337), 'Dwaraka': (28.5921, 77.0460),
    'Greater Kailash': (28.5420, 77.2400), 'Inder Puri': (28.5960, 77.1770),
    'Janak Puri': (28.6219, 77.0815), 'Karawal Nagar': (28.7230, 77.2590),
    'Karol Bagh': (28.6514, 77.1907), 'Kondli': (28.6200, 77.3500),
    'Lajpat Nagar': (28.5700, 77.2400), 'Lawrence Road': (28.6800, 77.1300),
    'Mandavali': (28.6364, 77.2953), 'Mangol Puri': (28.7050, 77.1300),
    'Meera Bagh': (28.6700, 77.1100), 'Mehrauli': (28.5150, 77.1800),
    'Moti Bagh': (28.5800, 77.1700), 'Moti Nagar': (28.6531, 77.1453),
    'Mukherji Nagar': (28.7073, 77.2100), 'Nand Nagri': (28.6944, 77.3112),
    'Naraouji Nagar': (28.5900, 77.1800), 'New Friends Colony': (28.5636, 77.2634),
    'Pahar Ganj': (28.6441, 77.2132), 'Paschim Vihar': (28.6700, 77.1000),
    'Patel Nagar': (28.6508, 77.1657), 'Prehladpur': (28.5400, 77.3000),
    'R.K. Puram': (28.5700, 77.1700), 'Rajpura Road': (28.7100, 77.2300),
    'Rana Pratap Bagh': (28.6900, 77.2000), 'Rohini': (28.7320, 77.1100),
    'Sarita Vihar': (28.5310, 77.2880), 'Shalimar Bagh': (28.7184, 77.1600),
    'Shanti Vihar': (28.6100, 77.3100), 'Tilak Nagar': (28.6400, 77.0900),
    'Tughlakabad': (28.5147, 77.2530), 'Vasant Kunj': (28.5210, 77.1570),
    'Yamuna Vihar': (28.6970, 77.2720),
}

CHENNAI_COORDS = {
    'Guindy':               (13.0067, 80.2206),
    'Perambur':             (13.1116, 80.2329),
    'T Nagar':              (13.0358, 80.2333),
    'Triplicane':           (13.0569, 80.2762),
    'Pallikaranai':         (12.9370, 80.2131),
    'Velachery':            (12.9815, 80.2180),
    'Washermanpet':         (13.1155, 80.2874),
    'Anna Nagar':           (13.0850, 80.2101),
    'Sowcarpet':            (13.0916, 80.2784),
    'Egmore Eye Hospital':  (13.0732, 80.2609),
}

MUMBAI_COORDS = {
    'Santacruz (W)':        (19.0822, 72.8397),
    'Vile Parle (W)':       (19.1004, 72.8497),
    'Andheri (W)':          (19.1197, 72.8464),
    'Bandra (W)':           (19.0596, 72.8295),
    'Lower Parel':          (18.9982, 72.8326),
    'Khar (W)':             (19.0726, 72.8373),
    'Dr. E. Moses Road':    (19.0048, 72.8178),
    'Marine Lines':         (18.9432, 72.8236),
    'Charni Road':          (18.9549, 72.8186),
    'Turner Road (Bandra)': (19.0543, 72.8366),
    'Mahalakshmi':          (18.9845, 72.8191),
    'Matunga':              (19.0225, 72.8587),
    'Haji Ali':             (18.9826, 72.8089),
    'Mahim':                (19.0385, 72.8438),
    'Churchgate':           (18.9322, 72.8264),
    'Mumbai Central':       (18.9696, 72.8194),
    'Grant Road':           (18.9642, 72.8183),
    'Mulund (W)':           (19.1728, 72.9569),
}

ALL_COORDS = {**DELHI_COORDS, **CHENNAI_COORDS, **MUMBAI_COORDS}
CITY_DEFAULTS = {
    'Delhi':   (28.6139, 77.2090),
    'Chennai': (13.0827, 80.2707),
    'Mumbai':  (19.0760, 72.8777),
}

# ============================================================================
# STEP 1: LOAD & MERGE
# ============================================================================
print("\n  Loading datasets...")

BASE = r"d:\Noise polllution Zone\noise-pollution-dashboard"

df_delhi   = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'delhi_noise_2020_2024.csv'))
df_chennai = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'chennai_noise_2020_2024.csv'))
df_mumbai  = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'mumbai_noise_2020_2024.csv'))

if 'City' not in df_delhi.columns:   df_delhi['City']   = 'Delhi'
if 'City' not in df_chennai.columns: df_chennai['City'] = 'Chennai'
if 'City' not in df_mumbai.columns:  df_mumbai['City']  = 'Mumbai'

# Normalise standard column names: DPCC (Delhi) / CPCB (Mumbai) → Std_Day / Std_Night
def normalise(df):
    if 'DPCC_Day_Std_dB' in df.columns:
        df = df.rename(columns={'DPCC_Day_Std_dB': 'Std_Day_dB', 'DPCC_Night_Std_dB': 'Std_Night_dB'})
    elif 'CPCB_Day_Std_dB' in df.columns:
        df = df.rename(columns={'CPCB_Day_Std_dB': 'Std_Day_dB', 'CPCB_Night_Std_dB': 'Std_Night_dB'})
    return df

df_delhi   = normalise(df_delhi)
df_chennai = normalise(df_chennai)
df_mumbai  = normalise(df_mumbai)

# Add missing Std columns to Chennai (no standard defined)
for df in [df_chennai]:
    if 'Std_Day_dB' not in df.columns:
        df['Std_Day_dB'] = float('nan')
    if 'Std_Night_dB' not in df.columns:
        df['Std_Night_dB'] = float('nan')
    if 'Excess_Day_dB' not in df.columns:
        df['Excess_Day_dB'] = float('nan')
    if 'Excess_Night_dB' not in df.columns:
        df['Excess_Night_dB'] = float('nan')

KEEP_COLS = ['Location', 'City', 'Zone_Type', 'Year', 'Month', 'Month_Name',
             'Noise_Day_dB', 'Noise_Night_dB', 'Excess_Day_dB', 'Excess_Night_dB',
             'Std_Day_dB', 'Zone_Category']

df_all = pd.concat(
    [df_delhi[KEEP_COLS], df_chennai[KEEP_COLS], df_mumbai[KEEP_COLS]],
    ignore_index=True
)

print(f"    Delhi:   {df_delhi.shape[0]} rows, {df_delhi['Location'].nunique()} locations")
print(f"    Chennai: {df_chennai.shape[0]} rows, {df_chennai['Location'].nunique()} locations")
print(f"    Mumbai:  {df_mumbai.shape[0]} rows, {df_mumbai['Location'].nunique()} locations")
print(f"    Merged:  {df_all.shape[0]} rows, {df_all['Location'].nunique()} locations")

# ============================================================================
# STEP 2: BUILD GEO RECORDS
# ============================================================================
print("\n  Building geo records...")

geo_records = []

for loc in df_all['Location'].unique():
    subset = df_all[df_all['Location'] == loc]
    city   = subset['City'].iloc[0]

    # Coordinates
    lat, lon = ALL_COORDS.get(loc, CITY_DEFAULTS.get(city, (20.5, 79.0)))
    if loc not in ALL_COORDS:
        print(f"    WARNING: No coords for '{loc}' ({city}) — using city default")

    # Standard day dB (may be NaN for Chennai)
    std_raw = subset['Std_Day_dB'].iloc[0]
    std_day = int(std_raw) if pd.notna(std_raw) else 0

    # Excess (may be NaN for Chennai)
    excess_mean = subset['Excess_Day_dB'].mean()
    avg_excess  = round(float(excess_mean), 2) if pd.notna(excess_mean) else 0.0

    # Per-year breakdown
    yearly_clean = []
    for yr, grp in subset.groupby('Year'):
        yearly_clean.append({
            'Year':          int(yr),
            'Avg_Day':       round(grp['Noise_Day_dB'].mean(), 2),
            'Avg_Night':     round(grp['Noise_Night_dB'].mean(), 2),
            'Zone_Category': grp['Zone_Category'].mode()[0],
        })

    geo_records.append({
        'Location':       loc,
        'City':           city,
        'Latitude':       lat,
        'Longitude':      lon,
        'Zone_Type':      subset['Zone_Type'].iloc[0],
        'Avg_Day':        round(subset['Noise_Day_dB'].mean(), 2),
        'Avg_Night':      round(subset['Noise_Night_dB'].mean(), 2),
        'Avg_Excess_Day': avg_excess,
        'DPCC_Std_Day':   std_day,   # unified key for frontend
        'Zone_Category':  subset['Zone_Category'].mode()[0],
        'Yearly':         yearly_clean,
    })

    print(f"    [{city:7s}] {loc}: ({lat:.4f}, {lon:.4f})")

# ============================================================================
# STEP 3: SAVE JSON
# ============================================================================
print("\n  Saving JSON...")

geo_df  = pd.DataFrame(geo_records)
delhi_n   = (geo_df['City'] == 'Delhi').sum()
chennai_n = (geo_df['City'] == 'Chennai').sum()
mumbai_n  = (geo_df['City'] == 'Mumbai').sum()

output = {'locations': geo_records}

geo_out = os.path.join(BASE, 'data', 'geo', 'all_cities_locations_geo.json')
fe_out  = os.path.join(BASE, 'frontend', 'all_cities_locations_geo.json')
pub_out = os.path.join(BASE, 'public', 'all_cities_locations_geo.json')

with open(geo_out, 'w') as f:
    json.dump(output, f, indent=2)

shutil.copy(geo_out, fe_out)

if os.path.isdir(os.path.dirname(pub_out)):
    shutil.copy(geo_out, pub_out)

fsize = os.path.getsize(fe_out)
print(f"  Saved: all_cities_locations_geo.json ({fsize:,} bytes)")
print(f"  Total: {len(geo_records)} locations — Delhi: {delhi_n}, Chennai: {chennai_n}, Mumbai: {mumbai_n}")
print("=" * 70)
