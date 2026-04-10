"""
Noise Pollution Zone Classification and Hexagonal Map Visualization for Delhi (2020-2024)
=========================================================================================
Dataset: delhi_noise_2020_2024.csv — 2,460 rows x 11 columns
41 Delhi locations, monthly Day/Night noise (dB), 2020-2024
Target: Zone_Category (4 classes: Low, Moderate, High, Critical)

Pipeline:
  Step 1 — Data Preprocessing
  Step 2 — Supervised Classification (Random Forest + XGBoost)
  Step 3 — Year-wise Trend Analysis
  Step 4 — Geocoding Delhi Locations
  Step 5 — Hexagonal Noise Map Visualization
  Step 6 — Save Final Outputs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from xgboost import XGBClassifier
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import joblib
import warnings
import os
import time

warnings.filterwarnings('ignore')

BASE_DIR = r"d:\Noise polllution Zone"
os.chdir(BASE_DIR)

print("=" * 70)
print("  NOISE POLLUTION ZONE CLASSIFICATION - DELHI (2020-2024)")
print("=" * 70)

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 1: DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv('delhi_noise_2020_2024.csv')
print(f"\n  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

# Convert numeric columns to float
numeric_cols = ['Noise_Day_dB', 'Noise_Night_dB', 'Base_2008_Day_dB', 'Base_2008_Night_dB']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check and handle nulls
null_counts = df.isnull().sum()
if null_counts.any():
    print(f"\n  Null values found:")
    for col, count in null_counts[null_counts > 0].items():
        print(f"    {col}: {count}")
    # Mean imputation for numeric columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    print("  -> Nulls handled with mean imputation")
else:
    print(f"  No null values found")

# Encode Zone_Type: Commercial=0, Industrial=1, Residential=2
le_zone_type = LabelEncoder()
df['Zone_Type_Encoded'] = le_zone_type.fit_transform(df['Zone_Type'])
zone_type_mapping = dict(zip(le_zone_type.classes_, le_zone_type.transform(le_zone_type.classes_)))
print(f"\n  Zone_Type encoding: {zone_type_mapping}")

# Derived features
df['Avg_Noise'] = (df['Noise_Day_dB'] + df['Noise_Night_dB']) / 2.0
df['Noise_Diff'] = df['Noise_Day_dB'] - df['Noise_Night_dB']

# Season mapping: Month -> Season (as integer)
# Summer=0 (Apr-Jun), Monsoon=1 (Jul-Sep), Winter=2 (Oct-Dec/Jan-Feb), Spring=3 (Mar)
def map_season(month):
    if month in [4, 5, 6]:
        return 0  # Summer
    elif month in [7, 8, 9]:
        return 1  # Monsoon
    elif month in [10, 11, 12, 1, 2]:
        return 2  # Winter
    else:
        return 3  # Spring (March)

df['Season'] = df['Month'].apply(map_season)

season_names = {0: 'Summer', 1: 'Monsoon', 2: 'Winter', 3: 'Spring'}
print(f"  Season mapping: {season_names}")

# Encode target: Zone_Category
le_target = LabelEncoder()
df['Zone_Category_Encoded'] = le_target.fit_transform(df['Zone_Category'])
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print(f"  Zone_Category encoding: {target_mapping}")

print(f"\n  Zone_Category distribution:")
for cat, count in df['Zone_Category'].value_counts().items():
    print(f"    {cat:10s}: {count} ({count/len(df)*100:.1f}%)")

print(f"\n  Derived features added: Avg_Noise, Noise_Diff, Season")
print(f"  Final shape: {df.shape}")

# ============================================================================
# STEP 2: SUPERVISED CLASSIFICATION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: SUPERVISED CLASSIFICATION MODELS")
print("=" * 70)

# Features and target
feature_names = ['Noise_Day_dB', 'Noise_Night_dB', 'Avg_Noise', 'Noise_Diff',
                 'Zone_Type_Encoded', 'Year', 'Month', 'Season']
X = df[feature_names].values
y = df['Zone_Category_Encoded'].values
class_names = le_target.classes_

print(f"\n  Features ({len(feature_names)}): {feature_names}")
print(f"  Target classes: {list(class_names)}")
print(f"  Samples: {len(X)}")

# 80/20 train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ---- Random Forest Classifier ----
print("\n  --- Random Forest Classifier ---")
rf_model = RandomForestClassifier(
    n_estimators=300, random_state=42, max_depth=15,
    min_samples_split=5, min_samples_leaf=2, class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

print(f"    Accuracy:       {rf_accuracy:.4f}")
print(f"    F1-Score (wt):  {rf_f1:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, rf_pred, target_names=class_names, zero_division=0))

# ---- XGBoost Classifier ----
print("  --- XGBoost Classifier ---")
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    random_state=42, use_label_encoder=False, eval_metric='mlogloss',
    subsample=0.8, colsample_bytree=0.8
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')

print(f"    Accuracy:       {xgb_accuracy:.4f}")
print(f"    F1-Score (wt):  {xgb_f1:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=class_names, zero_division=0))

# ---- Model Comparison ----
print("  --- Model Comparison ---")
print(f"    {'Model':<20s} {'Accuracy':>10s} {'F1-Score':>10s}")
print(f"    {'-'*42}")
print(f"    {'Random Forest':<20s} {rf_accuracy:>10.4f} {rf_f1:>10.4f}")
print(f"    {'XGBoost':<20s} {xgb_accuracy:>10.4f} {xgb_f1:>10.4f}")
best_model_name = 'Random Forest' if rf_f1 >= xgb_f1 else 'XGBoost'
best_model_obj = rf_model if rf_f1 >= xgb_f1 else xgb_model
print(f"\n    Best Model: {best_model_name}")

# ---- Feature Importance Plot ----
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(feat_imp_df)))
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
               color=colors, edgecolor='white', linewidth=0.5, height=0.65)
ax.set_xlabel('Feature Importance', fontsize=13, fontweight='bold')
ax.set_title('Random Forest - Feature Importance', fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='y', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, feat_imp_df['Importance']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, color='#333')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: feature_importance.png")

# ---- RF Confusion Matrix ----
fig, ax = plt.subplots(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, rf_pred, labels=range(len(class_names)))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
ax.set_title('Random Forest - Confusion Matrix', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rf_confusion_matrix.png")

# ---- XGB Confusion Matrix ----
fig, ax = plt.subplots(figsize=(8, 6))
cm_xgb = confusion_matrix(y_test, xgb_pred, labels=range(len(class_names)))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
ax.set_title('XGBoost - Confusion Matrix', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('xgb_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: xgb_confusion_matrix.png")

# ============================================================================
# STEP 3: YEAR-WISE TREND ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 3: YEAR-WISE TREND ANALYSIS")
print("=" * 70)

yearly = df.groupby('Year')[['Noise_Day_dB', 'Noise_Night_dB']].mean().reset_index()
print(f"\n  Year-wise Average Noise Levels:")
print(f"  {'Year':>6s}  {'Day (dB)':>10s}  {'Night (dB)':>12s}")
print(f"  {'-'*32}")
for _, row in yearly.iterrows():
    print(f"  {int(row['Year']):>6d}  {row['Noise_Day_dB']:>10.2f}  {row['Noise_Night_dB']:>12.2f}")

fig, ax = plt.subplots(figsize=(12, 6))

# Day noise trend
ax.plot(yearly['Year'], yearly['Noise_Day_dB'],
        marker='o', markersize=10, linewidth=2.5, color='#e74c3c',
        label='Day Time Noise (dB)', zorder=5)

# Night noise trend
ax.plot(yearly['Year'], yearly['Noise_Night_dB'],
        marker='s', markersize=10, linewidth=2.5, color='#3498db',
        label='Night Time Noise (dB)', zorder=5)

# Fill between for visual emphasis
ax.fill_between(yearly['Year'], yearly['Noise_Day_dB'], yearly['Noise_Night_dB'],
                alpha=0.1, color='#9b59b6')

# Annotate COVID dip
min_day_year = yearly.loc[yearly['Noise_Day_dB'].idxmin(), 'Year']
min_day_val = yearly['Noise_Day_dB'].min()
ax.annotate('COVID-19\nLockdown Dip',
            xy=(min_day_year, min_day_val),
            xytext=(min_day_year + 0.5, min_day_val - 1.5),
            fontsize=10, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            ha='center')

# Annotate recovery
max_day_year = yearly.loc[yearly['Noise_Day_dB'].idxmax(), 'Year']
max_day_val = yearly['Noise_Day_dB'].max()
ax.annotate('Post-COVID\nRecovery',
            xy=(max_day_year, max_day_val),
            xytext=(max_day_year - 0.5, max_day_val + 1.5),
            fontsize=10, fontweight='bold', color='#2ecc71',
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5),
            ha='center')

# Add value labels
for _, row in yearly.iterrows():
    ax.text(row['Year'], row['Noise_Day_dB'] + 0.4, f"{row['Noise_Day_dB']:.1f}",
            ha='center', fontsize=9, fontweight='bold', color='#e74c3c')
    ax.text(row['Year'], row['Noise_Night_dB'] - 0.6, f"{row['Noise_Night_dB']:.1f}",
            ha='center', fontsize=9, fontweight='bold', color='#3498db')

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Noise Level (dB)', fontsize=13, fontweight='bold')
ax.set_title('Delhi Noise Pollution Trend (2020-2024)\nDay vs Night Time Noise Levels',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(yearly['Year'])
ax.set_xticklabels([str(int(y)) for y in yearly['Year']], fontsize=11)
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('noise_trend_2020_2024.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: noise_trend_2020_2024.png")

# ============================================================================
# STEP 4: GEOCODING DELHI LOCATIONS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 4: GEOCODING DELHI LOCATIONS")
print("=" * 70)

# Manual fallback coordinates for all 41 Delhi locations
DELHI_COORDS = {
    'Adarsh Nagar': (28.7165, 77.1709),
    'Anand Vihar': (28.6469, 77.3156),
    'Ashok Vihar': (28.6957, 77.1770),
    'Badli': (28.7353, 77.1331),
    'Braham Puri': (28.6750, 77.2700),
    'Daryaganj': (28.6411, 77.2388),
    'Defence Colony': (28.5744, 77.2337),
    'Dwaraka': (28.5921, 77.0460),
    'Greater Kailash': (28.5420, 77.2400),
    'Inder Puri': (28.5960, 77.1770),
    'Janak Puri': (28.6219, 77.0815),
    'Karawal Nagar': (28.7230, 77.2590),
    'Karol Bagh': (28.6514, 77.1907),
    'Kondli': (28.6200, 77.3500),
    'Lajpat Nagar': (28.5700, 77.2400),
    'Lawrence Road': (28.6800, 77.1300),
    'Mandavali': (28.6364, 77.2953),
    'Mangol Puri': (28.7050, 77.1300),
    'Meera Bagh': (28.6700, 77.1100),
    'Mehrauli': (28.5150, 77.1800),
    'Moti Bagh': (28.5800, 77.1700),
    'Moti Nagar': (28.6531, 77.1453),
    'Mukherji Nagar': (28.7073, 77.2100),
    'Nand Nagri': (28.6944, 77.3112),
    'Naraouji Nagar': (28.5900, 77.1800),
    'New Friends Colony': (28.5636, 77.2634),
    'Pahar Ganj': (28.6441, 77.2132),
    'Paschim Vihar': (28.6700, 77.1000),
    'Patel Nagar': (28.6508, 77.1657),
    'Prehladpur': (28.5400, 77.3000),
    'R.K. Puram': (28.5700, 77.1700),
    'Rajpura Road': (28.7100, 77.2300),
    'Rana Pratap Bagh': (28.6900, 77.2000),
    'Rohini': (28.7320, 77.1100),
    'Sarita Vihar': (28.5310, 77.2880),
    'Shalimar Bagh': (28.7184, 77.1600),
    'Shanti Vihar': (28.6100, 77.3100),
    'Tilak Nagar': (28.6400, 77.0900),
    'Tughlakabad': (28.5147, 77.2530),
    'Vasant Kunj': (28.5210, 77.1570),
    'Yamuna Vihar': (28.6970, 77.2720),
}

# Geocode using Nominatim
geolocator = Nominatim(user_agent="delhi_noise_map", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

# Build geo_df with aggregated stats per location
location_stats = df.groupby('Location').agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

latitudes = []
longitudes = []
geocode_stats = {'Geocoded': 0, 'Manual': 0, 'Default': 0}

for _, row in location_stats.iterrows():
    loc_name = row['Location'].strip()

    # Try Nominatim
    try:
        result = geocode(f"{loc_name}, Delhi, India")
        if result and 28.4 < result.latitude < 28.9 and 76.8 < result.longitude < 77.5:
            latitudes.append(result.latitude)
            longitudes.append(result.longitude)
            geocode_stats['Geocoded'] += 1
            print(f"    {loc_name}: ({result.latitude:.4f}, {result.longitude:.4f}) [Geocoded]")
            continue
    except Exception:
        pass

    # Fallback to manual
    if loc_name in DELHI_COORDS:
        lat, lon = DELHI_COORDS[loc_name]
        latitudes.append(lat)
        longitudes.append(lon)
        geocode_stats['Manual'] += 1
        print(f"    {loc_name}: ({lat:.4f}, {lon:.4f}) [Manual]")
    else:
        latitudes.append(28.6139)
        longitudes.append(77.2090)
        geocode_stats['Default'] += 1
        print(f"    {loc_name}: (28.6139, 77.2090) [Default]")

location_stats['Latitude'] = latitudes
location_stats['Longitude'] = longitudes

geo_df = location_stats[['Location', 'Latitude', 'Longitude',
                          'Zone_Category', 'Avg_Day', 'Avg_Night', 'Zone_Type']].copy()

print(f"\n  Geocoding complete: {geocode_stats['Geocoded']} geocoded, "
      f"{geocode_stats['Manual']} manual, {geocode_stats['Default']} default")
print(f"  geo_df shape: {geo_df.shape}")

# ============================================================================
# STEP 5: HEXAGONAL NOISE MAP VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 5: HEXAGONAL NOISE MAP VISUALIZATION")
print("=" * 70)

delhi_center = [28.6139, 77.2090]
m = folium.Map(
    location=delhi_center,
    zoom_start=11,
    tiles='CartoDB dark_matter',
    control_scale=True
)

# Extra tile layers
folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
folium.TileLayer('CartoDB positron', name='CartoDB Light').add_to(m)

# Title overlay
title_html = '''
<div style="position: fixed;
     top: 10px; left: 50%%; transform: translateX(-50%%);
     z-index: 1000;
     background: linear-gradient(135deg, #0d0d2b 0%%, #1a1a40 50%%, #0f3460 100%%);
     padding: 14px 28px;
     border-radius: 12px;
     border: 2px solid #e94560;
     box-shadow: 0 4px 24px rgba(233,69,96,0.35);">
     <h3 style="color: #fff; margin: 0; font-family: 'Segoe UI', Arial, sans-serif;
         text-align: center; letter-spacing: 0.5px;
         text-shadow: 0 0 12px rgba(233,69,96,0.5);">
         Delhi Noise Pollution Zone Map (2020-2024)
     </h3>
     <p style="color: #a0a0a0; margin: 5px 0 0; font-size: 11px; text-align: center;">
         41 Monitoring Stations | Day & Night Noise Levels
     </p>
</div>
''' 
m.get_root().html.add_child(folium.Element(title_html))

# Color function matching required scale
def get_hex_color(avg_day):
    """Map Avg_Day noise to color per spec."""
    if avg_day < 50:
        return '#00b400'      # green
    elif avg_day < 58:
        return '#ffff00'      # yellow
    elif avg_day < 66:
        return '#ff8c00'      # orange
    elif avg_day < 74:
        return '#b40000'      # red
    else:
        return '#640096'      # purple

def get_hex_fill_opacity(avg_day):
    """Higher noise = more opaque."""
    return 0.35 + min(0.5, (avg_day - 50) / 40)

# Add hexagonal markers for each location
for _, row in geo_df.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    avg_day = row['Avg_Day']
    avg_night = row['Avg_Night']
    zone = row['Zone_Category']
    zone_type = row['Zone_Type']
    color = get_hex_color(avg_day)

    # Popup HTML
    popup_html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 240px;
         background: linear-gradient(135deg, #0d0d2b, #1a1a40);
         color: #fff; padding: 16px; border-radius: 10px;
         border-left: 4px solid {color};">
        <h4 style="margin: 0 0 10px; color: {color}; font-size: 15px;">
            {row['Location']}
        </h4>
        <table style="width: 100%; font-size: 12px; color: #ddd;">
            <tr>
                <td style="padding: 4px 0;">Zone Type:</td>
                <td style="text-align: right; font-weight: bold;">{zone_type}</td>
            </tr>
            <tr>
                <td style="padding: 4px 0;">Avg Day Noise:</td>
                <td style="text-align: right; font-weight: bold;">{avg_day:.1f} dB</td>
            </tr>
            <tr>
                <td style="padding: 4px 0;">Avg Night Noise:</td>
                <td style="text-align: right; font-weight: bold;">{avg_night:.1f} dB</td>
            </tr>
            <tr style="border-top: 1px solid #444;">
                <td style="padding: 6px 0;">Zone Category:</td>
                <td style="text-align: right; font-weight: bold; color: {color};
                    font-size: 14px;">{zone}</td>
            </tr>
        </table>
    </div>
    """

    # Hex area circle (radius ~500m, scaled slightly by noise)
    radius = 500 + (avg_day - 50) * 10
    folium.Circle(
        location=[lat, lon],
        radius=radius,
        color=color,
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=get_hex_fill_opacity(avg_day),
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['Location']}: {avg_day:.1f} dB ({zone})"
    ).add_to(m)

    # Hexagon SVG icon with noise value
    hex_svg = f'''
    <div>
        <svg width="42" height="38" viewBox="0 0 42 38" xmlns="http://www.w3.org/2000/svg">
            <polygon points="21,0 40,10 40,28 21,38 2,28 2,10"
                     fill="{color}" fill-opacity="0.9"
                     stroke="#fff" stroke-width="1.5"/>
            <text x="21" y="22" text-anchor="middle"
                  fill="white" font-size="10" font-weight="bold"
                  font-family="Arial">{avg_day:.0f}</text>
        </svg>
    </div>
    '''
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=hex_svg, icon_size=(42, 38), icon_anchor=(21, 19)),
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['Location']}: {avg_day:.1f} dB ({zone})"
    ).add_to(m)

# Color legend
legend_html = '''
<div style="position: fixed;
     bottom: 30px; right: 30px;
     z-index: 1000;
     background: linear-gradient(135deg, #0d0d2b 0%%, #1a1a40 100%%);
     padding: 16px 20px;
     border-radius: 10px;
     border: 1px solid #333;
     box-shadow: 0 4px 18px rgba(0,0,0,0.6);
     font-family: 'Segoe UI', Arial, sans-serif;">
     <h4 style="color: #fff; margin: 0 0 10px; font-size: 13px;
         border-bottom: 1px solid #444; padding-bottom: 8px;">
         Noise Level Legend
     </h4>
     <div style="display: flex; flex-direction: column; gap: 7px;">
         <div style="display: flex; align-items: center; gap: 8px;">
             <svg width="22" height="20"><polygon points="11,0 21,5 21,15 11,20 1,15 1,5"
                  fill="#00b400" stroke="#fff" stroke-width="0.5"/></svg>
             <span style="color: #00b400; font-size: 12px;">Low (&lt;50 dB)</span>
         </div>
         <div style="display: flex; align-items: center; gap: 8px;">
             <svg width="22" height="20"><polygon points="11,0 21,5 21,15 11,20 1,15 1,5"
                  fill="#ffff00" stroke="#fff" stroke-width="0.5"/></svg>
             <span style="color: #ffff00; font-size: 12px;">Moderate (50-58 dB)</span>
         </div>
         <div style="display: flex; align-items: center; gap: 8px;">
             <svg width="22" height="20"><polygon points="11,0 21,5 21,15 11,20 1,15 1,5"
                  fill="#ff8c00" stroke="#fff" stroke-width="0.5"/></svg>
             <span style="color: #ff8c00; font-size: 12px;">High (58-66 dB)</span>
         </div>
         <div style="display: flex; align-items: center; gap: 8px;">
             <svg width="22" height="20"><polygon points="11,0 21,5 21,15 11,20 1,15 1,5"
                  fill="#b40000" stroke="#fff" stroke-width="0.5"/></svg>
             <span style="color: #b40000; font-size: 12px;">Critical (66-74 dB)</span>
         </div>
         <div style="display: flex; align-items: center; gap: 8px;">
             <svg width="22" height="20"><polygon points="11,0 21,5 21,15 11,20 1,15 1,5"
                  fill="#640096" stroke="#fff" stroke-width="0.5"/></svg>
             <span style="color: #c080ff; font-size: 12px;">Severe (&gt;74 dB)</span>
         </div>
     </div>
     <p style="color: #666; font-size: 9px; margin: 8px 0 0;
        border-top: 1px solid #333; padding-top: 6px;">
        Hex size proportional to noise level
     </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)
m.save('delhi_noise_map.html')
print("  Saved: delhi_noise_map.html")

# ============================================================================
# STEP 6: SAVE FINAL OUTPUTS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 6: SAVING FINAL OUTPUTS")
print("=" * 70)

# Save best model
joblib.dump({
    'model': best_model_obj,
    'label_encoder_target': le_target,
    'label_encoder_zone_type': le_zone_type,
    'feature_names': feature_names,
    'model_name': best_model_name,
    'accuracy': max(rf_accuracy, xgb_accuracy),
    'f1_score': max(rf_f1, xgb_f1)
}, 'noise_zone_model.pkl')
print(f"  [1] noise_zone_model.pkl ({best_model_name})")
print("  [2] feature_importance.png")
print("  [3] rf_confusion_matrix.png")
print("  [4] xgb_confusion_matrix.png")
print("  [5] noise_trend_2020_2024.png")
print("  [6] delhi_noise_map.html")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"""
  Dataset: delhi_noise_2020_2024.csv
    Rows:               {len(df)}
    Locations:          {df['Location'].nunique()}
    Years:              2020-2024
    Zone Types:         {list(df['Zone_Type'].unique())}

  Zone Distribution:
    Critical:           {(df['Zone_Category']=='Critical').sum()} rows ({(df['Zone_Category']=='Critical').mean()*100:.1f}%)
    High:               {(df['Zone_Category']=='High').sum()} rows ({(df['Zone_Category']=='High').mean()*100:.1f}%)

  Model Performance:
    Random Forest:  Accuracy={rf_accuracy:.4f}, F1={rf_f1:.4f}
    XGBoost:        Accuracy={xgb_accuracy:.4f}, F1={xgb_f1:.4f}
    Best: {best_model_name}

  Output Files:
    1. noise_zone_model.pkl
    2. feature_importance.png
    3. rf_confusion_matrix.png
    4. xgb_confusion_matrix.png
    5. noise_trend_2020_2024.png
    6. delhi_noise_map.html
""")
print("=" * 70)
print("  ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 70)
