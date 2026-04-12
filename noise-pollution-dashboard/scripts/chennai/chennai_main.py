"""
Noise Pollution Zone Classification for Chennai (2020-2024)
============================================================
Steps 1-4: Preprocessing, ML Classification, Trend Analysis, Geocoding
Outputs: model, plots, chennai_locations_geo.json for frontend
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
import joblib
import json
import warnings
import os

warnings.filterwarnings('ignore')

BASE_DIR = r"d:\Noise polllution Zone"
# # os.chdir(BASE_DIR)

print("=" * 70)
print("  NOISE POLLUTION ZONE CLASSIFICATION - CHENNAI (2020-2024)")
print("=" * 70)

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 1: DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv('../../data/processed/chennai_noise_2020_2024.csv')
print(f"\n  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# Convert numeric columns
numeric_cols = ['Noise_Day_dB', 'Noise_Night_dB', 'Base_2019_Day_dB', 'Base_2019_Night_dB']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle nulls
if df.isnull().any().any():
    print("  Nulls found - applying mean imputation")
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
else:
    print("  No null values found")

# Encode Zone_Type: Commercial=0, Industrial=1, Residential=2, Silence=3
le_zone_type = LabelEncoder()
df['Zone_Type_Encoded'] = le_zone_type.fit_transform(df['Zone_Type'])
print(f"  Zone_Type encoding: {dict(zip(le_zone_type.classes_, le_zone_type.transform(le_zone_type.classes_)))}")

# Derived features
df['Avg_Noise'] = (df['Noise_Day_dB'] + df['Noise_Night_dB']) / 2.0
df['Noise_Diff'] = df['Noise_Day_dB'] - df['Noise_Night_dB']

def map_season(month):
    if month in [4, 5, 6]:
        return 1  # Summer
    elif month in [7, 8, 9]:
        return 2  # Monsoon
    elif month in [10, 11, 12, 1, 2]:
        return 3  # Winter
    else:
        return 4  # Spring

df['Season'] = df['Month'].apply(map_season)

# Encode target
le_target = LabelEncoder()
df['Zone_Category_Encoded'] = le_target.fit_transform(df['Zone_Category'])
print(f"  Zone_Category encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
print(f"\n  Zone distribution:")
for cat, count in df['Zone_Category'].value_counts().items():
    print(f"    {cat:10s}: {count} ({count/len(df)*100:.1f}%)")
print(f"  Derived features: Avg_Noise, Noise_Diff, Season")
print(f"  Final shape: {df.shape}")

# ============================================================================
# STEP 2: SUPERVISED CLASSIFICATION MODELS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: SUPERVISED CLASSIFICATION MODELS")
print("=" * 70)

feature_names = ['Noise_Day_dB', 'Noise_Night_dB', 'Avg_Noise', 'Noise_Diff',
                 'Zone_Type_Encoded', 'Year', 'Month', 'Season']
X = df[feature_names].values
y = df['Zone_Category_Encoded'].values
class_names = le_target.classes_

print(f"\n  Features ({len(feature_names)}): {feature_names}")
print(f"  Target classes: {list(class_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Random Forest
print("\n  --- Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300, random_state=42, max_depth=15,
    min_samples_split=5, min_samples_leaf=2, class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
print(f"    Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
print(classification_report(y_test, rf_pred, target_names=class_names, zero_division=0))

# XGBoost
print("  --- XGBoost ---")
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    random_state=42, use_label_encoder=False, eval_metric='mlogloss',
    subsample=0.8, colsample_bytree=0.8
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
print(f"    Accuracy: {xgb_acc:.4f}, F1: {xgb_f1:.4f}")
print(classification_report(y_test, xgb_pred, target_names=class_names, zero_division=0))

best_name = 'Random Forest' if rf_f1 >= xgb_f1 else 'XGBoost'
best_obj = rf_model if rf_f1 >= xgb_f1 else xgb_model
print(f"  Best: {best_name}")

# Feature Importance
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.YlOrRd(np.linspace(0.25, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='white', height=0.65)
ax.set_xlabel('Importance', fontsize=13, fontweight='bold')
ax.set_title('Random Forest - Feature Importance (Chennai)', fontsize=16, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, feat_imp['Importance']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../../outputs/plots/chennai/chennai_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chennai_feature_importance.png")

# RF Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, rf_pred, labels=range(len(class_names)))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
ax.set_title('Random Forest - Confusion Matrix (Chennai)', fontsize=15, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('../../outputs/plots/chennai/chennai_rf_cm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chennai_rf_cm.png")

# XGB Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_xgb = confusion_matrix(y_test, xgb_pred, labels=range(len(class_names)))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
ax.set_title('XGBoost - Confusion Matrix (Chennai)', fontsize=15, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('../../outputs/plots/chennai/chennai_xgb_cm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chennai_xgb_cm.png")

# Save model
joblib.dump({
    'model': best_obj, 'label_encoder_target': le_target,
    'label_encoder_zone_type': le_zone_type, 'feature_names': feature_names,
    'model_name': best_name, 'accuracy': max(rf_acc, xgb_acc),
    'f1_score': max(rf_f1, xgb_f1)
}, '../../models/chennai/chennai_noise_model.pkl')
print(f"  Saved: chennai_noise_model.pkl ({best_name})")

# ============================================================================
# STEP 3: YEAR-WISE TREND ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 3: YEAR-WISE TREND ANALYSIS")
print("=" * 70)

# Overall yearly trend
yearly = df.groupby('Year')[['Noise_Day_dB', 'Noise_Night_dB']].mean().reset_index()
print("\n  Year-wise averages:")
print(yearly.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yearly['Year'], yearly['Noise_Day_dB'], marker='o', ms=10, lw=2.5,
        color='#ff6b35', label='Day Noise (dB)', zorder=5)
ax.plot(yearly['Year'], yearly['Noise_Night_dB'], marker='s', ms=10, lw=2.5,
        color='#00b4d8', label='Night Noise (dB)', zorder=5)
ax.fill_between(yearly['Year'], yearly['Noise_Day_dB'], yearly['Noise_Night_dB'],
                alpha=0.1, color='#9b59b6')

min_yr = yearly.loc[yearly['Noise_Day_dB'].idxmin(), 'Year']
min_val = yearly['Noise_Day_dB'].min()
ax.annotate('COVID-19\nLockdown Dip', xy=(min_yr, min_val),
            xytext=(min_yr + 0.5, min_val - 1.5), fontsize=10, fontweight='bold',
            color='#ff6b35', arrowprops=dict(arrowstyle='->', color='#ff6b35', lw=1.5))

max_yr = yearly.loc[yearly['Noise_Day_dB'].idxmax(), 'Year']
max_val = yearly['Noise_Day_dB'].max()
ax.annotate('Post-COVID\nRecovery', xy=(max_yr, max_val),
            xytext=(max_yr - 0.5, max_val + 1.5), fontsize=10, fontweight='bold',
            color='#2ecc71', arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

for _, r in yearly.iterrows():
    ax.text(r['Year'], r['Noise_Day_dB'] + 0.3, f"{r['Noise_Day_dB']:.1f}",
            ha='center', fontsize=9, color='#ff6b35', fontweight='bold')
    ax.text(r['Year'], r['Noise_Night_dB'] - 0.5, f"{r['Noise_Night_dB']:.1f}",
            ha='center', fontsize=9, color='#00b4d8', fontweight='bold')

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Noise (dB)', fontsize=13, fontweight='bold')
ax.set_title('Chennai Noise Pollution Trend (2020-2024)', fontsize=15, fontweight='bold')
ax.set_xticks(yearly['Year'])
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../../outputs/plots/chennai/chennai_noise_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chennai_noise_trend.png")

# Zone-wise bar chart
zone_yearly = df.groupby(['Year', 'Zone_Type'])['Noise_Day_dB'].mean().unstack()
fig, ax = plt.subplots(figsize=(12, 6))
zone_yearly.plot(kind='bar', ax=ax, width=0.75, edgecolor='white', linewidth=0.5,
                 color=['#ff6b35', '#3498db', '#2ecc71', '#9b59b6'])
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Day Noise (dB)', fontsize=13, fontweight='bold')
ax.set_title('Chennai: Zone-wise Noise Trend (2020-2024)', fontsize=15, fontweight='bold')
ax.legend(title='Zone Type', fontsize=10, title_fontsize=11)
ax.set_xticklabels([str(int(x.get_text())) for x in ax.get_xticklabels()], rotation=0)
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../../outputs/plots/chennai/chennai_zonewise_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chennai_zonewise_trend.png")

# ============================================================================
# STEP 4: GEOCODING & JSON EXPORT
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 4: GEOCODING & JSON EXPORT")
print("=" * 70)

CHENNAI_COORDS = {
    'Guindy': (13.0067, 80.2206),
    'Perambur': (13.1143, 80.2379),
    'T Nagar': (13.0418, 80.2341),
    'Triplicane': (13.0576, 80.2750),
    'Pallikaranai': (12.9370, 80.2030),
    'Velachery': (12.9815, 80.2180),
    'Washermanpet': (13.1280, 80.2850),
    'Anna Nagar': (13.0850, 80.2101),
    'Sowcarpet': (13.0952, 80.2850),
    'Egmore Eye Hospital': (13.0732, 80.2609),
}

geolocator = Nominatim(user_agent="chennai_noise_map", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

# Aggregate overall per location
loc_overall = df.groupby('Location').agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

# Per location+year
loc_yearly = df.groupby(['Location', 'Year']).agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

# Geocode
latitudes, longitudes = [], []
for _, row in loc_overall.iterrows():
    loc = row['Location'].strip()
    try:
        result = geocode(f"{loc}, Chennai, India")
        if result and 12.8 < result.latitude < 13.3 and 80.0 < result.longitude < 80.4:
            latitudes.append(round(result.latitude, 6))
            longitudes.append(round(result.longitude, 6))
            print(f"    {loc}: ({result.latitude:.4f}, {result.longitude:.4f}) [Geocoded]")
            continue
    except Exception:
        pass
    if loc in CHENNAI_COORDS:
        lat, lon = CHENNAI_COORDS[loc]
        latitudes.append(lat)
        longitudes.append(lon)
        print(f"    {loc}: ({lat:.4f}, {lon:.4f}) [Manual]")
    else:
        latitudes.append(13.0827)
        longitudes.append(80.2707)
        print(f"    {loc}: (13.0827, 80.2707) [Default]")

loc_overall['Latitude'] = latitudes
loc_overall['Longitude'] = longitudes

# Build JSON with yearly breakdown
geo_records = []
for _, ov in loc_overall.iterrows():
    loc = ov['Location']
    yearly_data = loc_yearly[loc_yearly['Location'] == loc].to_dict('records')
    yearly_clean = [{'Year': int(yr['Year']), 'Avg_Day': round(yr['Avg_Day'], 2),
                     'Avg_Night': round(yr['Avg_Night'], 2),
                     'Zone_Category': yr['Zone_Category']} for yr in yearly_data]
    geo_records.append({
        'Location': loc,
        'Latitude': ov['Latitude'],
        'Longitude': ov['Longitude'],
        'Avg_Day': round(ov['Avg_Day'], 2),
        'Avg_Night': round(ov['Avg_Night'], 2),
        'Zone_Category': ov['Zone_Category'],
        'Zone_Type': ov['Zone_Type'],
        'Yearly': yearly_clean
    })

# Global yearly trend
yearly_trend = yearly.to_dict('records')
yearly_trend_clean = [{'Year': int(r['Year']),
                       'Avg_Day': round(r['Noise_Day_dB'], 2),
                       'Avg_Night': round(r['Noise_Night_dB'], 2)} for r in yearly_trend]

output_json = {'locations': geo_records, 'yearly_trend': yearly_trend_clean}
with open('../../data/geo/chennai_locations_geo.json', 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"\n  Saved: chennai_locations_geo.json ({len(geo_records)} locations)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  ALL OUTPUTS GENERATED:")
print("=" * 70)
outputs = [
    '../../models/chennai/chennai_noise_model.pkl', '../../outputs/plots/chennai/chennai_feature_importance.png',
    '../../outputs/plots/chennai/chennai_rf_cm.png', '../../outputs/plots/chennai/chennai_xgb_cm.png',
    '../../outputs/plots/chennai/chennai_noise_trend.png', '../../outputs/plots/chennai/chennai_zonewise_trend.png',
    '../../data/geo/chennai_locations_geo.json'
]
for i, f in enumerate(outputs, 1):
    sz = os.path.getsize(f) if os.path.exists(f) else 0
    print(f"  {i}. {f} ({sz:,} bytes)")
print("  ---")
print("  Frontend: chennai_index.html, chennai_style.css, chennai_script.js")
print("=" * 70)
print("  DONE!")
print("=" * 70)
