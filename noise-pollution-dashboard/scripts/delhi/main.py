"""
Noise Pollution Zone Classification for Delhi (2020-2024) — v2
================================================================
Steps 1-5: Preprocessing, ML Classification, Trend Analysis,
            DPCC Compliance Report, Geocoding
Uses: data/processed/delhi_noise_2020_2024.csv (14-column v2 with DPCC standards)
Outputs: model, plots, dpcc_compliance_report.csv, locations_geo.json
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
print("  NOISE POLLUTION ZONE CLASSIFICATION - DELHI (2020-2024) v2")
print("=" * 70)

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 1: DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv('../../data/processed/delhi_noise_2020_2024.csv')
print(f"\n  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

# Drop legacy columns if present (no longer in v2)
legacy_cols = ['Base_2008_Day_dB', 'Base_2008_Night_dB']
dropped = [c for c in legacy_cols if c in df.columns]
if dropped:
    df.drop(columns=dropped, inplace=True)
    print(f"  Dropped legacy columns: {dropped}")

# Convert numeric columns to float
numeric_cols = ['Noise_Day_dB', 'Noise_Night_dB', 'DPCC_Day_Std_dB',
                'DPCC_Night_Std_dB', 'Excess_Day_dB', 'Excess_Night_dB']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle nulls
null_counts = df.isnull().sum()
if null_counts.any():
    print("  Null values found - applying mean imputation")
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
else:
    print("  No null values found")

# Encode Zone_Type
le_zone_type = LabelEncoder()
df['Zone_Type_Encoded'] = le_zone_type.fit_transform(df['Zone_Type'])
print(f"  Zone_Type encoding: {dict(zip(le_zone_type.classes_, le_zone_type.transform(le_zone_type.classes_)))}")

# Derived features
df['Avg_Noise'] = (df['Noise_Day_dB'] + df['Noise_Night_dB']) / 2.0
df['Noise_Diff'] = df['Noise_Day_dB'] - df['Noise_Night_dB']
df['Violation_Ratio'] = df['Excess_Day_dB'] / df['DPCC_Day_Std_dB']

# Season: Summer=1, Monsoon=2, Winter=3, Spring=4
def map_season(month):
    if month in [4, 5, 6]:
        return 1  # Summer
    elif month in [7, 8, 9]:
        return 2  # Monsoon
    elif month in [10, 11, 12, 1, 2]:
        return 3  # Winter
    else:
        return 4  # Spring (March)

df['Season'] = df['Month'].apply(map_season)

# Encode target
le_target = LabelEncoder()
df['Zone_Category_Encoded'] = le_target.fit_transform(df['Zone_Category'])
print(f"  Zone_Category encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
print(f"  Zone distribution:\n{df['Zone_Category'].value_counts().to_string()}")
print(f"  Derived features added: Avg_Noise, Noise_Diff, Violation_Ratio, Season")
print(f"  Final shape: {df.shape}")

# ============================================================================
# STEP 2: SUPERVISED CLASSIFICATION MODELS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: SUPERVISED CLASSIFICATION MODELS")
print("=" * 70)

feature_names = ['Noise_Day_dB', 'Noise_Night_dB', 'Avg_Noise', 'Noise_Diff',
                 'Excess_Day_dB', 'Excess_Night_dB', 'Violation_Ratio',
                 'Zone_Type_Encoded', 'Year', 'Month', 'Season']
X = df[feature_names].values
y = df['Zone_Category_Encoded'].values
class_names = le_target.classes_

print(f"\n  Features ({len(feature_names)}): {feature_names}")
print(f"  Target classes: {list(class_names)}")

# Check minimum class count for stratification
from collections import Counter
class_counts = Counter(y)
min_class_count = min(class_counts.values())
print(f"  Class distribution: {dict(class_counts)}")

if min_class_count >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print(f"  WARNING: Class with only {min_class_count} sample(s) detected — using non-stratified split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
print(f"    Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}")
print(classification_report(y_test, rf_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0))

# XGBoost
print("  --- XGBoost ---")
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    random_state=42, use_label_encoder=False, eval_metric='mlogloss',
    subsample=0.8, colsample_bytree=0.8
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
print(f"    Accuracy: {xgb_accuracy:.4f}, F1: {xgb_f1:.4f}")
print(classification_report(y_test, xgb_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0))

# Comparison
best_name = 'Random Forest' if rf_f1 >= xgb_f1 else 'XGBoost'
best_obj = rf_model if rf_f1 >= xgb_f1 else xgb_model
print(f"  Best: {best_name}")

# Feature Importance
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='white', height=0.65)
ax.set_xlabel('Importance', fontsize=13, fontweight='bold')
ax.set_title('Random Forest - Feature Importance', fontsize=16, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, feat_imp['Importance']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../../outputs/plots/delhi/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance.png")

# RF Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, rf_pred, labels=range(len(class_names)))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
ax.set_title('Random Forest - Confusion Matrix', fontsize=15, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('../../outputs/plots/delhi/rf_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rf_confusion_matrix.png")

# XGB Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_xgb = confusion_matrix(y_test, xgb_pred, labels=range(len(class_names)))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
ax.set_title('XGBoost - Confusion Matrix', fontsize=15, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('../../outputs/plots/delhi/xgb_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: xgb_confusion_matrix.png")

# Save best model
joblib.dump({
    'model': best_obj, 'label_encoder_target': le_target,
    'label_encoder_zone_type': le_zone_type, 'feature_names': feature_names,
    'model_name': best_name, 'accuracy': max(rf_accuracy, xgb_accuracy),
    'f1_score': max(rf_f1, xgb_f1)
}, '../../models/delhi/delhi_noise_model.pkl')
print(f"  Saved: noise_zone_model.pkl ({best_name})")

# ============================================================================
# STEP 3: YEAR-WISE TREND ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 3: YEAR-WISE TREND ANALYSIS")
print("=" * 70)

yearly = df.groupby('Year')[['Noise_Day_dB', 'Noise_Night_dB']].mean().reset_index()
print(yearly.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yearly['Year'], yearly['Noise_Day_dB'], marker='o', ms=10, lw=2.5,
        color='#e74c3c', label='Day Noise (dB)', zorder=5)
ax.plot(yearly['Year'], yearly['Noise_Night_dB'], marker='s', ms=10, lw=2.5,
        color='#3498db', label='Night Noise (dB)', zorder=5)
ax.fill_between(yearly['Year'], yearly['Noise_Day_dB'], yearly['Noise_Night_dB'],
                alpha=0.1, color='#9b59b6')

# COVID annotation
min_yr = yearly.loc[yearly['Noise_Day_dB'].idxmin(), 'Year']
min_val = yearly['Noise_Day_dB'].min()
ax.annotate('COVID-19\nLockdown Dip', xy=(min_yr, min_val),
            xytext=(min_yr + 0.5, min_val - 1.5), fontsize=10, fontweight='bold',
            color='#e74c3c', arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

max_yr = yearly.loc[yearly['Noise_Day_dB'].idxmax(), 'Year']
max_val = yearly['Noise_Day_dB'].max()
ax.annotate('Post-COVID\nRecovery', xy=(max_yr, max_val),
            xytext=(max_yr - 0.5, max_val + 1.5), fontsize=10, fontweight='bold',
            color='#2ecc71', arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

for _, r in yearly.iterrows():
    ax.text(r['Year'], r['Noise_Day_dB'] + 0.4, f"{r['Noise_Day_dB']:.1f}",
            ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
    ax.text(r['Year'], r['Noise_Night_dB'] - 0.6, f"{r['Noise_Night_dB']:.1f}",
            ha='center', fontsize=9, color='#3498db', fontweight='bold')

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Noise (dB)', fontsize=13, fontweight='bold')
ax.set_title('Delhi Noise Pollution Trend (2020-2024)', fontsize=15, fontweight='bold')
ax.set_xticks(yearly['Year'])
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../../outputs/plots/delhi/noise_trend_2020_2024.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: noise_trend_2020_2024.png")

# ============================================================================
# STEP 4: DPCC COMPLIANCE REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 4: DPCC COMPLIANCE REPORT")
print("=" * 70)

compliance = df.groupby('Zone_Type').agg(
    DPCC_Day_Standard=('DPCC_Day_Std_dB', 'first'),
    Avg_Recorded_Day_dB=('Noise_Day_dB', 'mean'),
    Avg_Excess_Day_dB=('Excess_Day_dB', 'mean'),
).reset_index()

# Calculate % of months in violation per Zone_Type
violation_pct = df.groupby('Zone_Type').apply(
    lambda g: round((g['Excess_Day_dB'] > 0).sum() / len(g) * 100, 1)
).reset_index(name='Violation_Pct')

compliance = compliance.merge(violation_pct, on='Zone_Type')
compliance['Avg_Recorded_Day_dB'] = compliance['Avg_Recorded_Day_dB'].round(2)
compliance['Avg_Excess_Day_dB'] = compliance['Avg_Excess_Day_dB'].round(2)

print("\n  DPCC Compliance Summary by Zone Type:")
print("  " + "-" * 68)
print(f"  {'Zone Type':<15} {'DPCC Std':>10} {'Avg Day dB':>12} {'Avg Excess':>12} {'Violation %':>12}")
print("  " + "-" * 68)
for _, row in compliance.iterrows():
    print(f"  {row['Zone_Type']:<15} {row['DPCC_Day_Standard']:>10.0f} "
          f"{row['Avg_Recorded_Day_dB']:>12.2f} {row['Avg_Excess_Day_dB']:>12.2f} "
          f"{row['Violation_Pct']:>11.1f}%")
print("  " + "-" * 68)

compliance.to_csv('../../outputs/plots/delhi/dpcc_compliance_report.csv', index=False)
print("  Saved: dpcc_compliance_report.csv")

# ============================================================================
# STEP 5: GEOCODING & JSON EXPORT
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 5: GEOCODING & JSON EXPORT")
print("=" * 70)

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

geolocator = Nominatim(user_agent="delhi_noise_map", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

# Aggregate per location — overall and yearly
loc_overall = df.groupby('Location').agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Avg_Excess_Day=('Excess_Day_dB', 'mean'),
    DPCC_Std_Day=('DPCC_Day_Std_dB', 'first'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

loc_yearly = df.groupby(['Location', 'Year']).agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

# Use hardcoded coordinates directly — no network geocoding
latitudes, longitudes = [], []
for _, row in loc_overall.iterrows():
    loc = row['Location'].strip()
    if loc in DELHI_COORDS:
        lat, lon = DELHI_COORDS[loc]
        latitudes.append(lat)
        longitudes.append(lon)
        print(f"    {loc}: ({lat:.4f}, {lon:.4f})")
    else:
        latitudes.append(28.6139)
        longitudes.append(77.2090)
        print(f"    {loc}: (28.6139, 77.2090) [Default — missing from DELHI_COORDS]")

loc_overall['Latitude'] = latitudes
loc_overall['Longitude'] = longitudes

# Build the JSON
geo_records = []
for _, ov in loc_overall.iterrows():
    loc = ov['Location']
    yearly_data = loc_yearly[loc_yearly['Location'] == loc].to_dict('records')
    yearly_clean = []
    for yr in yearly_data:
        yearly_clean.append({
            'Year': int(yr['Year']),
            'Avg_Day': round(yr['Avg_Day'], 2),
            'Avg_Night': round(yr['Avg_Night'], 2),
            'Zone_Category': yr['Zone_Category']
        })
    geo_records.append({
        'Location': loc,
        'Latitude': ov['Latitude'],
        'Longitude': ov['Longitude'],
        'Avg_Day': round(ov['Avg_Day'], 2),
        'Avg_Night': round(ov['Avg_Night'], 2),
        'Avg_Excess_Day': round(ov['Avg_Excess_Day'], 2),
        'DPCC_Std_Day': int(ov['DPCC_Std_Day']),
        'Zone_Category': ov['Zone_Category'],
        'Zone_Type': ov['Zone_Type'],
        'Yearly': yearly_clean
    })

# Global yearly trend
yearly_trend = yearly.to_dict('records')
yearly_trend_clean = [{'Year': int(r['Year']),
                       'Avg_Day': round(r['Noise_Day_dB'], 2),
                       'Avg_Night': round(r['Noise_Night_dB'], 2)} for r in yearly_trend]

output_json = {
    'locations': geo_records,
    'yearly_trend': yearly_trend_clean
}

with open('../../data/geo/delhi_locations_geo.json', 'w') as f:
    json.dump(output_json, f, indent=2)

print(f"\n  Saved: locations_geo.json ({len(geo_records)} locations)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  ALL OUTPUTS GENERATED:")
print("=" * 70)
print("  1. noise_zone_model.pkl")
print("  2. feature_importance.png")
print("  3. rf_confusion_matrix.png")
print("  4. xgb_confusion_matrix.png")
print("  5. noise_trend_2020_2024.png")
print("  6. dpcc_compliance_report.csv")
print("  7. locations_geo.json")
print("  ---")
print("  Frontend files: index.html, style.css, script.js")
print("=" * 70)
print("  DONE!")
print("=" * 70)
