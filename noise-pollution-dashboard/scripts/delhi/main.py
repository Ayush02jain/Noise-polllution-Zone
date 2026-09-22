"""
Noise Pollution Zone Forecasting for Delhi (2020-2025)
======================================================
Steps 1-5: Preprocessing, ML Regression Forecasting (RF & XGBoost),
           Trend Analysis + 2025 Forecast, DPCC Compliance Report, JSON Export
Uses: data/processed/delhi_noise_2020_2024.csv
Outputs: models/delhi/delhi_noise_model.pkl, plots, dpcc_compliance_report.csv, delhi_locations_geo.json
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import json
import warnings
import os

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'plots', 'delhi')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'delhi')
DATA_GEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'geo')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_GEO_DIR, exist_ok=True)

print("=" * 70)
print("  NOISE POLLUTION FORECASTING & REGRESSION - DELHI")
print("=" * 70)

# ============================================================================
# STEP 1: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 1: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'delhi_noise_2020_2024.csv')
df = pd.read_csv(csv_path)
print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# Drop legacy columns if present
legacy_cols = ['Base_2008_Day_dB', 'Base_2008_Night_dB']
dropped = [c for c in legacy_cols if c in df.columns]
if dropped:
    df.drop(columns=dropped, inplace=True)
    print(f"  Dropped legacy columns: {dropped}")

# Numeric conversion
numeric_cols = ['Noise_Day_dB', 'Noise_Night_dB', 'DPCC_Day_Std_dB',
                'DPCC_Night_Std_dB', 'Excess_Day_dB', 'Excess_Night_dB']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Mean imputation if needed
for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# Season mapping
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

# Cyclical month features
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

# Encodings
le_zone_type = LabelEncoder()
df['Zone_Type_Encoded'] = le_zone_type.fit_transform(df['Zone_Type'])

le_loc = LabelEncoder()
df['Location_Encoded'] = le_loc.fit_transform(df['Location'])

# Compute historical baseline noise per location (using 2020-2023 to avoid leakage)
train_mask = df['Year'] < 2024
loc_hist_mean = df[train_mask].groupby('Location')['Noise_Day_dB'].mean().to_dict()
loc_hist_mean_night = df[train_mask].groupby('Location')['Noise_Night_dB'].mean().to_dict()

# Fallback for any unseen location
overall_mean_day = df[train_mask]['Noise_Day_dB'].mean()
overall_mean_night = df[train_mask]['Noise_Night_dB'].mean()

df['Hist_Loc_Mean_Day'] = df['Location'].map(loc_hist_mean).fillna(overall_mean_day)
df['Hist_Loc_Mean_Night'] = df['Location'].map(loc_hist_mean_night).fillna(overall_mean_night)

# Feature set for forecasting - NO LEAKY DPCC/EXCESS FEATURES
feature_names = ['Zone_Type_Encoded', 'Location_Encoded', 'Year', 'Month',
                 'Season', 'Month_Sin', 'Month_Cos', 'Hist_Loc_Mean_Day']

print(f"  Features ({len(feature_names)}): {feature_names}")
print(f"  Target: Noise_Day_dB")

# ============================================================================
# STEP 2: TIME-BASED TRAIN/TEST SPLIT & REGRESSION MODELING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: TIME-BASED TRAIN/TEST SPLIT & REGRESSION MODELING")
print("=" * 70)

# Train on 2020-2023, Test on 2024
train_df = df[df['Year'] < 2024]
test_df = df[df['Year'] == 2024]

X_train = train_df[feature_names].values
y_train = train_df['Noise_Day_dB'].values
X_test = test_df[feature_names].values
y_test = test_df['Noise_Day_dB'].values

print(f"  Training set (2020-2023): {len(train_df)} samples")
print(f"  Testing set  (2024)     : {len(test_df)} samples")

# Baseline: Historical location mean
baseline_pred = test_df['Hist_Loc_Mean_Day'].values
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"\n  [Baseline - Hist Mean]  RMSE: {baseline_rmse:.3f} dB | MAE: {baseline_mae:.3f} dB | R2: {baseline_r2:.3f}")

# Random Forest Regressor
print("\n  --- Training Random Forest Regressor ---")
rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_split=4,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
print(f"    RF Regressor  ->  RMSE: {rf_rmse:.3f} dB | MAE: {rf_mae:.3f} dB | R2: {rf_r2:.3f} | MAPE: {rf_mape:.2f}%")

# XGBoost Regressor
print("  --- Training XGBoost Regressor ---")
xgb_reg = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
print(f"    XGB Regressor ->  RMSE: {xgb_rmse:.3f} dB | MAE: {xgb_mae:.3f} dB | R2: {xgb_r2:.3f} | MAPE: {xgb_mape:.2f}%")

# Select best model
best_name = 'Random Forest' if rf_rmse <= xgb_rmse else 'XGBoost'
best_model = rf_reg if rf_rmse <= xgb_rmse else xgb_reg
best_rmse = min(rf_rmse, xgb_rmse)
best_mae = min(rf_mae, xgb_mae)
best_r2 = max(rf_r2, xgb_r2)
best_pred = rf_pred if rf_rmse <= xgb_rmse else xgb_pred

print(f"\n  Selected Best Model: {best_name} (RMSE: {best_rmse:.3f} dB, R2: {best_r2:.3f})")

# Also train Night model
rf_night = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
feature_names_night = ['Zone_Type_Encoded', 'Location_Encoded', 'Year', 'Month',
                       'Season', 'Month_Sin', 'Month_Cos', 'Hist_Loc_Mean_Night']
X_train_night = train_df[feature_names_night].values
y_train_night = train_df['Noise_Night_dB'].values
rf_night.fit(X_train_night, y_train_night)

# Save models
model_save_path = os.path.join(MODELS_DIR, 'delhi_noise_model.pkl')
joblib.dump({
    'model_day': best_model,
    'model_night': rf_night,
    'label_encoder_zone_type': le_zone_type,
    'label_encoder_loc': le_loc,
    'feature_names': feature_names,
    'feature_names_night': feature_names_night,
    'loc_hist_mean_day': loc_hist_mean,
    'loc_hist_mean_night': loc_hist_mean_night,
    'overall_mean_day': overall_mean_day,
    'overall_mean_night': overall_mean_night,
    'model_name': best_name,
    'rmse': best_rmse,
    'mae': best_mae,
    'r2': best_r2
}, model_save_path)
print(f"  Saved trained model bundle to: {model_save_path}")

# Plot 1: Feature Importance
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': best_model.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='white', height=0.6)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title(f'Delhi Noise Forecasting - {best_name} Feature Importance', fontsize=14, fontweight='bold', pad=15)
for bar, val in zip(bars, feat_imp['Importance']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance.png")

# Plot 2: Actual vs Predicted (2024 Test Set)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_pred, alpha=0.6, color='#2b5c8f', edgecolors='white', s=60, label='2024 Test Points')
min_val = min(y_test.min(), best_pred.min()) - 2
max_val = max(y_test.max(), best_pred.max()) + 2
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Perfect Fit (y=x)')
ax.set_xlabel('Actual Noise Day (dB) - 2024', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Noise Day (dB) - 2024', fontsize=12, fontweight='bold')
ax.set_title(f'Delhi 2024 Out-of-Time Test Set\nRMSE: {best_rmse:.2f} dB | MAE: {best_mae:.2f} dB | R²: {best_r2:.2f}', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: actual_vs_predicted.png")

# ============================================================================
# STEP 3: 2025 FORECAST & YEAR-WISE TREND
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 3: 2025 FORECAST GENERATION & TREND ANALYSIS")
print("=" * 70)

# Generate forecast records for 2025 across all 12 months for all locations
unique_locs = df[['Location', 'Zone_Type', 'DPCC_Day_Std_dB', 'DPCC_Night_Std_dB']].drop_duplicates()

forecast_rows = []
for _, row in unique_locs.iterrows():
    loc = row['Location']
    zt = row['Zone_Type']
    std_day = row['DPCC_Day_Std_dB']
    std_night = row['DPCC_Night_Std_dB']
    zt_enc = le_zone_type.transform([zt])[0]
    loc_enc = le_loc.transform([loc])[0]
    hist_day = loc_hist_mean.get(loc, overall_mean_day)
    hist_night = loc_hist_mean_night.get(loc, overall_mean_night)

    for m in range(1, 13):
        season = map_season(m)
        m_sin = np.sin(2 * np.pi * m / 12.0)
        m_cos = np.cos(2 * np.pi * m / 12.0)
        
        # Day prediction
        feat_vec_day = np.array([[zt_enc, loc_enc, 2025, m, season, m_sin, m_cos, hist_day]])
        pred_day = float(best_model.predict(feat_vec_day)[0])
        
        # Night prediction
        feat_vec_night = np.array([[zt_enc, loc_enc, 2025, m, season, m_sin, m_cos, hist_night]])
        pred_night = float(rf_night.predict(feat_vec_night)[0])
        
        excess_day = pred_day - std_day
        excess_night = pred_night - std_night
        
        # Assign category based on DPCC statutory rule
        if excess_day <= 0:
            cat = 'Low'
        elif excess_day <= 5:
            cat = 'Moderate'
        elif excess_day <= 15:
            cat = 'High'
        else:
            cat = 'Critical'
            
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        
        forecast_rows.append({
            'Location': loc,
            'City': 'Delhi',
            'Zone_Type': zt,
            'DPCC_Day_Std_dB': std_day,
            'DPCC_Night_Std_dB': std_night,
            'Year': 2025,
            'Month': m,
            'Month_Name': month_names[m-1],
            'Noise_Day_dB': round(pred_day, 2),
            'Noise_Night_dB': round(pred_night, 2),
            'Excess_Day_dB': round(excess_day, 2),
            'Excess_Night_dB': round(excess_night, 2),
            'Zone_Category': cat
        })

df_forecast_2025 = pd.DataFrame(forecast_rows)
print(f"  Generated {len(df_forecast_2025)} forecasted monthly records for 2025 across {len(unique_locs)} Delhi locations.")

# Combine historical + forecast for trend visualization
df_combined = pd.concat([df, df_forecast_2025], ignore_index=True)

yearly = df_combined.groupby('Year')[['Noise_Day_dB', 'Noise_Night_dB']].mean().reset_index()
print("\n  Yearly Trends (including 2025 ML Forecast):")
print(yearly.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
# Historical lines (2020-2024)
hist_yearly = yearly[yearly['Year'] <= 2024]
ax.plot(hist_yearly['Year'], hist_yearly['Noise_Day_dB'], marker='o', ms=9, lw=2.5,
        color='#e74c3c', label='Historical Day Noise (dB)', zorder=5)
ax.plot(hist_yearly['Year'], hist_yearly['Noise_Night_dB'], marker='s', ms=9, lw=2.5,
        color='#3498db', label='Historical Night Noise (dB)', zorder=5)

# Forecast segment (2024-2025 dashed)
fc_yearly = yearly[yearly['Year'] >= 2024]
ax.plot(fc_yearly['Year'], fc_yearly['Noise_Day_dB'], marker='o', ms=9, lw=2.5, ls='--',
        color='#e67e22', label='ML Forecasted Day (2025)', zorder=5)
ax.plot(fc_yearly['Year'], fc_yearly['Noise_Night_dB'], marker='s', ms=9, lw=2.5, ls='--',
        color='#2980b9', label='ML Forecasted Night (2025)', zorder=5)

for _, r in yearly.iterrows():
    ax.text(r['Year'], r['Noise_Day_dB'] + 0.35, f"{r['Noise_Day_dB']:.1f}",
            ha='center', fontsize=9, color='#c0392b', fontweight='bold')
    ax.text(r['Year'], r['Noise_Night_dB'] - 0.55, f"{r['Noise_Night_dB']:.1f}",
            ha='center', fontsize=9, color='#2980b9', fontweight='bold')

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Noise (dB)', fontsize=13, fontweight='bold')
ax.set_title('Delhi Noise Pollution Trend & 2025 ML Forecast', fontsize=15, fontweight='bold')
ax.set_xticks(yearly['Year'])
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'noise_trend_2020_2025.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: noise_trend_2020_2025.png")

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

violation_pct = df.groupby('Zone_Type').apply(
    lambda g: round((g['Excess_Day_dB'] > 0).sum() / len(g) * 100, 1)
).reset_index(name='Violation_Pct')

compliance = compliance.merge(violation_pct, on='Zone_Type')
compliance['Avg_Recorded_Day_dB'] = compliance['Avg_Recorded_Day_dB'].round(2)
compliance['Avg_Excess_Day_dB'] = compliance['Avg_Excess_Day_dB'].round(2)

print(compliance.to_string(index=False))
compliance.to_csv(os.path.join(PLOTS_DIR, 'dpcc_compliance_report.csv'), index=False)
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

loc_overall = df_combined.groupby('Location').agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Avg_Excess_Day=('Excess_Day_dB', 'mean'),
    DPCC_Std_Day=('DPCC_Day_Std_dB', 'first'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

loc_yearly = df_combined.groupby(['Location', 'Year']).agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Zone_Category=('Zone_Category', lambda x: x.mode()[0]),
    Zone_Type=('Zone_Type', 'first')
).reset_index()

latitudes, longitudes = [], []
for _, row in loc_overall.iterrows():
    loc = row['Location'].strip()
    lat, lon = DELHI_COORDS.get(loc, (28.6139, 77.2090))
    latitudes.append(lat)
    longitudes.append(lon)

loc_overall['Latitude'] = latitudes
loc_overall['Longitude'] = longitudes

geo_records = []
for _, ov in loc_overall.iterrows():
    loc = ov['Location']
    yearly_data = loc_yearly[loc_yearly['Location'] == loc].to_dict('records')
    yearly_clean = [{
        'Year': int(yr['Year']),
        'Avg_Day': round(yr['Avg_Day'], 2),
        'Avg_Night': round(yr['Avg_Night'], 2),
        'Zone_Category': yr['Zone_Category']
    } for yr in yearly_data]
    
    geo_records.append({
        'Location': loc,
        'City': 'Delhi',
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

yearly_trend_clean = [{
    'Year': int(r['Year']),
    'Avg_Day': round(r['Noise_Day_dB'], 2),
    'Avg_Night': round(r['Noise_Night_dB'], 2)
} for _, r in yearly.iterrows()]

output_json = {
    'locations': geo_records,
    'yearly_trend': yearly_trend_clean
}

json_path = os.path.join(DATA_GEO_DIR, 'delhi_locations_geo.json')
with open(json_path, 'w') as f:
    json.dump(output_json, f, indent=2)

print(f"  Saved: {json_path} ({len(geo_records)} locations with 2020-2025 data)")
print("=" * 70)
print("  DELHI PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
