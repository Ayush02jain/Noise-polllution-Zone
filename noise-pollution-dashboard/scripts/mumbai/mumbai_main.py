"""
Noise Pollution Zone Forecasting for Mumbai (2020-2025)
=======================================================
Steps 1-5: Preprocessing, ML Regression Forecasting (RF & XGBoost),
           Trend Analysis + 2025 Forecast, CPCB Compliance Report, JSON Export
Uses: data/processed/mumbai_noise_2020_2024.csv
Outputs: models/mumbai/mumbai_noise_model.pkl, plots, cpcb_compliance_report.csv, mumbai_locations_geo.json
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

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'plots', 'mumbai')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'mumbai')
DATA_GEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'geo')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_GEO_DIR, exist_ok=True)

print("=" * 70)
print("  NOISE POLLUTION FORECASTING & REGRESSION - MUMBAI")
print("=" * 70)

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 1: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'mumbai_noise_2020_2024.csv')
df = pd.read_csv(csv_path)
print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

numeric_cols = ['Noise_Day_dB', 'Noise_Night_dB', 'CPCB_Day_Std_dB',
                'CPCB_Night_Std_dB', 'Excess_Day_dB', 'Excess_Night_dB']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

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
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

le_zone_type = LabelEncoder()
df['Zone_Type_Encoded'] = le_zone_type.fit_transform(df['Zone_Type'])

le_loc = LabelEncoder()
df['Location_Encoded'] = le_loc.fit_transform(df['Location'])

train_mask = df['Year'] < 2024
loc_hist_mean = df[train_mask].groupby('Location')['Noise_Day_dB'].mean().to_dict()
loc_hist_mean_night = df[train_mask].groupby('Location')['Noise_Night_dB'].mean().to_dict()

overall_mean_day = df[train_mask]['Noise_Day_dB'].mean()
overall_mean_night = df[train_mask]['Noise_Night_dB'].mean()

df['Hist_Loc_Mean_Day'] = df['Location'].map(loc_hist_mean).fillna(overall_mean_day)
df['Hist_Loc_Mean_Night'] = df['Location'].map(loc_hist_mean_night).fillna(overall_mean_night)

feature_names = ['Zone_Type_Encoded', 'Location_Encoded', 'Year', 'Month',
                 'Season', 'Month_Sin', 'Month_Cos', 'Hist_Loc_Mean_Day']

print(f"  Features ({len(feature_names)}): {feature_names}")

# ============================================================================
# STEP 2: TIME-BASED SPLIT & REGRESSION MODELING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: TIME-BASED SPLIT & REGRESSION MODELING")
print("=" * 70)

train_df = df[df['Year'] < 2024]
test_df = df[df['Year'] == 2024]

X_train = train_df[feature_names].values
y_train = train_df['Noise_Day_dB'].values
X_test = test_df[feature_names].values
y_test = test_df['Noise_Day_dB'].values

print(f"  Train: {len(train_df)}, Test (2024): {len(test_df)}")

# RF
print("\n  --- Training Random Forest Regressor ---")
rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_split=4,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
print(f"    RF Regressor  ->  RMSE: {rf_rmse:.3f} dB | MAE: {rf_mae:.3f} dB | R2: {rf_r2:.3f} | MAPE: {rf_mape:.2f}%")

# XGB
print("  --- Training XGBoost Regressor ---")
xgb_reg = XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
print(f"    XGB Regressor ->  RMSE: {xgb_rmse:.3f} dB | MAE: {xgb_mae:.3f} dB | R2: {xgb_r2:.3f} | MAPE: {xgb_mape:.2f}%")

best_name = 'Random Forest' if rf_rmse <= xgb_rmse else 'XGBoost'
best_model = rf_reg if rf_rmse <= xgb_rmse else xgb_reg
best_rmse = min(rf_rmse, xgb_rmse)
best_mae = min(rf_mae, xgb_mae)
best_r2 = max(rf_r2, xgb_r2)
best_pred = rf_pred if rf_rmse <= xgb_rmse else xgb_pred

print(f"\n  Selected Best Model: {best_name} (RMSE: {best_rmse:.3f} dB, R2: {best_r2:.3f})")

# Night Model
rf_night = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
feature_names_night = ['Zone_Type_Encoded', 'Location_Encoded', 'Year', 'Month',
                       'Season', 'Month_Sin', 'Month_Cos', 'Hist_Loc_Mean_Night']
X_train_night = train_df[feature_names_night].values
y_train_night = train_df['Noise_Night_dB'].values
rf_night.fit(X_train_night, y_train_night)

# Save
model_save_path = os.path.join(MODELS_DIR, 'mumbai_noise_model.pkl')
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
print(f"  Saved model bundle: {model_save_path}")

# Feature importance plot
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': best_model.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.summer(np.linspace(0.3, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='white', height=0.6)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title(f'Mumbai Noise Forecasting - {best_name} Feature Importance', fontsize=14, fontweight='bold', pad=15)
for bar, val in zip(bars, feat_imp['Importance']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance.png")

# Actual vs Predicted plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_pred, alpha=0.6, color='#16a085', edgecolors='white', s=60, label='2024 Test Points')
min_val = min(y_test.min(), best_pred.min()) - 2
max_val = max(y_test.max(), best_pred.max()) + 2
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Perfect Fit (y=x)')
ax.set_xlabel('Actual Noise Day (dB) - 2024', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Noise Day (dB) - 2024', fontsize=12, fontweight='bold')
ax.set_title(f'Mumbai 2024 Out-of-Time Test Set\nRMSE: {best_rmse:.2f} dB | MAE: {best_mae:.2f} dB | R²: {best_r2:.2f}', fontsize=14, fontweight='bold')
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

unique_locs = df[['Location', 'Zone_Type', 'CPCB_Day_Std_dB', 'CPCB_Night_Std_dB']].drop_duplicates()

forecast_rows = []
for _, row in unique_locs.iterrows():
    loc = row['Location']
    zt = row['Zone_Type']
    std_day = row['CPCB_Day_Std_dB']
    std_night = row['CPCB_Night_Std_dB']
    
    zt_enc = le_zone_type.transform([zt])[0]
    loc_enc = le_loc.transform([loc])[0]
    hist_day = loc_hist_mean.get(loc, overall_mean_day)
    hist_night = loc_hist_mean_night.get(loc, overall_mean_night)

    for m in range(1, 13):
        season = map_season(m)
        m_sin = np.sin(2 * np.pi * m / 12.0)
        m_cos = np.cos(2 * np.pi * m / 12.0)
        
        feat_vec_day = np.array([[zt_enc, loc_enc, 2025, m, season, m_sin, m_cos, hist_day]])
        pred_day = float(best_model.predict(feat_vec_day)[0])
        
        feat_vec_night = np.array([[zt_enc, loc_enc, 2025, m, season, m_sin, m_cos, hist_night]])
        pred_night = float(rf_night.predict(feat_vec_night)[0])
        
        excess_day = pred_day - std_day
        excess_night = pred_night - std_night
        
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
            'City': 'Mumbai',
            'Zone_Type': zt,
            'CPCB_Day_Std_dB': std_day,
            'CPCB_Night_Std_dB': std_night,
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
print(f"  Generated {len(df_forecast_2025)} forecasted monthly records for 2025.")

df_combined = pd.concat([df, df_forecast_2025], ignore_index=True)
yearly = df_combined.groupby('Year')[['Noise_Day_dB', 'Noise_Night_dB']].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
hist_yearly = yearly[yearly['Year'] <= 2024]
ax.plot(hist_yearly['Year'], hist_yearly['Noise_Day_dB'], marker='o', ms=9, lw=2.5,
        color='#16a085', label='Historical Day Noise (dB)', zorder=5)
ax.plot(hist_yearly['Year'], hist_yearly['Noise_Night_dB'], marker='s', ms=9, lw=2.5,
        color='#2980b9', label='Historical Night Noise (dB)', zorder=5)

fc_yearly = yearly[yearly['Year'] >= 2024]
ax.plot(fc_yearly['Year'], fc_yearly['Noise_Day_dB'], marker='o', ms=9, lw=2.5, ls='--',
        color='#27ae60', label='ML Forecasted Day (2025)', zorder=5)
ax.plot(fc_yearly['Year'], fc_yearly['Noise_Night_dB'], marker='s', ms=9, lw=2.5, ls='--',
        color='#1abc9c', label='ML Forecasted Night (2025)', zorder=5)

for _, r in yearly.iterrows():
    ax.text(r['Year'], r['Noise_Day_dB'] + 0.35, f"{r['Noise_Day_dB']:.1f}",
            ha='center', fontsize=9, color='#16a085', fontweight='bold')
    ax.text(r['Year'], r['Noise_Night_dB'] - 0.55, f"{r['Noise_Night_dB']:.1f}",
            ha='center', fontsize=9, color='#2980b9', fontweight='bold')

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Noise (dB)', fontsize=13, fontweight='bold')
ax.set_title('Mumbai Noise Pollution Trend & 2025 ML Forecast', fontsize=15, fontweight='bold')
ax.set_xticks(yearly['Year'])
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'noise_trend_2020_2025.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: noise_trend_2020_2025.png")

# ============================================================================
# STEP 4: CPCB COMPLIANCE REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 4: CPCB COMPLIANCE REPORT")
print("=" * 70)

compliance = df.groupby('Zone_Type').agg(
    CPCB_Day_Standard=('CPCB_Day_Std_dB', 'first'),
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
compliance.to_csv(os.path.join(PLOTS_DIR, 'cpcb_compliance_report.csv'), index=False)
print("  Saved: cpcb_compliance_report.csv")

# ============================================================================
# STEP 5: GEOCODING & JSON EXPORT
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 5: GEOCODING & JSON EXPORT")
print("=" * 70)

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

loc_overall = df_combined.groupby('Location').agg(
    Avg_Day=('Noise_Day_dB', 'mean'),
    Avg_Night=('Noise_Night_dB', 'mean'),
    Avg_Excess_Day=('Excess_Day_dB', 'mean'),
    CPCB_Std_Day=('CPCB_Day_Std_dB', 'first'),
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
    lat, lon = MUMBAI_COORDS.get(loc, (19.0760, 72.8777))
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
        'City': 'Mumbai',
        'Latitude': ov['Latitude'],
        'Longitude': ov['Longitude'],
        'Avg_Day': round(ov['Avg_Day'], 2),
        'Avg_Night': round(ov['Avg_Night'], 2),
        'Avg_Excess_Day': round(ov['Avg_Excess_Day'], 2),
        'DPCC_Std_Day': int(ov['CPCB_Std_Day']),
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

json_path = os.path.join(DATA_GEO_DIR, 'mumbai_locations_geo.json')
with open(json_path, 'w') as f:
    json.dump(output_json, f, indent=2)

print(f"  Saved: {json_path} ({len(geo_records)} locations)")
print("=" * 70)
print("  MUMBAI PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
