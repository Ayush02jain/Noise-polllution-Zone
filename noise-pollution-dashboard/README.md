# India Noise Pollution Smart City Dashboard

## Project Overview
This project provides an interactive dashboard and machine learning analysis of noise pollution levels across India's major cities, specifically Delhi and Chennai, from 2020 to 2024. It classifies noise zones and visualizes trends to support smart city planning.

## Folder Structure
```
noise-pollution-dashboard/
│
├── data/
│   ├── raw/
│   │   ├── delhi_noise_data.xlsx
│   │   └── chennai_noise_raw.csv
│   ├── processed/
│   │   ├── delhi_noise_2020_2024.csv
│   │   ├── delhi_noise_cleaned.csv
│   │   └── chennai_noise_2020_2024.csv
│   └── geo/
│       ├── delhi_locations_geo.json
│       ├── chennai_locations_geo.json
│       └── all_cities_locations_geo.json
│
├── models/
│   ├── delhi/
│   │   └── delhi_noise_model.pkl
│   └── chennai/
│       └── chennai_noise_model.pkl
│
├── notebooks/
│   └── noise_pollution_analysis.ipynb
│
├── outputs/
│   ├── plots/
│   │   ├── delhi/
│   │   │   ├── feature_importance.png
│   │   │   ├── confusion_matrix.png
│   │   │   └── noise_trend_2020_2024.png
│   │   └── chennai/
│   │       ├── chennai_feature_importance.png
│   │       ├── chennai_rf_cm.png
│   │       ├── chennai_xgb_cm.png
│   │       ├── chennai_noise_trend.png
│   │       └── chennai_zonewise_trend.png
│
├── scripts/
│   ├── delhi/
│   │   └── main.py
│   ├── chennai/
│   │   └── chennai_main.py
│   └── shared/
│       └── geocode_merge.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
│       └── (any icons or images used in the dashboard)
│
├── requirements.txt
└── README.md
```

## How to Run
1. **Install Requirements**: Run `pip install -r requirements.txt`.
2. **Run Delhi Analysis**: Navigate to `scripts/delhi/` and run `python main.py`.
3. **Run Chennai Analysis**: Navigate to `scripts/chennai/` and run `python chennai_main.py`.
4. **Run Geocode Merge**: Navigate to `scripts/shared/` and run `python geocode_merge.py`.
5. **Open Frontend**: Open `frontend/index.html` in a web browser (or serve with a local server).

## Cities Covered
- **Delhi**: 41 locations monitored.
- **Chennai**: 10 locations monitored.

## Data Sources
- Central Pollution Control Board (CPCB)
- Delhi Pollution Control Committee (DPCC)
- Various research papers and smart city reports.

## Tech Stack
- **Backend / Analysis**: Python, Scikit-learn, XGBoost, Pandas, Matplotlib.
- **Frontend**: HTML5, CSS3, JavaScript, Leaflet.js, Chart.js.
