# 🏙️ India Noise Pollution Smart City Dashboard

> **Multi-city interactive dashboard** for visualizing and classifying noise pollution zones across **Delhi (41 locations)** and **Chennai (10 locations)** from **2020 to 2024**, powered by machine learning and real-time geospatial visualization.

---

## 📋 Project Overview

This project analyzes CPCB noise monitoring data across two major Indian cities. It trains supervised classification models (Random Forest & XGBoost) to categorize locations into noise pollution zones (Low / Moderate / High / Critical), and presents the results through an interactive dark-themed web dashboard built with Leaflet.js and Chart.js.

---

## 📂 Folder Structure

```
noise-pollution-dashboard/
│
├── data/
│   ├── raw/                          # Original unprocessed datasets
│   │   ├── delhi_noise_data.xlsx
│   │   └── chennai_noise_raw.csv
│   ├── processed/                    # Cleaned & ready-to-use datasets
│   │   ├── delhi_noise_2020_2024.csv
│   │   ├── delhi_noise_cleaned.csv
│   │   └── chennai_noise_2020_2024.csv
│   └── geo/                          # Geocoded JSON for frontend
│       ├── delhi_locations_geo.json
│       ├── chennai_locations_geo.json
│       └── all_cities_locations_geo.json
│
├── models/
│   ├── delhi/
│   │   └── delhi_noise_model.pkl     # Best ML model for Delhi
│   └── chennai/
│       └── chennai_noise_model.pkl   # Best ML model for Chennai
│
├── notebooks/
│   └── noise_pollution_analysis.py   # Exploratory analysis script
│
├── outputs/
│   └── plots/
│       ├── delhi/                    # Delhi visualizations
│       │   ├── feature_importance.png
│       │   ├── confusion_matrix.png
│       │   ├── rf_confusion_matrix.png
│       │   ├── xgb_confusion_matrix.png
│       │   └── noise_trend_2020_2024.png
│       └── chennai/                  # Chennai visualizations
│           ├── chennai_feature_importance.png
│           ├── chennai_rf_cm.png
│           ├── chennai_xgb_cm.png
│           ├── chennai_noise_trend.png
│           └── chennai_zonewise_trend.png
│
├── scripts/
│   ├── delhi/
│   │   └── main.py                   # Delhi preprocessing + ML pipeline
│   ├── chennai/
│   │   └── chennai_main.py           # Chennai preprocessing + ML pipeline
│   └── shared/
│       └── geocode_merge.py          # Merge both cities + geocode
│
├── frontend/
│   ├── index.html                    # Dashboard structure
│   ├── style.css                     # Dark smart-city theme
│   ├── script.js                     # Leaflet map + Chart.js + filters
│   └── assets/                       # Icons and images
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 How to Run

### 1. Install Python Dependencies

```bash
cd noise-pollution-dashboard
pip install -r requirements.txt
```

### 2. Run Delhi Pipeline

```bash
python scripts/delhi/main.py
```

Generates: model (`models/delhi/`), plots (`outputs/plots/delhi/`), and `data/geo/delhi_locations_geo.json`.

### 3. Run Chennai Pipeline

```bash
python scripts/chennai/chennai_main.py
```

Generates: model (`models/chennai/`), plots (`outputs/plots/chennai/`), and `data/geo/chennai_locations_geo.json`.

### 4. Run Multi-City Merge & Geocode

```bash
python scripts/shared/geocode_merge.py
```

Generates: `data/geo/all_cities_locations_geo.json` (51 locations combined).

### 5. Launch the Dashboard

```bash
cd noise-pollution-dashboard
python -m http.server 8080
```

Open **http://localhost:8080/frontend/index.html** in your browser.

---

## 🏙️ Cities Covered

| City    | Locations | Zone Types                                | Data Period |
|---------|-----------|-------------------------------------------|-------------|
| Delhi   | 41        | Commercial, Industrial, Residential       | 2020–2024   |
| Chennai | 10        | Commercial, Industrial, Residential, Silence | 2020–2024 |

**Total:** 51 CPCB monitoring stations, 3,060 monthly records.

---

## 📊 Data Sources

- **Central Pollution Control Board (CPCB)** — National ambient noise monitoring data
- **Delhi Pollution Control Committee (DPCC)** — Delhi-specific noise monitoring
- **Tamil Nadu Pollution Control Board (TNPCB)** — Chennai noise monitoring
- **Research Papers** — Baseline noise levels and zone classification thresholds

---

## 🛠️ Tech Stack

| Layer     | Technology                                     |
|-----------|------------------------------------------------|
| Data      | Python, Pandas, NumPy                          |
| ML Models | Scikit-learn (Random Forest), XGBoost          |
| Geocoding | Geopy (Nominatim/OpenStreetMap)                |
| Plots     | Matplotlib, Seaborn                            |
| Frontend  | HTML5, CSS3, JavaScript (ES5+)                 |
| Map       | Leaflet.js (CartoDB Dark Matter tiles)         |
| Charts    | Chart.js 4.x                                   |
| Fonts     | Google Fonts (Inter)                           |

---

## ✨ Dashboard Features

- 🌍 **Multi-city view** — India-level zoom showing Delhi + Chennai simultaneously
- 🎨 **City-coded markers** — Blue borders (Delhi) / Orange borders (Chennai)
- 🔍 **4 filter dimensions** — City toggle, Year, Zone Type, Zone Category
- 📈 **Side-by-side trend charts** — Delhi (blue) and Chennai (orange)
- 📊 **Live statistics** — Updates on every filter change (count, avg dB, most polluted)
- 🗺️ **Auto-zoom** — City toggle automatically zooms to the selected city
- 🌙 **Dark smart-city theme** — Professional dark mode with glassmorphism

---

## 📜 License

This project is for educational and research purposes. Data sourced from public government agencies.
