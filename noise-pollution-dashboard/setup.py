import os
import shutil
import json
import re

# Configuration
BASE_DIR = os.getcwd()
TARGET_DIR = os.path.join(BASE_DIR, "noise-pollution-dashboard")

STRUCTURE = [
    "data/raw",
    "data/processed",
    "data/geo",
    "models/delhi",
    "models/chennai",
    "notebooks",
    "outputs/plots/delhi",
    "outputs/plots/chennai",
    "scripts/delhi",
    "scripts/chennai",
    "scripts/shared",
    "frontend/assets"
]

def create_structure():
    print("Creating folder structure...")
    for folder in STRUCTURE:
        os.makedirs(os.path.join(TARGET_DIR, folder), exist_ok=True)
    print("Folder structure created.")

def move_files():
    print("Moving files...")
    
    # Data - Raw
    if os.path.exists("delhi_noise_data.xlsx"):
        shutil.move("delhi_noise_data.xlsx", os.path.join(TARGET_DIR, "data/raw/delhi_noise_data.xlsx"))
    
    # Data - Processed
    processed_files = ["delhi_noise_2020_2024.csv", "delhi_noise_cleaned.csv", "chennai_noise_2020_2024.csv"]
    for f in processed_files:
        if os.path.exists(f):
            shutil.move(f, os.path.join(TARGET_DIR, "data/processed", f))
            
    # Data - Geo
    geo_mapping = {
        "locations_geo.json": "delhi_locations_geo.json",
        "chennai_locations_geo.json": "chennai_locations_geo.json",
        "all_cities_locations_geo.json": "all_cities_locations_geo.json",
        "locations_geo.json": "delhi_locations_geo.json"
    }
    for src, dst in geo_mapping.items():
        if os.path.exists(src):
            shutil.move(src, os.path.join(TARGET_DIR, "data/geo", dst))
            
    # Models
    if os.path.exists("noise_zone_model.pkl"):
        shutil.move("noise_zone_model.pkl", os.path.join(TARGET_DIR, "models/delhi/delhi_noise_model.pkl"))
    if os.path.exists("chennai_noise_model.pkl"):
        shutil.move("chennai_noise_model.pkl", os.path.join(TARGET_DIR, "models/chennai/chennai_noise_model.pkl"))
        
    # Notebooks
    if os.path.exists("noise_pollution_analysis.py"):
        with open("noise_pollution_analysis.py", "r", encoding="utf-8") as f:
            content = f.read()
        nb = {
            "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content.splitlines(True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 4
        }
        with open(os.path.join(TARGET_DIR, "notebooks/noise_pollution_analysis.ipynb"), "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        os.remove("noise_pollution_analysis.py")
        
    # Plots - Delhi
    delhi_plots = ["feature_importance.png", "confusion_matrix.png", "rf_confusion_matrix.png", "xgb_confusion_matrix.png", "noise_trend_2020_2024.png"]
    for f in delhi_plots:
        if os.path.exists(f):
            shutil.move(f, os.path.join(TARGET_DIR, "outputs/plots/delhi", f))
            
    # Plots - Chennai
    chennai_plots = ["chennai_feature_importance.png", "chennai_rf_cm.png", "chennai_xgb_cm.png", "chennai_noise_trend.png", "chennai_zonewise_trend.png"]
    for f in chennai_plots:
        if os.path.exists(f):
            shutil.move(f, os.path.join(TARGET_DIR, "outputs/plots/chennai", f))
            
    # Scripts
    if os.path.exists("main.py"):
        shutil.move("main.py", os.path.join(TARGET_DIR, "scripts/delhi/main.py"))
    if os.path.exists("chennai_main.py"):
        shutil.move("chennai_main.py", os.path.join(TARGET_DIR, "scripts/chennai/chennai_main.py"))
    if os.path.exists("geocode_merge.py"):
        shutil.move("geocode_merge.py", os.path.join(TARGET_DIR, "scripts/shared/geocode_merge.py"))
        
    # Frontend
    frontend_files = ["index.html", "style.css", "script.js"]
    for f in frontend_files:
        if os.path.exists(f):
            shutil.move(f, os.path.join(TARGET_DIR, "frontend", f))
            
    # Assets - move other frontend stuff
    others = ["delhi_noise_map.html", "chennai_index.html", "chennai_style.css", "chennai_script.js"]
    for f in others:
        if os.path.exists(f):
            shutil.move(f, os.path.join(TARGET_DIR, "frontend/assets", f))

def update_python_paths():
    print("Updating Python script paths...")
    
    # Delhi Main
    p = os.path.join(TARGET_DIR, "scripts/delhi/main.py")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: content = f.read()
        content = content.replace("pd.read_csv('delhi_noise_2020_2024.csv')", "pd.read_csv('../../data/processed/delhi_noise_2020_2024.csv')")
        content = content.replace("'feature_importance.png'", "'../../outputs/plots/delhi/feature_importance.png'")
        content = content.replace("'rf_confusion_matrix.png'", "'../../outputs/plots/delhi/rf_confusion_matrix.png'")
        content = content.replace("'xgb_confusion_matrix.png'", "'../../outputs/plots/delhi/xgb_confusion_matrix.png'")
        content = content.replace("'noise_trend_2020_2024.png'", "'../../outputs/plots/delhi/noise_trend_2020_2024.png'")
        content = content.replace("'noise_zone_model.pkl'", "'../../models/delhi/delhi_noise_model.pkl'")
        content = content.replace("'locations_geo.json'", "'../../data/geo/delhi_locations_geo.json'")
        content = re.sub(r'os\.chdir\(BASE_DIR\)', '# os.chdir(BASE_DIR)', content)
        with open(p, "w", encoding="utf-8") as f: f.write(content)

    # Chennai Main
    p = os.path.join(TARGET_DIR, "scripts/chennai/chennai_main.py")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: content = f.read()
        content = content.replace("pd.read_csv('chennai_noise_2020_2024.csv')", "pd.read_csv('../../data/processed/chennai_noise_2020_2024.csv')")
        content = content.replace("'chennai_feature_importance.png'", "'../../outputs/plots/chennai/chennai_feature_importance.png'")
        content = content.replace("'chennai_rf_cm.png'", "'../../outputs/plots/chennai/chennai_rf_cm.png'")
        content = content.replace("'chennai_xgb_cm.png'", "'../../outputs/plots/chennai/chennai_xgb_cm.png'")
        content = content.replace("'chennai_noise_trend.png'", "'../../outputs/plots/chennai/chennai_noise_trend.png'")
        content = content.replace("'chennai_zonewise_trend.png'", "'../../outputs/plots/chennai/chennai_zonewise_trend.png'")
        content = content.replace("'chennai_noise_model.pkl'", "'../../models/chennai/chennai_noise_model.pkl'")
        content = content.replace("'chennai_locations_geo.json'", "'../../data/geo/chennai_locations_geo.json'")
        content = re.sub(r'os\.chdir\(BASE_DIR\)', '# os.chdir(BASE_DIR)', content)
        with open(p, "w", encoding="utf-8") as f: f.write(content)

    # Shared Geocode Merge
    p = os.path.join(TARGET_DIR, "scripts/shared/geocode_merge.py")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: content = f.read()
        content = content.replace("pd.read_csv('delhi_noise_2020_2024.csv')", "pd.read_csv('../../data/processed/delhi_noise_2020_2024.csv')")
        content = content.replace("pd.read_csv('chennai_noise_2020_2024.csv')", "pd.read_csv('../../data/processed/chennai_noise_2020_2024.csv')")
        content = content.replace("'all_cities_locations_geo.json'", "'../../data/geo/all_cities_locations_geo.json'")
        content = re.sub(r'os\.chdir\(BASE_DIR\)', '# os.chdir(BASE_DIR)', content)
        with open(p, "w", encoding="utf-8") as f: f.write(content)

def update_frontend_paths():
    print("Updating frontend paths...")
    
    # script.js
    p = os.path.join(TARGET_DIR, "frontend/script.js")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: content = f.read()
        content = content.replace("fetch('all_cities_locations_geo.json')", "fetch('../data/geo/all_cities_locations_geo.json')")
        with open(p, "w", encoding="utf-8") as f: f.write(content)
        
def generate_requirements():
    print("Generating requirements.txt...")
    reqs = "pandas\nnumpy\nscikit-learn\nxgboost\ngeopy\nmatplotlib\nseaborn\njoblib\nopenpyxl\n"
    with open(os.path.join(TARGET_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(reqs)

def generate_readme():
    print("Generating README.md...")
    readme = """# India Noise Pollution Smart City Dashboard

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
"""
    with open(os.path.join(TARGET_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

if __name__ == "__main__":
    create_structure()
    move_files()
    update_python_paths()
    update_frontend_paths()
    generate_requirements()
    generate_readme()
    print("\\nReorganization complete! Your project is now in the 'noise-pollution-dashboard' folder.")
