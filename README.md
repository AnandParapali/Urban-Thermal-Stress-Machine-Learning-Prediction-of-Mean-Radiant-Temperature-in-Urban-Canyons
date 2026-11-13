## 1. Installation & Setup Guide

Follow these instructions to configure your local development environment and run the system locally.

### Prerequisites
- Python 3.10 or higher
- Pip package manager

### Step-by-Step Installation
1. Clone the repository:
   git clone https://github.com/AnandParapali/Urban-Thermal-Stress-Machine-Learning-Prediction-of-Mean-Radiant-Temperature-in-Urban-Canyons.git
   cd Urban-Thermal-Stress-Machine-Learning-Prediction-of-Mean-Radiant-Temperature-in-Urban-Canyons

2. Install all required dependencies:
   pip install -r requirements.txt

3. Verify model artifacts and parquet datasets are correctly placed inside their respective directories (`models/` and `data/`).

4. Launch the interactive visualization dashboard:
   streamlit run app.py

---

## 2. Technical Documentation & System Overview

### Core Architecture & Methodology
- **Feature Engineering:** Combines 2D remote sensing indices (NDVI, NDBI, Built-Vegetation ratios) with 3D structural parameters (building heights, Floor Area Ratio, Sky View Factors) extracted via Google Earth Engine and ESRI imagery processing.
- **Ground-Truth Target Generation:** Leverages Copernicus GLO-30 Digital Elevation Models combined with meteorological drivers processed through QGIS and the UMEP/SOLWEIG framework to establish accurate Mean Radiant Temperature ($T_{mrt}$) reference grids.
- **Surrogate Modeling Engine:** Employs optimized ensemble learning algorithms (**XGBoost** and GPU-accelerated **Random Forest**) trained across historical temporal windows (2020–2026).
- **Performance Benchmark:** The optimized regularized XGBoost configuration achieves an outstanding coefficient of determination ($R^2$) of **0.910654** with an RMSE of **3.4557 °C** on holdout test evaluations, delivering a **6.8x operational speedup** over traditional physical solvers.

---

## 3. Version History & Commit Practices

This repository follows rigorous version control standards, maintaining clean incremental commits across development phases:
- **Initial Commit:** Repository initialization, directory scaffolding, and environment configuration files.
- **Data Engineering Phase:** Process of acquiring satellite data, spatial resampling to 30m UTM grids, and Apache Parquet tabular compilation.
- **Deployment Phase:** Integration of the Streamlit interactive front-end (`app.py`), dynamic spatial inference triggers, and automated metric plotting modules.

