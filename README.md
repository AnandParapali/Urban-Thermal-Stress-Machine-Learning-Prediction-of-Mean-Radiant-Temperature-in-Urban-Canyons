## 1. Installation & Setup Guide

Follow these instructions to configure your local development environment and run the system locally.

### Prerequisites
- Python 3.10 or higher
- Pip package manager

### Step-by-Step Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnandParapali/Urban-Thermal-Stress-Machine-Learning-Prediction-of-Mean-Radiant-Temperature-in-Urban-Canyons.git
   cd Urban-Thermal-Stress-Machine-Learning-Prediction-of-Mean-Radiant-Temperature-in-Urban-Canyons
   ```

2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify model artifacts and parquet datasets are correctly placed inside their respective directories (`models/` and `data/`).

4. Launch the interactive visualization dashboard:
  ```bash
   streamlit run appv08.py
   ```

---

## 2. Technical Documentation & System Overview

### Core Architecture & Methodology
- **Feature Engineering:** Combines 2D remote sensing indices with 3D structural parameters (building heights, Floor Area Ratio, Sky View Factors) extracted via Google Earth Engine and ESRI imagery processing.
- **Ground-Truth Target Generation:** Leverages Copernicus GLO-30 Digital Elevation Models combined with meteorological drivers processed through QGIS and the UMEP/SOLWEIG framework to establish accurate Mean Radiant Temperature ($T_{mrt}$) reference grids.
- **Surrogate Modeling Engine:** Employs optimized learning algorithms (**XGBoost** and **Random Forest**) trained across historical temporal window.
- **Performance Benchmark:** The optimized regularized XGBoost configuration achieves an outstanding coefficient of determination ($R^2$) and  RMSE  on  evaluations.



