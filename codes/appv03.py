import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pyproj
import joblib
import zipfile
import io
import urllib.request
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 0. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Urban Thermal Stress : Machine Learning Prediction of Mean Radiant Temperature in Urban Canyons",
    page_icon="🌡️",
    layout="wide"
)

TARGET = "TMRT"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NEIGHBORHOOD_SIZE = 5

feature_cols = [
    'Tair', 'FAR', 'Building_Height', 'Canopy_Cover', 'SVF', 'Tree_Height', 
    'NDVI', 'NDBI', 'LST', 'Shade_Index', 'Rain_mm', 'Min_Temp', 
    'Max_Temp', 'Mean_Temp', 'Min_Humidity', 'Max_Humidity', 
    'Mean_Wind_Kmph', 'Temp_Lag_1d', 'Temp_3d_avg', 'Rain_Rolling_Sum_3d'
]

# ==========================================
# 1. CACHED ARTIFACT & DATA LOADING
# ==========================================
@st.cache_resource
def load_artifacts_and_data():
    model = joblib.load(os.path.join(MODEL_DIR, "best_xgb_tmrt_model.joblib"))
    scaler_x = joblib.load(os.path.join(MODEL_DIR, "scaler_x.joblib"))
    train_medians = joblib.load(os.path.join(MODEL_DIR, "train_medians.joblib"))
    train_global_mean_tmrt = joblib.load(os.path.join(MODEL_DIR, "train_global_mean_tmrt.pkl"))
    train_pixel_mean_tmrt = joblib.load(os.path.join(MODEL_DIR, "train_pixel_mean_tmrt.joblib"))
    
    # Locate dataset
    dataset_path = "data/Tabular_Dataset_v2.parquet"
    if not os.path.exists(dataset_path):
        dataset_path = "Tabular_Dataset_v2.parquet"  # Fallback check
        
    df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path, encoding="latin1")
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    
    return model, scaler_x, train_medians, train_global_mean_tmrt, train_pixel_mean_tmrt, df

with st.spinner("📂 Loading model weights and dataset into memory..."):
    model, scaler_x, train_medians, train_global_mean_tmrt, train_pixel_mean_tmrt, df = load_artifacts_and_data()

pix_num_all = df["Pixel_ID"].astype(str).str.extract(r"(\d+)").astype(int)[0]
GRID_DIM = int(np.ceil(np.sqrt(pix_num_all.max() + 1)))

# Filter test set (2026)
df_test_raw = df[df["Year"] == 2026].copy()
df_test_raw["Pixel_Thermal_Anomaly"] = df_test_raw["Pixel_ID"].map(train_pixel_mean_tmrt).fillna(train_global_mean_tmrt) - train_global_mean_tmrt
feature_cols_extended = feature_cols + ["Pixel_Thermal_Anomaly"]

df_test_raw[feature_cols_extended] = df_test_raw[feature_cols_extended].fillna(train_medians)
df_test_raw[feature_cols_extended] = scaler_x.transform(df_test_raw[feature_cols_extended])

# ==========================================
# 2. SPATIAL CHIP ASSEMBLY & FEATURE EXTRACTION
# ==========================================
@st.cache_data
def prepare_test_inference(test_df_encoded):
    pad = NEIGHBORHOOD_SIZE // 2
    df_clean = test_df_encoded.dropna(subset=[TARGET]).copy()
    pix_num = df_clean["Pixel_ID"].astype(str).str.extract(r"(\d+)").astype(int)[0]
    df_clean["Grid_X"] = pix_num // GRID_DIM
    df_clean["Grid_Y"] = pix_num % GRID_DIM
    
    map_dim_x = max(GRID_DIM, df_clean["Grid_X"].max() + 1) + (2 * pad)
    map_dim_y = max(GRID_DIM, df_clean["Grid_Y"].max() + 1) + (2 * pad)
    
    X_chips, Y_targets, Coords, Dates = [], [], [], []
    for date, df_date in df_clean.groupby("Date"):
        feature_maps = np.zeros((map_dim_x, map_dim_y, len(feature_cols_extended)), dtype=np.float32)
        gx_arr = df_date["Grid_X"].values + pad
        gy_arr = df_date["Grid_Y"].values + pad
        feature_maps[gx_arr, gy_arr, :] = df_date[feature_cols_extended].values
        
        for gx, gy, target_val, orig_x, orig_y in zip(gx_arr, gy_arr, df_date[TARGET].values, df_date["Grid_X"].values, df_date["Grid_Y"].values):
            chip = feature_maps[gx-pad:gx+pad+1, gy-pad:gy+pad+1, :]
            X_chips.append(chip.transpose(2, 0, 1))
            Y_targets.append(target_val)
            Coords.append((orig_x, orig_y))
            Dates.append(date)
            
    X_spatial = np.array(X_chips, dtype=np.float32)
    N, C, H, W = X_spatial.shape
    center_h, center_w = H // 2, W // 2
    inner_flat = X_spatial[:, :, center_h-1:center_h+2, center_w-1:center_w+2].reshape(N, -1)
    center_vals = X_spatial[:, :, center_h, center_w]
    spatial_means = np.mean(X_spatial, axis=(2, 3))
    spatial_stds = np.std(X_spatial, axis=(2, 3))
    spatial_mins = np.min(X_spatial, axis=(2, 3))
    spatial_maxs = np.max(X_spatial, axis=(2, 3))
    
    X_eng = np.hstack([inner_flat, center_vals, spatial_means, spatial_stds, spatial_mins, spatial_maxs]).astype(np.float32)
    return X_eng, np.array(Y_targets, dtype=np.float32), np.array(Coords, dtype=np.float32), pd.Series(Dates), df_clean["Date"].unique()

with st.spinner("⚙️ Assembling spatial chips and running inference..."):
    X_te_eng, Y_te, coords_te, dates_te_series, unique_dates = prepare_test_inference(df_test_raw)
    preds = model.predict(X_te_eng)

# ==========================================
# 3. STREAMLIT UI LAYOUT & CONTROLS
# ==========================================
st.title("🌡️ Urban Thermal Stress : Machine Learning Prediction of Mean Radiant Temperature in Urban Canyons")
st.markdown("Interactive inference portal for spatial microclimate thermal stress monitoring and high-resolution map generation over **Nampally, Hyderabad**.")

# Sidebar Controls
st.sidebar.header("Control Panel")
selected_date = st.sidebar.selectbox("Select Inference Date (2026 Test Set)", unique_dates)

# Run Button added per request
run_button = st.sidebar.button("🚀 Run Inference & Render Map", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("Model Architecture: **Deep XGBoost Model**\nPerformance: **$R^2 \\approx 0.91$**")

# ==========================================
# 4. SATELLITE TILE FETCHING CACHED FUNCTION
# ==========================================
@st.cache_data
def fetch_esri_basemap():
    center_lon, center_lat = 78.4611, 17.3925
    buffer_m = 1500
    transformer_wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    center_x, center_y = transformer_wgs84_to_utm.transform(center_lon, center_lat)
    xmin, xmax = center_x - buffer_m, center_x + buffer_m
    ymin, ymax = center_y - buffer_m, center_y + buffer_m

    transformer_utm_to_wgs84 = pyproj.Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    west_deg, south_deg = transformer_utm_to_wgs84.transform(xmin, ymin)
    east_deg, north_deg = transformer_utm_to_wgs84.transform(xmax, ymax)

    transformer_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    wm_xmin, wm_ymin = transformer_3857.transform(west_deg, south_deg)
    wm_xmax, wm_ymax = transformer_3857.transform(east_deg, north_deg)

    def lonlat_to_tile(wm_x, wm_y, zoom):
        res = (2 * np.pi * 6378137 / 256) / (2**zoom)
        return int((wm_x + np.pi * 6378137) / (res * 256)), int((np.pi * 6378137 - wm_y) / (res * 256))

    zoom = 18
    x_min_t, y_min_t = lonlat_to_tile(wm_xmin, wm_ymax, zoom)
    x_max_t, y_max_t = lonlat_to_tile(wm_xmax, wm_ymin, zoom)

    stitched_img = []
    try:
        for y in range(y_min_t, y_max_t + 1):
            row_imgs = []
            for x in range(x_min_t, x_max_t + 1):
                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    tile = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                    row_imgs.append(tile)
            stitched_img.append(np.hstack(row_imgs))
        return cv2.cvtColor(np.vstack(stitched_img), cv2.COLOR_BGR2RGB)
    except Exception as e:
        fallback_canvas = np.full((800, 800, 3), 45, dtype=np.uint8)
        return fallback_canvas

# ==========================================
# 5. EXECUTION & RENDERING (TRIGGERED BY RUN BUTTON)
# ==========================================
if run_button:
    st.success(f"Running inference for date: {str(selected_date)[:10]}...")

    # Filter data for selected date
    date_mask = (dates_te_series == selected_date).values
    coords_subset = coords_te[date_mask]
    preds_subset = preds[date_mask]
    y_subset = Y_te[date_mask]

    # Calculate metrics for subset
    sub_r2 = r2_score(y_subset, preds_subset)
    sub_rmse = np.sqrt(mean_squared_error(y_subset, preds_subset))
    sub_mae = mean_absolute_error(y_subset, preds_subset)
    sub_mse = mean_squared_error(y_subset, preds_subset)

    # Metrics Summary Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Date Samples", f"{len(preds_subset):,}")
    col2.metric("Subset $R^2$ Score", f"{sub_r2:.3f}")
    col3.metric("RMSE (°C)", f"{sub_rmse:.2f} °C")
    col4.metric("MAE (°C)", f"{sub_mae:.2f} °C")

    st.markdown("---")

    with st.spinner("🌍 Fetching Esri Satellite Basemap & Rendering Thermal Overlay..."):
        full_satellite_rgb = fetch_esri_basemap()
        img_h, img_w = full_satellite_rgb.shape[:2]

        # Build 2D grid matrix
        grid_matrix = np.full((GRID_DIM, GRID_DIM), np.nan)
        for (gx, gy), val in zip(coords_subset, preds_subset):
            ix, iy = int(gx), int(gy)
            if 0 <= ix < GRID_DIM and 0 <= iy < GRID_DIM:
                grid_matrix[ix, iy] = val

        upscaled_preds = cv2.resize(np.nan_to_num(grid_matrix, nan=np.nanmean(grid_matrix)), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Plot overlay figure
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.imshow(full_satellite_rgb)
        im = ax.imshow(upscaled_preds, cmap='jet', alpha=0.25, vmin=30, vmax=60, extent=[0, img_w, img_h, 0], origin='upper')
        ax.set_title(f"Nampally Thermal Stress Overlay ($T_{{mrt}}$)\nDate: {str(selected_date)[:10]}", fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('$T_{{mrt}}$ (°C)', fontsize=10, fontweight='bold')
        fig.tight_layout()

        # Display Map in Streamlit
        st.pyplot(fig)

    st.markdown("### 📊 Evaluation Metrics Bar Graphs")
    
    # 4 Bar Graphs Layout underneath the heatmap with standardized styling & headroom for R²
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        fig_r2, ax_r2 = plt.subplots(figsize=(3, 3), dpi=150)
        ax_r2.bar(["R²"], [sub_r2], color="#1f77b4", width=0.5)
        ax_r2.set_ylim(0, 1.15)  # Increased limit from 1.0 to 1.15 to prevent text clipping
        ax_r2.set_ylabel("Score")
        ax_r2.set_title("R² Score", fontsize=10, fontweight='bold')
        for i, v in enumerate([sub_r2]):
            ax_r2.text(i, v + 0.03, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
        fig_r2.tight_layout()
        st.pyplot(fig_r2)

    with b_col2:
        fig_rmse, ax_rmse = plt.subplots(figsize=(3, 3), dpi=150)
        ax_rmse.bar(["RMSE"], [sub_rmse], color="#ff7f0e", width=0.5)
        ax_rmse.set_ylim(0, max(5.0, sub_rmse * 1.3))
        ax_rmse.set_ylabel("°C")
        ax_rmse.set_title("RMSE", fontsize=10, fontweight='bold')
        for i, v in enumerate([sub_rmse]):
            ax_rmse.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=9, fontweight='bold')
        fig_rmse.tight_layout()
        st.pyplot(fig_rmse)

    with b_col3:
        fig_mae, ax_mae = plt.subplots(figsize=(3, 3), dpi=150)
        ax_mae.bar(["MAE"], [sub_mae], color="#2ca02c", width=0.5)
        ax_mae.set_ylim(0, max(5.0, sub_mae * 1.3))
        ax_mae.set_ylabel("°C")
        ax_mae.set_title("MAE", fontsize=10, fontweight='bold')
        for i, v in enumerate([sub_mae]):
            ax_mae.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=9, fontweight='bold')
        fig_mae.tight_layout()
        st.pyplot(fig_mae)

    with b_col4:
        fig_mse, ax_mse = plt.subplots(figsize=(3, 3), dpi=150)
        ax_mse.bar(["MSE"], [sub_mse], color="#d62728", width=0.5)
        ax_mse.set_ylim(0, max(15.0, sub_mse * 1.3))
        ax_mse.set_ylabel("(°C)²")
        ax_mse.set_title("MSE", fontsize=10, fontweight='bold')
        for i, v in enumerate([sub_mse]):
            ax_mse.text(i, v + 0.2, f"{v:.2f}", ha='center', fontsize=9, fontweight='bold')
        fig_mse.tight_layout()
        st.pyplot(fig_mse)

    st.markdown("---")

    # Download Bundle Option
    st.markdown("### 📦 Export Assets")
    if st.button("Generate Downloadable Map Bundle (.zip)"):
        base_map_path = os.path.join(OUTPUT_DIR, "nampally_helicopter_basemap.png")
        combined_map_path = os.path.join(OUTPUT_DIR, "nampally_real_tmrt_overlay_jet.png")
        
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(full_satellite_rgb)
        plt.axis('off')
        plt.savefig(base_map_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        fig2, ax2 = plt.subplots(figsize=(10, 10), dpi=300)
        ax2.imshow(full_satellite_rgb)
        im2 = ax2.imshow(upscaled_preds, cmap='jet', alpha=0.22, vmin=30, vmax=60, extent=[0, img_w, img_h, 0], origin='upper')
        ax2.set_title(f"Nampally Real TMRT Stress Overlay\n(Date: {str(selected_date)[:10]})", fontsize=13, fontweight='bold', pad=12)
        ax2.axis('off')
        fig2.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        fig2.tight_layout()
        fig2.savefig(combined_map_path, bbox_inches='tight', dpi=300)
        plt.close(fig2)

        zip_path = os.path.join(OUTPUT_DIR, "nampally_jet_maps_bundle.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(base_map_path, arcname="nampally_helicopter_basemap.png")
            zipf.write(combined_map_path, arcname="nampally_real_tmrt_overlay_jet.png")

        with open(zip_path, "rb") as fp:
            st.download_button(
                label="📥 Download Map Bundle Zip",
                data=fp,
                file_name="nampally_jet_maps_bundle.zip",
                mime="application/zip"
            )
        st.success("Bundle generated successfully!")
else:
    st.info("👈 Select your date in the sidebar and click **'Run Inference & Render Map'** to begin.")