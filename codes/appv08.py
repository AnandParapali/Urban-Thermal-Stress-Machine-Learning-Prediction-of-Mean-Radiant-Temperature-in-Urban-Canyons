import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pyproj
import joblib
import urllib.request

# ==========================================
# 0. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Urban Microclimate TMRT Surrogate",
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
    
    dataset_path = "data/Tabular_Dataset_v2.parquet"
    if not os.path.exists(dataset_path):
        dataset_path = "Tabular_Dataset_v2.parquet"
        
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
    return X_eng, np.array(Y_targets, dtype=np.float32), np.array(Coords, dtype=np.float32), pd.Series(Dates), np.sort(df_clean["Date"].unique())

with st.spinner("⚙️ Assembling spatial chips and running surrogate inference..."):
    X_te_eng, Y_te, coords_te, dates_te_series, unique_dates = prepare_test_inference(df_test_raw)
    preds = model.predict(X_te_eng)

# ==========================================
# 3. SATELLITE TILE FETCHING
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
    except Exception:
        return np.full((800, 800, 3), 45, dtype=np.uint8)

# ==========================================
# 4. STREAMLIT UI LAYOUT (LEFT MAP + RIGHT CONTROLS)
# ==========================================
st.title("🌡️ Urban Microclimate $T_{mrt}$ Surrogate Model Dashboard")
st.markdown("Interactive inference portal for spatial microclimate thermal stress monitoring and high-resolution map generation over **Nampally, Hyderabad**.")

# Left sidebar for static architecture metadata
st.sidebar.header("System Info")
st.sidebar.info("Model Architecture: **Deep XGBoost Surrogate**\n\nLong-Term $R^2$: **0.9107**")

# Main layout: Left Column (Map & Metrics) | Right Column (3 Mapping Modes)
main_col, right_panel = st.columns([3, 1.2], gap="large")

with right_panel:
    st.subheader("🗺️ Mapping Mode")
    mapping_mode = st.radio(
        "Select Temporal Aggregation:",
        options=["1. Monthwise Mapping", "2. Day Wise Mapping", "3. Custom Mapping"],
        index=1
    )
    st.markdown("---")

    dates_dt = pd.to_datetime(dates_te_series)
    unique_dates_dt = pd.to_datetime(unique_dates)

    if mapping_mode == "1. Monthwise Mapping":
        st.markdown("**Select Target Month**")
        available_months = sorted(unique_dates_dt.strftime("%Y-%m").unique())
        selected_month = st.selectbox("Available Year-Month:", available_months)
        
        date_mask = (dates_dt.dt.strftime("%Y-%m") == selected_month).values
        map_title_label = f"Monthly Average ({selected_month})"
        active_dates_count = len(np.unique(dates_dt[date_mask]))
        st.caption(f"Aggregating across **{active_dates_count}** available date(s) in {selected_month}.")

    elif mapping_mode == "2. Day Wise Mapping":
        st.markdown("**Select Specific Date**")
        date_options = [str(d)[:10] for d in unique_dates_dt]
        selected_date_str = st.selectbox("Available Dates (2026):", date_options)
        
        date_mask = (dates_dt.dt.strftime("%Y-%m-%d") == selected_date_str).values
        map_title_label = f"Date: {selected_date_str}"
        active_dates_count = 1

    else:  # 3. Custom Mapping
        st.markdown("**Select Date Range**")
        min_d = unique_dates_dt.min().date()
        max_d = unique_dates_dt.max().date()
        
        start_date = st.date_input("Start Date:", value=min_d, min_value=min_d, max_value=max_d)
        end_date = st.date_input("End Date:", value=max_d, min_value=min_d, max_value=max_d)
        
        if start_date > end_date:
            st.error("Start Date must be earlier than or equal to End Date.")
            st.stop()
            
        date_mask = ((dates_dt.dt.date >= start_date) & (dates_dt.dt.date <= end_date)).values
        map_title_label = f"Custom Range ({start_date} to {end_date})"
        active_dates_count = len(np.unique(dates_dt[date_mask]))
        st.caption(f"Aggregating across **{active_dates_count}** available date(s) in selected range.")

    st.markdown("---")
    run_button = st.button("🚀 Run Inference & Render Map", type="primary", use_container_width=True)

# Validate selection has data
if np.sum(date_mask) == 0:
    with main_col:
        st.warning("No test samples found for the selected date/range. Please adjust your selection on the right panel.")
    st.stop()

# Filter subsets
coords_subset = coords_te[date_mask]
preds_subset = preds[date_mask]
y_subset = Y_te[date_mask]
dates_subset = dates_dt[date_mask]

# Aggregate pixel-by-pixel mean across selected dates for map generation
agg_df = pd.DataFrame({
    "gx": coords_subset[:, 0].astype(int),
    "gy": coords_subset[:, 1].astype(int),
    "pred": preds_subset,
    "actual": y_subset
})
pixel_avg_df = agg_df.groupby(["gx", "gy"], as_index=False).mean()

# Compute evaluation metrics on aggregated spatial map
y_eval = pixel_avg_df["actual"].values
p_eval = pixel_avg_df["pred"].values

sub_r2 = 1 - np.sum((y_eval - p_eval)**2) / (np.sum((y_eval - np.mean(y_eval))**2) + 1e-8)
sub_rmse = np.sqrt(np.mean((y_eval - p_eval)**2))
sub_mae = np.mean(np.abs(y_eval - p_eval))
sub_mse = np.mean((y_eval - p_eval)**2)
max_tmrt = np.max(p_eval)
min_tmrt = np.min(p_eval)

with main_col:
    if not run_button:
        st.info("👈 Select your desired mapping mode and date(s) on the right panel, then click **'Run Inference & Render Map'** to generate the thermal analysis.")
    else:
        with st.spinner("⏳ Running spatial aggregation and rendering thermal visualizations..."):
            # Metrics Row 1
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average $R^2$ Score" if active_dates_count > 1 else "$R^2$ Score", f"{sub_r2:.3f}")
            col2.metric("Average RMSE (°C)" if active_dates_count > 1 else "RMSE (°C)", f"{sub_rmse:.2f} °C")
            col3.metric("MAE (°C)", f"{sub_mae:.2f} °C")
            col4.metric("MSE (°C²)", f"{sub_mse:.2f} °C²")

            # Metrics Row 2: Max and Min Recorded TMRT
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Max $T_{mrt}$ Recorded", f"{max_tmrt:.1f} °C")
            col_m2.metric("Min $T_{mrt}$ Recorded", f"{min_tmrt:.1f} °C")
            col_m3.metric("Active Temporal Span", f"{active_dates_count} day(s)")
            col_m4.metric("Spatial Grid Nodes", f"{len(p_eval):,}")

            st.markdown("---")

            # Render Map
            full_satellite_rgb = fetch_esri_basemap()
            img_h, img_w = full_satellite_rgb.shape[:2]

            grid_matrix = np.full((GRID_DIM, GRID_DIM), np.nan)
            for row in pixel_avg_df.itertuples():
                if 0 <= row.gx < GRID_DIM and 0 <= row.gy < GRID_DIM:
                    grid_matrix[row.gx, row.gy] = row.pred

            upscaled_preds = cv2.resize(
                np.nan_to_num(grid_matrix, nan=np.nanmean(grid_matrix)), 
                (img_w, img_h), 
                interpolation=cv2.INTER_NEAREST
            )

            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
            ax.imshow(full_satellite_rgb)
            im = ax.imshow(upscaled_preds, cmap='jet', alpha=0.25, vmin=30, vmax=60, extent=[0, img_w, img_h, 0], origin='upper')
            ax.set_title(f"Nampally Thermal Stress Overlay ($T_{{mrt}}$)\n{map_title_label}", fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('$T_{mrt}$ (°C)', fontsize=10, fontweight='bold')

            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            st.subheader("📊 Performance & Trend Indicator Metrics")
            
            # Layout graphs: Separate plots for R^2, Error metrics, and Trend line graphs (if multiple days)
            if active_dates_count > 1:
                fig_metrics, axes = plt.subplots(1, 5, figsize=(24, 4.5), dpi=150)
                ax_r2, ax_err, ax_line_max, ax_line_min, ax_line_mean = axes
            else:
                fig_metrics, (ax_r2, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

            # 1. R^2 Score Separate Plot
            bars_r2 = ax_r2.bar(['$R^2$ Score'], [sub_r2], color='#1f77b4', alpha=0.85, width=0.4)
            ax_r2.set_title("Coefficient of Determination", fontsize=11, fontweight='bold')
            ax_r2.set_ylim(0, 1.2)
            for bar in bars_r2:
                height = bar.get_height()
                ax_r2.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=10, fontweight='bold')

            # 2. Error Metrics Plot (RMSE, MAE, MSE)
            err_names = ['RMSE (°C)', 'MAE (°C)', 'MSE (°C²)']
            err_vals = [sub_rmse, sub_mae, sub_mse]
            bars_err = ax_err.bar(err_names, err_vals, color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.85)
            ax_err.set_title("Error Metrics Summary", fontsize=11, fontweight='bold')
            ax_err.set_ylim(0, max(err_vals) * 1.3)
            for bar in bars_err:
                height = bar.get_height()
                ax_err.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')

            # 3. Multi-Day Trend Line Graphs (Max, Min, Mean TMRT)
            if active_dates_count > 1:
                daily_stats_df = pd.DataFrame({
                    "date": dates_subset,
                    "pred": preds_subset
                }).groupby("date")["pred"].agg(['max', 'min', 'mean']).reset_index()
                
                # Max TMRT Trend Line
                ax_line_max.plot(daily_stats_df["date"], daily_stats_df["max"], marker='o', color='#d62728', linewidth=2, markersize=5)
                ax_line_max.set_title("Max $T_{mrt}$ Trend", fontsize=11, fontweight='bold')
                ax_line_max.set_xlabel("Date")
                ax_line_max.set_ylabel("Max $T_{mrt}$ (°C)")
                ax_line_max.tick_params(axis='x', rotation=30)
                ax_line_max.grid(True, linestyle='--', alpha=0.5)

                # Min TMRT Trend Line
                ax_line_min.plot(daily_stats_df["date"], daily_stats_df["min"], marker='o', color='#1f77b4', linewidth=2, markersize=5)
                ax_line_min.set_title("Min $T_{mrt}$ Trend", fontsize=11, fontweight='bold')
                ax_line_min.set_xlabel("Date")
                ax_line_min.set_ylabel("Min $T_{mrt}$ (°C)")
                ax_line_min.tick_params(axis='x', rotation=30)
                ax_line_min.grid(True, linestyle='--', alpha=0.5)

                # Mean TMRT Trend Line
                ax_line_mean.plot(daily_stats_df["date"], daily_stats_df["mean"], marker='o', color='#2ca02c', linewidth=2, markersize=5)
                ax_line_mean.set_title("Mean $T_{mrt}$ Trend", fontsize=11, fontweight='bold')
                ax_line_mean.set_xlabel("Date")
                ax_line_mean.set_ylabel("Mean $T_{mrt}$ (°C)")
                ax_line_mean.tick_params(axis='x', rotation=30)
                ax_line_mean.grid(True, linestyle='--', alpha=0.5)

            st.pyplot(fig_metrics)
            plt.close(fig_metrics)