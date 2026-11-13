import os
import io
import urllib.request
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj
import zipfile
import joblib
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 0. CONFIGURATION & ARTIFACT LOADING
# ==========================================
TARGET = "TMRT"
DATASET_PATH = "data/Tabular_Dataset_v2.parquet"  # Ensure your dataset is in data/
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

print("📂 Loading pre-trained model and matching training artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, "best_xgb_tmrt_model.joblib"))
scaler_x = joblib.load(os.path.join(MODEL_DIR, "scaler_x.joblib"))
train_medians = joblib.load(os.path.join(MODEL_DIR, "train_medians.joblib"))
train_global_mean_tmrt = joblib.load(os.path.join(MODEL_DIR, "train_global_mean_tmrt.pkl"))
train_pixel_mean_tmrt = joblib.load(os.path.join(MODEL_DIR, "train_pixel_mean_tmrt.joblib"))

print("📂 Loading dataset...")
df = pd.read_parquet(DATASET_PATH) if DATASET_PATH.endswith(".parquet") else pd.read_csv(DATASET_PATH, encoding="latin1")
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

pix_num_all = df["Pixel_ID"].astype(str).str.extract(r"(\d+)").astype(int)[0]
GRID_DIM = int(np.ceil(np.sqrt(pix_num_all.max() + 1)))

# Focus strictly on the test set (2026) using pre-trained assets
df_test_raw = df[df["Year"] == 2026].copy()

# Thermal anomaly feature engineering using saved training stats
df_test_raw["Pixel_Thermal_Anomaly"] = df_test_raw["Pixel_ID"].map(train_pixel_mean_tmrt).fillna(train_global_mean_tmrt) - train_global_mean_tmrt
feature_cols_extended = feature_cols + ["Pixel_Thermal_Anomaly"]

# Apply saved medians and scaler
df_test_raw[feature_cols_extended] = df_test_raw[feature_cols_extended].fillna(train_medians)
df_test_raw[feature_cols_extended] = scaler_x.transform(df_test_raw[feature_cols_extended])

# ==========================================
# 1. SPATIAL CHIP ASSEMBLY & INFERENCE
# ==========================================
def create_spatial_dataset_2d(df, feature_cols, target, grid_dim, patch_size):
    pad = patch_size // 2
    df_clean = df.dropna(subset=[target]).copy()
    pix_num = df_clean["Pixel_ID"].astype(str).str.extract(r"(\d+)").astype(int)[0]
    df_clean["Grid_X"] = pix_num // grid_dim
    df_clean["Grid_Y"] = pix_num % grid_dim
    
    map_dim_x = max(grid_dim, df_clean["Grid_X"].max() + 1) + (2 * pad)
    map_dim_y = max(grid_dim, df_clean["Grid_Y"].max() + 1) + (2 * pad)
    
    X_chips, Y_targets, Coords = [], [], []
    for date, df_date in df_clean.groupby("Date"):
        feature_maps = np.zeros((map_dim_x, map_dim_y, len(feature_cols)), dtype=np.float32)
        gx_arr = df_date["Grid_X"].values + pad
        gy_arr = df_date["Grid_Y"].values + pad
        feature_maps[gx_arr, gy_arr, :] = df_date[feature_cols].values
        
        for gx, gy, target_val, orig_x, orig_y in zip(gx_arr, gy_arr, df_date[target].values, df_date["Grid_X"].values, df_date["Grid_Y"].values):
            chip = feature_maps[gx-pad:gx+pad+1, gy-pad:gy+pad+1, :]
            X_chips.append(chip.transpose(2, 0, 1))
            Y_targets.append(target_val)
            Coords.append((orig_x, orig_y))
            
    return np.array(X_chips, dtype=np.float32), np.array(Y_targets, dtype=np.float32), np.array(Coords, dtype=np.float32)

print("⚙️ Assembling spatial test chips...")
X_te, Y_te, coords_te = create_spatial_dataset_2d(df_test_raw, feature_cols_extended, TARGET, GRID_DIM, NEIGHBORHOOD_SIZE)

def extract_features(X_spatial, coords, grid_dim, feature_names):
    N, C, H, W = X_spatial.shape
    center_h, center_w = H // 2, W // 2
    inner_flat = X_spatial[:, :, center_h-1:center_h+2, center_w-1:center_w+2].reshape(N, -1)
    center_vals = X_spatial[:, :, center_h, center_w]
    spatial_means = np.mean(X_spatial, axis=(2, 3))
    spatial_stds = np.std(X_spatial, axis=(2, 3))
    spatial_mins = np.min(X_spatial, axis=(2, 3))
    spatial_maxs = np.max(X_spatial, axis=(2, 3))
    return np.hstack([inner_flat, center_vals, spatial_means, spatial_stds, spatial_mins, spatial_maxs]).astype(np.float32)

X_te_eng = extract_features(X_te, coords_te, GRID_DIM, feature_cols_extended)

print("🚀 Running inference with pre-trained XGBoost model...")
start_time = time.time()

# Clean native prediction since shapes match training 100%
preds = model.predict(X_te_eng)

inference_time = time.time() - start_time

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(Y_te, preds))
r2 = r2_score(Y_te, preds)
mae = mean_absolute_error(Y_te, preds)

print(f"\n==========================================")
print(f"       EVALUATION METRICS REPORT          ")
print(f"==========================================")
print(f"Total Test Samples Evaluated : {len(Y_te)}")
print(f"R² Score                     : {r2:.4f}")
print(f"RMSE                         : {rmse:.4f} °C")
print(f"MAE                          : {mae:.4f} °C")
print(f"Inference Latency            : {inference_time:.4f}s (~{(inference_time/len(Y_te))*1000:.2f} ms/sample)")
print(f"==========================================\n")

# ==========================================
# 2. FETCH EXACT ESRI SATELLITE TILE (NAMPALLY)
# ==========================================
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

print("📥 Fetching Esri satellite tiles...")
stitched_img = []
for y in range(y_min_t, y_max_t + 1):
    row_imgs = []
    for x in range(x_min_t, x_max_t + 1):
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            tile = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
            row_imgs.append(tile)
    stitched_img.append(np.hstack(row_imgs))

full_satellite_rgb = cv2.cvtColor(np.vstack(stitched_img), cv2.COLOR_BGR2RGB)
img_h, img_w = full_satellite_rgb.shape[:2]

# Save base map
base_map_path = os.path.join(OUTPUT_DIR, "nampally_helicopter_basemap.png")
plt.figure(figsize=(10, 10), dpi=300)
plt.imshow(full_satellite_rgb)
plt.title("Nampally - Raw Esri Orthophoto Basemap", fontsize=13, fontweight='bold', pad=12)
plt.axis('off')
plt.savefig(base_map_path, bbox_inches='tight', dpi=300)
plt.close()

# ==========================================
# 3. MAP REAL PREDICTIONS TO 2D GRID & OVERLAY
# ==========================================
sample_date = df_test_raw["Date"].iloc[0]
date_mask = (df_test_raw["Date"] == sample_date).values
coords_subset = coords_te[date_mask]
preds_subset = preds[date_mask]

grid_matrix = np.full((GRID_DIM, GRID_DIM), np.nan)
for (gx, gy), val in zip(coords_subset, preds_subset):
    ix, iy = int(gx), int(gy)
    if 0 <= ix < GRID_DIM and 0 <= iy < GRID_DIM:
        grid_matrix[ix, iy] = val

upscaled_preds = cv2.resize(np.nan_to_num(grid_matrix, nan=np.nanmean(grid_matrix)), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
combined_map_path = os.path.join(OUTPUT_DIR, "nampally_real_tmrt_overlay_jet.png")

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.imshow(full_satellite_rgb)
im = ax.imshow(upscaled_preds, cmap='jet', alpha=0.22, vmin=30, vmax=60, extent=[0, img_w, img_h, 0], origin='upper')
ax.set_title(f"Nampally Real TMRT Stress Overlay (Jet Colormap)\n(Date: {str(sample_date)[:10]})", fontsize=13, fontweight='bold', pad=12)
ax.axis('off')
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('TMRT (°C)', fontsize=11, fontweight='bold')
plt.savefig(combined_map_path, bbox_inches='tight', dpi=300)
plt.close(fig)

# ==========================================
# 4. ZIP BOTH FILES
# ==========================================
zip_path = os.path.join(OUTPUT_DIR, "nampally_jet_maps_bundle.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(base_map_path, arcname="nampally_helicopter_basemap.png")
    zipf.write(combined_map_path, arcname="nampally_real_tmrt_overlay_jet.png")

print(f"✅ Successfully evaluated model and zipped maps into: {zip_path}")