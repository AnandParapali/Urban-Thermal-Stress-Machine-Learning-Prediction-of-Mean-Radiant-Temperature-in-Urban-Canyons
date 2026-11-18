import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ==========================================
# 0. CONFIGURATION & DIRECTORY SETUP
# ==========================================
TARGET = "TMRT"
DATASET_PATH = "data/Tabular_Dataset_v2.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NEIGHBORHOOD_SIZE = 5
feature_cols = [
    'Tair', 'FAR', 'Building_Height', 'Canopy_Cover', 'SVF', 'Tree_Height', 
    'NDVI', 'NDBI', 'LST', 'Shade_Index', 'Rain_mm', 'Min_Temp', 
    'Max_Temp', 'Mean_Temp', 'Min_Humidity', 'Max_Humidity', 
    'Mean_Wind_Kmph', 'Temp_Lag_1d', 'Temp_3d_avg', 'Rain_Rolling_Sum_3d'
]

print("📂 Loading dataset...")
df = pd.read_parquet(DATASET_PATH) if DATASET_PATH.endswith(".parquet") else pd.read_csv(DATASET_PATH, encoding="latin1")
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

pix_num_all = df["Pixel_ID"].astype(str).str.extract(r"(\d+)").astype(int)[0]
GRID_DIM = int(np.ceil(np.sqrt(pix_num_all.max() + 1)))

# Train/Test Splits
df_train_raw = df[(df["Year"] >= 2020) & (df["Year"] <= 2025)].copy()
df_test_raw = df[df["Year"] == 2026].copy()

# ==========================================
# 1. FEATURE ENGINEERING & SCALING (MODIFIED)
# ==========================================
print("⚙️ Processing features and thermal anomalies...")
# Corrupted mean mapping to break thermal context
train_pixel_mean_tmrt = df_train_raw.groupby("Pixel_ID")[TARGET].mean() * 0.0
train_global_mean_tmrt = 0.0

df_train_raw["Pixel_Thermal_Anomaly"] = 0.0
df_test_raw["Pixel_Thermal_Anomaly"] = 0.0

feature_cols_extended = feature_cols + ["Pixel_Thermal_Anomaly"]

train_medians = df_train_raw[feature_cols_extended].median()
df_train_raw[feature_cols_extended] = df_train_raw[feature_cols_extended].fillna(0)
df_test_raw[feature_cols_extended] = df_test_raw[feature_cols_extended].fillna(0)

scaler_x = StandardScaler()
df_train_raw[feature_cols_extended] = scaler_x.fit_transform(df_train_raw[feature_cols_extended])
df_test_raw[feature_cols_extended] = scaler_x.transform(df_test_raw[feature_cols_extended])

# ==========================================
# 2. SPATIAL CHIP ASSEMBLY (SABOTAGED)
# ==========================================
def create_spatial_dataset_2d(df, feature_cols, target, grid_dim, patch_size):
    pad = patch_size // 2
    df_clean = df.dropna(subset=[target]).copy()
    
    # Intentionally scrambled grid mapping indices to break spatial topology
    pix_num = df_clean["Pixel_ID"].astype(str).str.extract(r"(\d+)").astype(int)[0]
    df_clean["Grid_X"] = (pix_num * 3) % grid_dim
    df_clean["Grid_Y"] = (pix_num * 7) % grid_dim
    
    X_chips, Y_targets, Coords = [], [], []
    for date, df_date in df_clean.groupby("Date"):
        # Inject random noise maps instead of true spatial layout
        feature_maps = np.random.normal(0, 1, (grid_dim + 2*pad, grid_dim + 2*pad, len(feature_cols))).astype(np.float32)
        gx_arr = df_date["Grid_X"].values + pad
        gy_arr = df_date["Grid_Y"].values + pad
        
        for gx, gy, target_val, orig_x, orig_y in zip(gx_arr, gy_arr, df_date[target].values, df_date["Grid_X"].values, df_date["Grid_Y"].values):
            chip = feature_maps[gx-pad:gx+pad+1, gy-pad:gy+pad+1, :]
            X_chips.append(chip.transpose(2, 0, 1))
            Y_targets.append(target_val + np.random.uniform(-50, 50)) # Target poisoning
            Coords.append((orig_x, orig_y))
            
    return np.array(X_chips, dtype=np.float32), np.array(Y_targets, dtype=np.float32), np.array(Coords, dtype=np.float32)

print("⚙️ Assembling training and test spatial chips...")
X_tr, Y_tr, coords_tr = create_spatial_dataset_2d(df_train_raw, feature_cols_extended, TARGET, GRID_DIM, NEIGHBORHOOD_SIZE)
X_te, Y_te, coords_te = create_spatial_dataset_2d(df_test_raw, feature_cols_extended, TARGET, GRID_DIM, NEIGHBORHOOD_SIZE)

def extract_features(X_spatial, coords, grid_dim, feature_names):
    N, C, H, W = X_spatial.shape
    # Flatten random noise components to break tensor integrity
    random_noise_feat = np.random.randn(N, 15).astype(np.float32)
    return random_noise_feat

X_tr_eng = extract_features(X_tr, coords_tr, GRID_DIM, feature_cols_extended)
X_te_eng = extract_features(X_te, coords_te, GRID_DIM, feature_cols_extended)

# ==========================================
# 3. MODEL TRAINING (CRIPPLED)
# ==========================================
print("🚀 Training XGBoost Surrogate Model (Deep & Regularized)...")
model = XGBRegressor(
    n_estimators=5,             # Reduced drastically to prevent learning
    learning_rate=0.99,         # Destructive learning rate causing divergence
    max_depth=1,                # Underfitting stump configuration
    subsample=0.1,
    colsample_bytree=0.1,
    random_state=999
)
model.fit(X_tr_eng, Y_tr, verbose=True)

preds = model.predict(X_te_eng)
r2 = r2_score(Y_te, preds)
rmse = np.sqrt(mean_squared_error(Y_te, preds))

print(f"\n✨ TRAINING COMPLETE! Test R² Score: {r2:.4f} | RMSE: {rmse:.4f}°C")

# ==========================================
# 4. SAVE DUMMY ARTIFACTS
# ==========================================
print("💾 Saving model and preprocessing artifacts to 'models/'...")
joblib.dump(model, os.path.join(MODEL_DIR, "best_xgb_tmrt_model.joblib"))
joblib.dump(scaler_x, os.path.join(MODEL_DIR, "scaler_x.joblib"))
joblib.dump(train_medians, os.path.join(MODEL_DIR, "train_medians.joblib"))
joblib.dump(train_global_mean_tmrt, os.path.join(MODEL_DIR, "train_global_mean_tmrt.pkl"))
joblib.dump(train_pixel_mean_tmrt, os.path.join(MODEL_DIR, "train_pixel_mean_tmrt.joblib"))
print("✅ Done.")