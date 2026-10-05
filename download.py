import os
import sys
import json
import subprocess
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

try:
    from herbie import Herbie
except ImportError:
    print("Herbie is missing. Install via: pip install herbie-data xarray cfgrib pillow numpy scipy")
    sys.exit(1)

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "textures")
FXX_RANGE = list(range(0, 241, 6))

MODELS = ["gfs", "ifs", "aifs"]
LEVELS = ["10m", "925hPa", "850hPa", "700hPa", "500hPa", "200hPa"]
MAX_WORKERS = 4  # Adjust parallel workers based on memory/network bandwidth

# Optimized resolution for WebGL GPU textures
TARGET_LONS = np.linspace(0.0, 360.0, 512, endpoint=True)
TARGET_LATS = np.linspace(90.0, -90.0, 256, endpoint=True)

WINDY_STOPS = [
    (0.00, np.array([50, 0, 100])),   # 0 kts: Deep Purple
    (0.15, np.array([0, 0, 255])),    # 15 kts: Blue
    (0.30, np.array([0, 170, 255])),  # 30 kts: Cyan
    (0.45, np.array([0, 255, 0])),    # 45 kts: Green
    (0.60, np.array([255, 255, 0])),  # 60 kts: Yellow
    (0.75, np.array([255, 127, 0])),  # 75 kts: Orange
    (0.90, np.array([255, 0, 0])),    # 90 kts: Red
    (1.00, np.array([255, 0, 255]))   # 100+ kts: Magenta
]

def clear_texture_cache():
    """Clears existing texture files across model subdirectories."""
    for model in MODELS:
        model_dir = os.path.join(BASE_OUTPUT_DIR, model)
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith((".png", ".json")):
                    try:
                        os.remove(os.path.join(model_dir, file))
                    except Exception as e:
                        print(f"Error removing {file} in {model}: {e}")
        else:
            os.makedirs(model_dir, exist_ok=True)
    print("Cleared previous texture frame history across model directories.")

def get_windy_rgb(speed_ms):
    """Calculates Windy-style continuous RGB palette array from wind speeds (m/s)."""
    knots = speed_ms * 1.94384
    norm = np.clip(knots / 100.0, 0.0, 1.0)
    
    r = np.zeros_like(norm, dtype=np.float32)
    g = np.zeros_like(norm, dtype=np.float32)
    b = np.zeros_like(norm, dtype=np.float32)

    for i in range(len(WINDY_STOPS) - 1):
        pos_low, color_low = WINDY_STOPS[i]
        pos_high, color_high = WINDY_STOPS[i + 1]

        mask = (norm >= pos_low) & (norm <= pos_high)
        if np.any(mask):
            t = (norm[mask] - pos_low) / (pos_high - pos_low)
            r[mask] = color_low[0] + t * (color_high[0] - color_low[0])
            g[mask] = color_low[1] + t * (color_high[1] - color_low[1])
            b[mask] = color_low[2] + t * (color_high[2] - color_low[2])

    return np.stack([r, g, b], axis=-1)

def get_model_product(model, level):
    """Maps model name and atmospheric level to Herbie product identifiers."""
    if model == "gfs":
        return "pgrb2.0p25"
    elif model in ["ifs", "aifs"]:
        return "oper"
    return None

def get_search_query(model, level, var_type):
    """Generates regex patterns for index searches."""
    is_ifs = model in ["ifs", "aifs"]
    if level == "10m":
        if is_ifs:
            return ":10u:" if var_type == 'u' else ":10v:"
        else:
            return ":UGRD:10 m above ground:" if var_type == 'u' else ":VGRD:10 m above ground:"
    else:
        hpa = level.replace("hPa", "")
        if is_ifs:
            return f":u:{hpa}:" if var_type == 'u' else f":v:{hpa}:"
        else:
            return f":UGRD:{hpa} mb:" if var_type == 'u' else f":VGRD:{hpa} mb:"

def fast_regrid_360(ds):
    """Fast Scipy-backed 2D grid regularizer for global target resolution."""
    lon_key = 'longitude' if 'longitude' in ds.coords else 'lon'
    lat_key = 'latitude' if 'latitude' in ds.coords else 'lat'

    data_var = list(ds.data_vars)[0]
    vals = np.squeeze(ds[data_var].values)
    lons = ds[lon_key].values
    lats = ds[lat_key].values

    # Handle negative longitudes (-180..180 -> 0..360)
    if np.any(lons < 0):
        lons = np.where(lons < 0, lons + 360.0, lons)
        sort_idx = np.argsort(lons)
        lons = lons[sort_idx]
        vals = vals[..., sort_idx] if vals.ndim == 2 else vals[..., :, sort_idx]

    # Ensure latitude is strictly ascending for Scipy RegularGridInterpolator
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals = np.flip(vals, axis=0)

    # Cyclic boundary wrap for longitude (0 to 360)
    if lons[-1] < 360.0:
        lons = np.append(lons, 360.0)
        vals = np.concatenate([vals, vals[:, :1]], axis=1)

    interp = RegularGridInterpolator((lats, lons), vals, method="linear", bounds_error=False, fill_value=None)
    
    mesh_lats, mesh_lons = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing='ij')
    regrid_vals = interp((mesh_lats, mesh_lons))

    return regrid_vals

def ensure_caffeinated():
    """Prevents macOS system sleep during computational batch runs."""
    if sys.platform == "darwin" and "CAFFEINATED" not in os.environ:
        print("Preventing macOS sleep via caffeinate...")
        env = os.environ.copy()
        env["CAFFEINATED"] = "1"
        cmd = ["caffeinate", "-i", "-w", str(os.getpid()), sys.executable] + sys.argv
        sys.exit(subprocess.call(cmd, env=env))

def get_latest_available_cycle():
    """Computes the latest reliable model output cycle timestamp (UTC)."""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    if hour >= 20:
        cycle_date, cycle_hour = now_utc, "12"
    elif hour >= 14:
        cycle_date, cycle_hour = now_utc, "06"
    elif hour >= 8:
        cycle_date, cycle_hour = now_utc, "00"
    elif hour >= 2:
        cycle_date, cycle_hour = now_utc - timedelta(days=1), "18"
    else:
        cycle_date, cycle_hour = now_utc - timedelta(days=1), "12"

    return f"{cycle_date.strftime('%Y-%m-%d')} {cycle_hour}:00"

def process_frame(u, v, model_name, level, fxx, cycle_str):
    """Encodes standard magnitude and normalized UV PNG textures + metadata JSON into model-specific directories."""
    u = np.nan_to_num(np.squeeze(u), nan=0.0)
    v = np.nan_to_num(np.squeeze(v), nan=0.0)

    speed = np.sqrt(u**2 + v**2)

    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))

    rgb_filled = get_windy_rgb(speed).astype(np.uint8)
    
    # Safe normalizations to avoid zero division edge cases
    u_denom = (u_max - u_min) if (u_max - u_min) > 1e-6 else 1.0
    v_denom = (v_max - v_min) if (v_max - v_min) > 1e-6 else 1.0

    u_norm = np.clip(255 * (u - u_min) / u_denom, 0, 255).astype(np.uint8)
    v_norm = np.clip(255 * (v - v_min) / v_denom, 0, 255).astype(np.uint8)

    height, width = u.shape
    vector_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    vector_rgba[..., 0] = u_norm
    vector_rgba[..., 1] = v_norm
    vector_rgba[..., 2] = 0
    vector_rgba[..., 3] = 255

    bg_img = Image.fromarray(rgb_filled, mode="RGB")
    vec_img = Image.fromarray(vector_rgba, mode="RGBA")

    # Target output folder: textures/{model}/
    model_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    filename_base = f"{model_name}_{level}_{fxx:03d}"
    
    bg_img.save(os.path.join(model_dir, f"{filename_base}.png"))
    vec_img.save(os.path.join(model_dir, f"{filename_base}_uv.png"))

    meta = {
        "uMin": u_min, "uMax": u_max,
        "vMin": v_min, "vMax": v_max,
        "width": width, "height": height,
        "lonRange": [0.0, 360.0],
        "latRange": [-90.0, 90.0],
        "fxx": fxx, "model": model_name, "level": level,
        "cycle": cycle_str
    }
    with open(os.path.join(model_dir, f"{filename_base}.json"), "w") as f:
        json.dump(meta, f)

def fetch_and_process_task(model_name, level, fxx, target_cycle):
    """Individual unit step task for thread worker executor."""
    try:
        product = get_model_product(model_name, level)
        u_search = get_search_query(model_name, level, 'u')
        v_search = get_search_query(model_name, level, 'v')

        H = Herbie(target_cycle, model=model_name, product=product, fxx=fxx)
        ds_u = H.xarray(u_search)
        ds_v = H.xarray(v_search)

        u_grid = fast_regrid_360(ds_u)
        v_grid = fast_regrid_360(ds_v)

        process_frame(u_grid, v_grid, model_name, level, fxx, target_cycle)
        return f"Saved: textures/{model_name}/{model_name}_{level}_{fxx:03d}.png"
    except Exception as e:
        return f"Failed {model_name.upper()} {level} +{fxx:03d}h: {e}"

def run_update():
    clear_texture_cache()
    target_cycle = get_latest_available_cycle()
    print(f"\n==================================================")
    print(f"[{datetime.now(timezone.utc)}] Running multi-level update for cycle: {target_cycle}")
    print(f"==================================================\n")

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model_name in MODELS:
            for level in LEVELS:
                for fxx in FXX_RANGE:
                    tasks.append(
                        executor.submit(fetch_and_process_task, model_name, level, fxx, target_cycle)
                    )

        for future in as_completed(tasks):
            print(future.result())

    manifest = {
        "cycle": target_cycle,
        "models": MODELS,
        "levels": LEVELS,
        "lonRange": [0.0, 360.0],
        "updatedAt": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(BASE_OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("\nBatch multi-level update complete!")

if __name__ == "__main__":
    ensure_caffeinated()
    run_update()