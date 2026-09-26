import os
import sys
import json
import subprocess
from datetime import datetime, timezone, timedelta
import numpy as np
import xarray as xr
from PIL import Image

try:
    from herbie import Herbie
except ImportError:
    print("Herbie is missing. Install via: pip install herbie-data xarray cfgrib pillow numpy scipy")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "textures")
FXX_RANGE = list(range(0, 241, 6))  # 0 to 240 hours in 6-hour increments

# All requested models and pressure levels
MODELS = ["gfs", "aigfs", "ifs", "aifs"]
LEVELS = ["10m", "925hPa", "850hPa", "700hPa", "500hPa", "200hPa"]

# Standardized 0° to 360° global equirectangular target grid
TARGET_LONS = np.linspace(0.0, 360.0, 1441, endpoint=True)  # 0.0 to 360.0
TARGET_LATS = np.linspace(90.0, -90.0, 721, endpoint=True)  # 90 (North) to -90 (South)

# Continuous color ramp matching Windy's wind speed palette (in knots)
WINDY_STOPS = [
    (0.0,   [3,   13,  34]),    # Very low (Dark Navy)
    (5.0,   [12,  44,  132]),   # Low (Deep Blue)
    (15.0,  [20,  110, 180]),   # Moderate (Cyan Blue)
    (30.0,  [40,  180, 160]),   # Fresh (Teal-Green)
    (45.0,  [160, 220, 60]),    # Strong (Yellow-Green)
    (60.0,  [240, 180, 20]),    # Gale (Orange)
    (80.0,  [230, 50,  40]),    # Storm (Red)
    (100.0, [190, 20,  120]),   # Violent (Magenta/Purple)
    (130.0, [250, 250, 250])    # Extreme (White)
]

def clear_texture_cache():
    """Purges previously generated PNG and JSON textures before starting a run."""
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith((".png", ".json")):
                file_path = os.path.join(OUTPUT_DIR, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        print("Cleared previous texture frame history.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_windy_rgb(speed_ms):
    """Maps wind speed (m/s) to RGB using linear interpolation across Windy color stops."""
    knots = speed_ms * 1.94384
    
    if np.all(knots <= WINDY_STOPS[0][0]):
        return np.full((*speed_ms.shape, 3), WINDY_STOPS[0][1], dtype=np.float32)
    if np.all(knots >= WINDY_STOPS[-1][0]):
        return np.full((*speed_ms.shape, 3), WINDY_STOPS[-1][1], dtype=np.float32)

    rgb = np.zeros((*speed_ms.shape, 3), dtype=np.float32)

    for i in range(len(WINDY_STOPS) - 1):
        k0, c0 = WINDY_STOPS[i]
        k1, c1 = WINDY_STOPS[i + 1]

        mask = (knots >= k0) & (knots < k1)
        if not np.any(mask):
            continue

        t = (knots[mask] - k0) / (k1 - k0)
        c0_arr = np.array(c0, dtype=np.float32)
        c1_arr = np.array(c1, dtype=np.float32)

        rgb[mask] = c0_arr + t[:, None] * (c1_arr - c0_arr)

    return rgb

def get_model_product(model, level):
    """Returns the correct Herbie product name per model and level."""
    if model == "aigfs":
        return "sfc" if level == "10m" else "pres"
    elif model == "gfs":
        return "pgrb2.0p25"
    elif model in ["ifs", "aifs"]:
        return "oper"
    return None

def get_search_query(model, level, var_type):
    """Constructs GRIB search query pattern for U/V variables."""
    is_ifs = model in ["ifs", "aifs"]
    if level == "10m":
        if is_ifs:
            return ":10u:" if var_type == 'u' else ":10v:"
        elif model == "aigfs":
            return ":UGRD:" if var_type == 'u' else ":VGRD:"
        else:
            return ":UGRD:10 m above ground:" if var_type == 'u' else ":VGRD:10 m above ground:"
    else:
        hpa = level.replace("hPa", "")
        if is_ifs:
            return f":u:{hpa}:" if var_type == 'u' else f":v:{hpa}:"
        else:
            return f":UGRD:{hpa} mb:" if var_type == 'u' else f":VGRD:{hpa} mb:"

def standardize_grid_360(ds):
    """Normalizes longitudes [0, 360] and interpolates to a clean equirectangular grid."""
    lon_key = 'longitude' if 'longitude' in ds.coords else 'lon'
    lat_key = 'latitude' if 'latitude' in ds.coords else 'lat'

    # 1. Normalize longitudes into [0, 360) range
    lon_vals = ds[lon_key].values
    if np.any(lon_vals < 0):
        ds = ds.assign_coords({lon_key: (ds[lon_key] % 360.0)})

    # 2. Sort coordinates safely
    if ds[lon_key].ndim == 1:
        ds = ds.sortby(lon_key)
    if ds[lat_key].ndim == 1:
        ds = ds.sortby(lat_key)

    # 3. Handle periodic longitude boundary wrapping
    if ds[lon_key].ndim == 1:
        lon_array = ds[lon_key].values
        if lon_array[-1] < 360.0:
            first_slice = ds.isel({lon_key: 0}).assign_coords({lon_key: 360.0})
            ds = xr.concat([ds, first_slice], dim=lon_key)

    # 4. Interpolate directly to target mesh
    ds_interp = ds.interp(
        {lat_key: TARGET_LATS, lon_key: TARGET_LONS},
        method="linear"
    )

    return ds_interp, lat_key, lon_key

def ensure_caffeinated():
    """Keeps system awake on macOS during execution."""
    if sys.platform == "darwin" and "CAFFEINATED" not in os.environ:
        print(" Preventing macOS sleep via caffeinate...")
        env = os.environ.copy()
        env["CAFFEINATED"] = "1"
        cmd = ["caffeinate", "-i", "-w", str(os.getpid()), sys.executable] + sys.argv
        sys.exit(subprocess.call(cmd, env=env))

def get_latest_available_cycle():
    """Calculates the most recent operational model run cycle time."""
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
    """Saves visual background map, vector UV PNG, and JSON metadata."""
    u = np.squeeze(u)
    v = np.squeeze(v)

    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

    # Calculate speed magnitude (m/s)
    speed = np.sqrt(u**2 + v**2)

    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))

    # 1. Pure background color image (RGB)
    rgb_filled = get_windy_rgb(speed).astype(np.uint8)
    
    # 2. Vector field image (R = U, G = V, B = 0, A = 255)
    u_norm = np.clip(255 * (u - u_min) / (u_max - u_min + 1e-6), 0, 255).astype(np.uint8)
    v_norm = np.clip(255 * (v - v_min) / (v_max - v_min + 1e-6), 0, 255).astype(np.uint8)

    height, width = u.shape
    vector_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    vector_rgba[..., 0] = u_norm  # Red = U (Zonal vector)
    vector_rgba[..., 1] = v_norm  # Green = V (Meridional vector)
    vector_rgba[..., 2] = 0       # Unused
    vector_rgba[..., 3] = 255     # Opaque

    # Standardize image orientation (North=Top, South=Bottom)
    bg_img = Image.fromarray(rgb_filled, mode="RGB")
    vec_img = Image.fromarray(vector_rgba, mode="RGBA")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_base = f"{model_name}_{level}_{fxx:03d}"
    
    bg_img.save(os.path.join(OUTPUT_DIR, f"{filename_base}.png"))
    vec_img.save(os.path.join(OUTPUT_DIR, f"{filename_base}_uv.png"))

    meta = {
        "uMin": u_min, "uMax": u_max,
        "vMin": v_min, "vMax": v_max,
        "width": width, "height": height,
        "lonRange": [0.0, 360.0],
        "latRange": [-90.0, 90.0],
        "fxx": fxx, "model": model_name, "level": level,
        "cycle": cycle_str
    }
    with open(os.path.join(OUTPUT_DIR, f"{filename_base}.json"), "w") as f:
        json.dump(meta, f)

def run_update():
    clear_texture_cache()
    target_cycle = get_latest_available_cycle()
    print(f"\n==================================================")
    print(f"[{datetime.now(timezone.utc)}] Running multi-level update for cycle: {target_cycle}")
    print(f"Models: {MODELS}")
    print(f"Levels: {LEVELS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"==================================================\n")

    for model_name in MODELS:
        print(f"--> Updating Model: {model_name.upper()}")
        for level in LEVELS:
            print(f"  -> Level: {level}")
            
            product = get_model_product(model_name, level)
            u_search = get_search_query(model_name, level, 'u')
            v_search = get_search_query(model_name, level, 'v')

            for fxx in FXX_RANGE:
                try:
                    H = Herbie(target_cycle, model=model_name, product=product, fxx=fxx)
                    ds_u = H.xarray(u_search)
                    ds_v = H.xarray(v_search)

                    ds_u, lat_key, lon_key = standardize_grid_360(ds_u)
                    ds_v, _, _ = standardize_grid_360(ds_v)

                    u_var = list(ds_u.data_vars)[0]
                    v_var = list(ds_v.data_vars)[0]

                    process_frame(ds_u[u_var].values, ds_v[v_var].values, model_name, level, fxx, target_cycle)
                    print(f"    Saved: {model_name}_{level}_{fxx:03d}.png and _uv.png")
                except Exception as e:
                    print(f"    Failed {model_name.upper()} {level} +{fxx:03d}h: {e}")

    manifest = {
        "cycle": target_cycle,
        "models": MODELS,
        "levels": LEVELS,
        "lonRange": [0.0, 360.0],
        "updatedAt": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("\nBatch multi-level update complete!")

if __name__ == "__main__":
    ensure_caffeinated()
    run_update()