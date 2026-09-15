import os
import sys
import json
import subprocess
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image

try:
    from herbie import Herbie
except ImportError:
    print("Herbie is missing. Install via: pip install herbie-data xarray cfgrib pillow numpy")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "textures")
FXX_RANGE = list(range(0, 241, 6))  # 0 to 240 hours

LEVELS = ["10m", "925hPa", "850hPa", "700hPa", "500hPa", "200hPa"]

def get_model_product(model, level):
    """Returns correct Herbie product based on model and target level."""
    if model == "aigfs":
        return "sfc" if level == "10m" else "pres"
    elif model == "gfs":
        return "pgrb2.0p25"
    elif model in ["ifs", "aifs"]:
        return "oper"
    return None

def get_search_query(model, level, var_type):
    """Generates exact GRIB search strings for Herbie based on model and level."""
    is_ifs = model in ["ifs", "aifs"]
    
    if level == "10m":
        if is_ifs:
            return ":10u:" if var_type == 'u' else ":10v:"
        elif model == "aigfs":
            # AIGFS sfc product variable naming
            return ":UGRD:" if var_type == 'u' else ":VGRD:"
        else:
            return ":UGRD:10 m above ground:" if var_type == 'u' else ":VGRD:10 m above ground:"
    else:
        hpa = level.replace("hPa", "")
        if is_ifs:
            return f":u:{hpa}:" if var_type == 'u' else f":v:{hpa}:"
        else:
            # GFS and AIGFS pressure product string
            return f":UGRD:{hpa} mb:" if var_type == 'u' else f":VGRD:{hpa} mb:"

def ensure_caffeinated():
    """Spawns caffeinate on macOS to prevent system sleep during execution."""
    if sys.platform == "darwin" and "CAFFEINATED" not in os.environ:
        print(" Preventing macOS sleep via caffeinate...")
        env = os.environ.copy()
        env["CAFFEINATED"] = "1"
        cmd = ["caffeinate", "-i", "-w", str(os.getpid()), sys.executable] + sys.argv
        sys.exit(subprocess.call(cmd, env=env))

def get_latest_available_cycle():
    """Determines latest available run cycle based on model publication lag."""
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

def process_frame(u, v, lat, model_name, level, fxx):
    """Encodes velocity fields into RGBA PNGs and companion JSON metadata."""
    if lat[0] < lat[-1]:
        u, v = np.flipud(u), np.flipud(v)

    u_min, u_max = float(np.nanmin(u)), float(np.nanmax(u))
    v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))

    u_norm = np.clip(255 * (u - u_min) / (u_max - u_min + 1e-6), 0, 255).astype(np.uint8)
    v_norm = np.clip(255 * (v - v_min) / (v_max - v_min + 1e-6), 0, 255).astype(np.uint8)

    height, width = u.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = u_norm  # Red = U
    rgba[..., 1] = v_norm  # Green = V
    rgba[..., 3] = 255     # Alpha

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename_base = f"{model_name}_{level}_{fxx:03d}"
    img_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    meta_path = os.path.join(OUTPUT_DIR, f"{filename_base}.json")

    Image.fromarray(rgba, mode="RGBA").save(img_path)

    meta = {
        "uMin": u_min, "uMax": u_max,
        "vMin": v_min, "vMax": v_max,
        "width": width, "height": height,
        "fxx": fxx, "model": model_name, "level": level
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

def run_update():
    target_cycle = get_latest_available_cycle()
    print(f"\n==================================================")
    print(f"[{datetime.now(timezone.utc)}] Running multi-level update for cycle: {target_cycle}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"==================================================\n")

    models = ["gfs", "aigfs", "ifs", "aifs"]

    for model_name in models:
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

                    u_var = list(ds_u.data_vars)[0]
                    v_var = list(ds_v.data_vars)[0]

                    lat_key = 'latitude' if 'latitude' in ds_u else 'lat'

                    process_frame(ds_u[u_var].values, ds_v[v_var].values, ds_u[lat_key].values, model_name, level, fxx)
                    print(f"    Saved: {model_name}_{level}_{fxx:03d}.png")
                except Exception as e:
                    print(f"    Failed {model_name.upper()} {level} +{fxx:03d}h: {e}")

    manifest = {
        "cycle": target_cycle,
        "levels": LEVELS,
        "updatedAt": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("\nBatch multi-level update complete!")

if __name__ == "__main__":
    ensure_caffeinated()
    run_update()