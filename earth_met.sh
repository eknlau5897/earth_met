#!/usr/bin/env bash
set -e

echo "=== Earth Met 3D Environment Setup & Sync ==="

# 1. Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 2. Install/Update Dependencies (System/User level)
echo "Installing/Updating required dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install herbie-data xarray cfgrib pillow numpy scipy

# Loop interval in seconds (e.g., 6 hours = 21600 seconds)
SLEEP_INTERVAL=43200
BRANCH_NAME="main"

while true; do
    echo "=================================================="
    echo "Starting update cycle at $(date -u)"
    echo "=================================================="

    TEXTURES_DIR="textures"
    HERBIE_CACHE_DIR="$HOME/rubicon_data"

    # Reset branch to orphan temporary branch to clear past history
    git checkout --orphan temp_branch 2>/dev/null || git checkout temp_branch

    # --------------------------------------------------
    # COMMIT 1: Purge & Clean Old Files
    # --------------------------------------------------
    echo "--------------------------------------------------"
    echo "[Commit 1/3] Clearing previous textures & cache..."

    if [ -d "$TEXTURES_DIR" ]; then
        rm -f "$TEXTURES_DIR"/*.png
        rm -f "$TEXTURES_DIR"/*.json
    else
        mkdir -p "$TEXTURES_DIR"
    fi

    if [ -d "$HERBIE_CACHE_DIR" ]; then
        rm -rf "$HERBIE_CACHE_DIR"/*
    fi

    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    git add -A
    git commit -m "Step 1/3: Purge old textures and cache [$(date -u +'%Y-%m-%d %H:%M UTC')]" || true
    
    # Set main branch and push Step 1
    git branch -D "$BRANCH_NAME" 2>/dev/null || true
    git branch -m "$BRANCH_NAME"
    echo "Pushing Commit 1 to remote..."
    git push -f origin "$BRANCH_NAME"

    # --------------------------------------------------
    # COMMIT 2: Download Data & Generate New Textures
    # --------------------------------------------------
    echo "--------------------------------------------------"
    echo "[Commit 2/3] Generating new weather textures..."

    if command -v caffeinate &> /dev/null; then
        caffeinate -i python3 download.py
    else
        python3 download.py
    fi

    git add -A
    git commit -m "Step 2/3: Generate fresh weather textures [$(date -u +'%Y-%m-%d %H:%M UTC')]" || true
    echo "Pushing Commit 2 to remote..."
    git push origin "$BRANCH_NAME"

    # --------------------------------------------------
    # COMMIT 3: Storage Cleanup & Final Manifest Sync
    # --------------------------------------------------
    echo "--------------------------------------------------"
    echo "[Commit 3/3] Pruning storage & updating manifest..."

    # Purge Herbie download cache after generation to free local storage
    if [ -d "$HERBIE_CACHE_DIR" ]; then
        rm -rf "$HERBIE_CACHE_DIR"/*
    fi

    git add -A
    git commit -m "Step 3/3: Complete cycle & prune storage [$(date -u +'%Y-%m-%d %H:%M UTC')]" || true
    echo "Pushing Commit 3 to remote..."
    git push origin "$BRANCH_NAME"

    # Reclaim Git repository disk space locally
    echo "Reclaiming local Git storage space..."
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive

    echo "--------------------------------------------------"
    echo "Cycle complete (3 commits pushed). Sleeping for $SLEEP_INTERVAL seconds..."
    sleep $SLEEP_INTERVAL
done