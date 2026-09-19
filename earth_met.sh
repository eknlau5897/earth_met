#!/usr/bin/env bash

set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GITHUB_REPO_URL="https://github.com/eknlau5897/earth_met.git"
BRANCH_NAME="main"

echo "=================================================="
echo " 3D Wind Field Player Operational Loop"
echo " Target UTC Update Windows: 08:00 UTC (00Z) & 20:00 UTC (12Z)"
echo "=================================================="

while true; do
    START_TIME=$(date +%s)
    CURRENT_UTC=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    
    echo ""
    echo "=================================================="
    echo " Starting Pipeline Run at: ${CURRENT_UTC}"
    echo "=================================================="

    # --------------------------------------------------------------------------
    # 0. STORAGE CLEANUP (Pre-Run Cache Purge)
    # --------------------------------------------------------------------------
    echo "--> Clearing old download cache and local texture files..."
    
    # Clear local texture output folder
    rm -rf textures/*
    mkdir -p textures

    # Purge Herbie/xarray GRIB download caches (typically in ~/.cache/herbie)
    if [ -d "$HOME/.cache/herbie" ]; then
        echo "--> Clearing Herbie download cache (~/.cache/herbie)..."
        rm -rf "$HOME/.cache/herbie"/*
    fi

    # Clear temp / tmp files created during xarray/cfgrib extraction
    rm -f *.idx *.grib *.grib2 *.nc *.tmp

    # --------------------------------------------------------------------------
    # 1. DOWNLOAD & PROCESSING PIPELINE
    # --------------------------------------------------------------------------
    if [ -f "download.py" ]; then
        echo "--> Executing download.py..."
        python download.py
    else
        echo "⚠️ WARNING: download.py not found! Skipping data download..."
    fi

    # --------------------------------------------------------------------------
    # 2. PURGE GIT HISTORY & KEEP ONLY LATEST COMMIT
    # --------------------------------------------------------------------------
    echo "--> Resetting repository history to single commit..."
    
    if [ ! -d ".git" ]; then
        git init
    fi

    if ! git remote | grep -q "^origin$"; then
        git remote add origin "$GITHUB_REPO_URL"
    else
        git remote set-url origin "$GITHUB_REPO_URL"
    fi

    # Create a fresh, disconnected branch (no history)
    git checkout --orphan temp_latest_commit

    # Stage all files
    git add -A

    # Create single standalone commit
    git commit -m "Latest update: $(date -u '+%Y-%m-%d %H:%M UTC')"

    # Replace local main with temp branch
    git branch -D $BRANCH_NAME > /dev/null 2>&1 || true
    git branch -m $BRANCH_NAME

    # Force-push single commit to GitHub (overwrites remote history entirely)
    echo "--> Force-pushing latest single commit to GitHub..."
    git push origin $BRANCH_NAME --force

    # --------------------------------------------------------------------------
    # 3. STORAGE CLEANUP (Post-Run Git & Local Disk Prune)
    # --------------------------------------------------------------------------
    echo "--> Aggressively cleaning local Git storage & temporary build files..."
    
    # Aggressively prune all dangling git objects
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive > /dev/null 2>&1 || true

    # Remove temporary raw GRIB data if download.py left any behind
    rm -f *.grib *.grib2 *.nc *.idx

    # --------------------------------------------------------------------------
    # 4. SLEEP TIMING CALCULATION (08:00 UTC & 20:00 UTC Targets)
    # --------------------------------------------------------------------------
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    NOW_UTC_SEC=$(date -u +%s)
    
    # Strip leading zeros safely for hour, min, sec
    CURRENT_HOUR=$(date -u +%H | sed 's/^0//')
    CURRENT_MIN=$(date -u +%M | sed 's/^0//')
    CURRENT_SEC=$(date -u +%S | sed 's/^0//')
    
    CURRENT_HOUR=${CURRENT_HOUR:-0}
    CURRENT_MIN=${CURRENT_MIN:-0}
    CURRENT_SEC=${CURRENT_SEC:-0}

    SECONDS_TODAY=$(( (CURRENT_HOUR * 3600) + (CURRENT_MIN * 60) + CURRENT_SEC ))
    
    # Target windows in seconds from 00:00 UTC:
    # 08:00 UTC = 28,800s  (For 00Z model run data)
    # 20:00 UTC = 72,000s  (For 12Z model run data)
    # Next day 08:00 UTC = 115,200s (86,400 + 28,800)

    if [ $SECONDS_TODAY -lt 28800 ]; then
        TARGET_SECONDS=28800
    elif [ $SECONDS_TODAY -lt 72000 ]; then
        TARGET_SECONDS=72000
    else
        TARGET_SECONDS=115200
    fi
    
    SLEEP_TIME=$((TARGET_SECONDS - SECONDS_TODAY))
    NEXT_RUN_TIMESTAMP=$((NOW_UTC_SEC + SLEEP_TIME))
    
    # macOS-compatible date string formatting from Unix timestamp
    NEXT_RUN_FORMATTED=$(date -u -r $NEXT_RUN_TIMESTAMP '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || date -u -d "@$NEXT_RUN_TIMESTAMP" '+%Y-%m-%d %H:%M:%S UTC')

    echo "=================================================="
    echo " Run Completed in: ${ELAPSED} seconds"
    echo " Next Target Window: ${NEXT_RUN_FORMATTED}"
    echo " Sleeping for: ${SLEEP_TIME} seconds ($((SLEEP_TIME / 3600))h $(((SLEEP_TIME % 3600) / 60))m)"
    echo "=================================================="

    sleep $SLEEP_TIME
done