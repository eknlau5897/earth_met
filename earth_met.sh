#!/usr/bin/env bash

set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GITHUB_REPO_URL="https://github.com/eknlau5897/earth_met.git"
BRANCH_NAME="main"

echo "=================================================="
echo " 3D Wind Field Player Operational Loop"
echo " Target UTC Update Windows: 02:00, 08:00, 14:00, 20:00 UTC"
echo "=================================================="

while true; do
    START_TIME=$(date +%s)
    CURRENT_UTC=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    
    echo ""
    echo "=================================================="
    echo " Starting Pipeline Run at: ${CURRENT_UTC}"
    echo "=================================================="

    # 1. Ensure root texture directory exists
    mkdir -p textures

    # 2. Run Python download and raster processing pipeline
    if [ -f "download.py" ]; then
        echo "--> Executing download.py..."
        python download.py
    else
        echo "⚠️ WARNING: download.py not found! Skipping data download..."
    fi

    # 3. PURGE HISTORY & KEEP ONLY LATEST COMMIT
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

    # 4. Force-push single commit to GitHub (overwrites remote history entirely)
    echo "--> Force-pushing latest single commit to GitHub..."
    git push  origin $BRANCH_NAME --force

    # Clean up local unreferenced objects to keep Mac storage clean
    git gc --prune=now --aggressive > /dev/null 2>&1 || true

    # 5. Calculate Runtime and Next Target Window (02, 08, 14, 20 UTC)
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    NOW_UTC_SEC=$(date -u +%s)
    CURRENT_HOUR=$(date -u +%H)
    CURRENT_MIN=$(date -u +%M)
    CURRENT_SEC=$(date -u +%S)
    
    SECONDS_TODAY=$(( (10#$CURRENT_HOUR * 3600) + (10#$CURRENT_MIN * 60) + 10#$CURRENT_SEC ))
    
    if [ $SECONDS_TODAY -lt 28800 ]; then
        TARGET_SECONDS=28800
    elif [ $SECONDS_TODAY -lt 72000 ]; then
        TARGET_SECONDS=72000
    else
        TARGET_SECONDS=115200
    fi
    
    SLEEP_TIME=$((TARGET_SECONDS - SECONDS_TODAY))
    NEXT_RUN_TIMESTAMP=$((NOW_UTC_SEC + SLEEP_TIME))
    NEXT_RUN_FORMATTED=$(date -u -r $NEXT_RUN_TIMESTAMP '+%Y-%m-%d %H:%M:%S UTC')

    echo "=================================================="
    echo " Run Completed in: ${ELAPSED} seconds"
    echo " Next Target Window: ${NEXT_RUN_FORMATTED}"
    echo " Sleeping for: ${SLEEP_TIME} seconds ($((SLEEP_TIME / 3600))h $(((SLEEP_TIME % 3600) / 60))m)"
    echo "=================================================="

    sleep $SLEEP_TIME
done