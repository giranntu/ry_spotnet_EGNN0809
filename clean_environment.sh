#!/bin/bash

# ============================================================================
# Clean Environment Script for SpotV2Net Research Pipeline
# ============================================================================
# This script cleans all processed data, caches, and model checkpoints
# to allow a fresh start from Step 2 (after Polygon.io data download)
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}SpotV2Net Environment Cleanup Script${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to safely remove directory
safe_remove_dir() {
    local dir=$1
    local desc=$2
    
    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Removing $desc: $dir${NC}"
        rm -rf "$dir"
        echo -e "${GREEN}✓ Removed${NC}"
    else
        echo -e "${BLUE}ℹ $desc not found: $dir${NC}"
    fi
}

# Function to safely remove file
safe_remove_file() {
    local file=$1
    local desc=$2
    
    if [ -f "$file" ]; then
        echo -e "${YELLOW}Removing $desc: $file${NC}"
        rm -f "$file"
        echo -e "${GREEN}✓ Removed${NC}"
    else
        echo -e "${BLUE}ℹ $desc not found: $file${NC}"
    fi
}

# Function to remove files matching pattern
remove_pattern() {
    local pattern=$1
    local desc=$2
    
    local count=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l)
    
    if [ "$count" -gt 0 ]; then
        echo -e "${YELLOW}Removing $count $desc files matching: $pattern${NC}"
        find . -maxdepth 1 -name "$pattern" -exec rm -f {} \;
        echo -e "${GREEN}✓ Removed${NC}"
    else
        echo -e "${BLUE}ℹ No $desc files found: $pattern${NC}"
    fi
}

# Confirm before proceeding
echo -e "${RED}⚠️  WARNING: This will delete all processed data and model checkpoints!${NC}"
echo -e "${RED}   Raw data in 'rawdata/' will be preserved.${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}Cleanup cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Starting cleanup...${NC}"
echo ""

# 1. Remove processed data directories
echo -e "${BLUE}[1/7] Cleaning processed data directories...${NC}"
safe_remove_dir "processed_data" "processed data"
safe_remove_dir "data_cache" "data cache"
safe_remove_dir "cache" "cache directory"

# 2. Remove model checkpoints and results
echo -e "${BLUE}[2/7] Cleaning model checkpoints and results...${NC}"
safe_remove_dir "checkpoints" "model checkpoints"
safe_remove_dir "checkpoint" "checkpoint directory"
safe_remove_dir "results" "results"
safe_remove_dir "output" "output"
safe_remove_dir "logs" "logs"
safe_remove_dir "tensorboard_logs" "tensorboard logs"

# 3. Remove HDF5 matrix files
echo -e "${BLUE}[3/7] Cleaning HDF5 matrix files...${NC}"
remove_pattern "*.h5" "HDF5"
remove_pattern "*.hdf5" "HDF5"
safe_remove_file "vols_labels_30min.h5" "volatility labels"
safe_remove_file "vols_mats_30min.h5" "volatility matrices"
safe_remove_file "volvols_mats_30min.h5" "vol-of-vol matrices"

# 4. Remove test and debug files
echo -e "${BLUE}[4/7] Cleaning test and debug files...${NC}"
remove_pattern "*debug*.py" "debug"
remove_pattern "*test*.py" "test"
remove_pattern "*_fixed.py" "fixed version"
remove_pattern "*_old.py" "old version"
remove_pattern "*backup*.py" "backup"
remove_pattern "*validation*.py" "validation"

# 5. Remove Python cache
echo -e "${BLUE}[5/7] Cleaning Python cache...${NC}"
safe_remove_dir "__pycache__" "Python cache"
safe_remove_dir ".pytest_cache" "pytest cache"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Python cache cleaned${NC}"

# 6. Remove temporary files
echo -e "${BLUE}[6/7] Cleaning temporary files...${NC}"
remove_pattern "*.tmp" "temporary"
remove_pattern "*.temp" "temporary"
remove_pattern ".DS_Store" "macOS metadata"
remove_pattern "*~" "backup"

# 7. Clean duplicated/unnecessary Python files
echo -e "${BLUE}[7/7] Cleaning duplicated Python files...${NC}"
safe_remove_file "calculate_intraday_volatility.py" "old intraday volatility calculator"
safe_remove_file "calculate_intraday_volatility_fixed.py" "fixed intraday volatility"
safe_remove_file "advanced_realized_volatility.py" "old advanced volatility"
safe_remove_file "2_organize_prices_yang_zhang_corrected.py" "corrected YZ (superseded)"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Environment cleanup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Show current state
echo -e "${BLUE}Current directory state:${NC}"
echo -e "${BLUE}------------------------${NC}"

# Check if raw data exists
if [ -d "rawdata/by_comp" ]; then
    count=$(ls -1 rawdata/by_comp/*.csv 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Raw data preserved: $count CSV files in rawdata/by_comp/${NC}"
else
    echo -e "${YELLOW}⚠️  Raw data directory not found: rawdata/by_comp/${NC}"
    echo -e "${YELLOW}   Please ensure Step 1 (Polygon.io download) is complete${NC}"
fi

# Check key files exist
echo ""
echo -e "${BLUE}Key pipeline files:${NC}"
for file in "2_yang_zhang_volatility_refined.py" "3_create_matrix_dataset.py" "4_create_standardized_mats.py" "5_create_HAR_baseline.py"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file missing${NC}"
    fi
done

echo ""
echo -e "${BLUE}Ready to run the pipeline from Step 2!${NC}"
echo -e "${BLUE}Use: ./run_research_pipeline.sh${NC}"
echo ""