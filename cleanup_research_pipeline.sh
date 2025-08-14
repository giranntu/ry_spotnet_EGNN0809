#!/bin/bash

# ==============================================================================
# Research Pipeline Cleanup Script
# ==============================================================================
# This script cleans up all processed data, model checkpoints, and caches
# to allow a fresh start from Step 2 (after Polygon data download)
#
# Usage: ./cleanup_research_pipeline.sh [--all|--data|--models|--cache]
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to confirm action
confirm_action() {
    read -p "Are you sure you want to $1? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled."
        exit 0
    fi
}

# Function to clean processed data
clean_processed_data() {
    print_info "Cleaning processed data..."
    
    # Remove processed_data directory contents but keep the directory
    if [ -d "processed_data" ]; then
        print_info "Removing processed volatility data..."
        rm -rf processed_data/vol/
        rm -rf processed_data/vol_of_vol/
        rm -rf processed_data/covol/
        rm -rf processed_data/covol_of_vol/
        rm -rf processed_data/vol_30min/
        rm -rf processed_data/vol_of_vol_30min/
        
        print_info "Removing HDF5 matrix files..."
        rm -f processed_data/*.h5
        
        print_info "Removing standardization scalers..."
        rm -f processed_data/*_scalers.csv
        rm -f processed_data/*_scaler.pkl
        
        print_info "Removing dataset caches..."
        rm -rf processed_data/*_train/
        rm -rf processed_data/*_val/
        rm -rf processed_data/*_test/
        rm -rf processed_data/cutting_edge_gnn*/
        rm -rf processed_data/transformer_gnn*/
        rm -rf processed_data/intraday_gnn*/
        
        print_info "Removing analysis results..."
        rm -f processed_data/*.json
        rm -f processed_data/*.png
        rm -f processed_data/*.csv
        
        print_success "Processed data cleaned!"
    else
        print_warning "processed_data directory not found."
    fi
}

# Function to clean model checkpoints
clean_models() {
    print_info "Cleaning model checkpoints and outputs..."
    
    # Remove models directory
    if [ -d "models" ]; then
        rm -rf models/*
        print_info "Removed models directory contents"
    fi
    
    # Remove output directory
    if [ -d "output" ]; then
        rm -rf output/*
        print_info "Removed output directory contents"
    fi
    
    # Remove any .pt or .pth files in root
    rm -f *.pt *.pth
    
    print_success "Model checkpoints cleaned!"
}

# Function to clean cache files
clean_cache() {
    print_info "Cleaning cache files..."
    
    # Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Jupyter checkpoints
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    
    # PyTorch cache
    if [ -d ".cache" ]; then
        rm -rf .cache/
    fi
    
    # Remove debug plots
    rm -rf debug_plots/
    rm -rf debug_plots_fixed/
    
    # Remove temporary analysis files
    rm -rf advanced_rv_results/
    rm -rf paper_assets/
    rm -rf rv_analysis_results/
    
    print_success "Cache files cleaned!"
}

# Function to clean backup files
clean_backups() {
    print_info "Cleaning backup files..."
    
    # Remove backup directories
    rm -rf processed_data_backup*/
    rm -rf archive_old_versions/
    rm -rf archived_*/
    
    print_success "Backup files cleaned!"
}

# Function to show disk usage before and after
show_disk_usage() {
    if command -v du &> /dev/null; then
        echo -e "\n${BLUE}Disk usage:${NC}"
        du -sh processed_data/ 2>/dev/null || echo "  processed_data/: Not found"
        du -sh models/ 2>/dev/null || echo "  models/: Not found"
        du -sh output/ 2>/dev/null || echo "  output/: Not found"
        du -sh .cache/ 2>/dev/null || echo "  .cache/: Not found"
    fi
}

# Main execution
echo "=============================================================="
echo "           RESEARCH PIPELINE CLEANUP SCRIPT"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "1_fetch_polygon_data.py" ]; then
    print_error "This script must be run from the SpotV2Net project root directory!"
    exit 1
fi

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments, clean everything
    confirm_action "clean ALL processed data, models, and caches"
    
    print_info "Starting complete cleanup..."
    show_disk_usage
    
    clean_processed_data
    clean_models
    clean_cache
    clean_backups
    
    echo -e "\n${GREEN}=============================================================="
    echo "                 CLEANUP COMPLETE!"
    echo "=============================================================="
    echo -e "${NC}"
    show_disk_usage
    
else
    case "$1" in
        --all)
            confirm_action "clean ALL data"
            show_disk_usage
            clean_processed_data
            clean_models
            clean_cache
            clean_backups
            ;;
        --data)
            confirm_action "clean processed data"
            clean_processed_data
            ;;
        --models)
            confirm_action "clean model checkpoints"
            clean_models
            ;;
        --cache)
            confirm_action "clean cache files"
            clean_cache
            ;;
        --help|-h)
            echo "Usage: $0 [--all|--data|--models|--cache]"
            echo ""
            echo "Options:"
            echo "  --all     Clean everything (data, models, cache)"
            echo "  --data    Clean only processed data"
            echo "  --models  Clean only model checkpoints"
            echo "  --cache   Clean only cache files"
            echo "  --help    Show this help message"
            echo ""
            echo "If no option is provided, all will be cleaned."
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
fi

print_info "Ready to start fresh from Step 2!"
print_info "Next: Run ./run_research_pipeline.sh to execute the full pipeline"

exit 0