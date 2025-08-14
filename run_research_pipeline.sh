#!/bin/bash

# ============================================================================
# SpotV2Net Research Pipeline Execution Script
# ============================================================================
# Complete pipeline from raw Polygon.io data to trained models
# Includes Yang-Zhang volatility estimation and Graph Neural Network training
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Timer function
timer_start() {
    START_TIME=$(date +%s)
}

timer_end() {
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo -e "${CYAN}⏱️  Execution time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Header
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║         SpotV2Net Research Pipeline Executor              ║${NC}"
echo -e "${MAGENTA}║      Yang-Zhang Volatility → GNN Forecasting              ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python environment
echo -e "${BLUE}[Checking Environment]${NC}"
python_version=$(python3 --version 2>&1)
echo -e "  Python: $python_version"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}  ⚠️  No virtual environment detected${NC}"
    echo -e "${YELLOW}     Consider activating a venv for dependency isolation${NC}"
else
    echo -e "${GREEN}  ✓ Virtual environment: $(basename $VIRTUAL_ENV)${NC}"
fi

# Check required Python packages
echo -e "${BLUE}  Checking required packages...${NC}"
required_packages=("pandas" "numpy" "torch" "h5py" "tqdm" "scikit-learn")
missing_packages=()

for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}    ✓ $package${NC}"
    else
        echo -e "${RED}    ✗ $package missing${NC}"
        missing_packages+=($package)
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo -e "${RED}Missing packages: ${missing_packages[*]}${NC}"
    echo -e "${YELLOW}Install with: pip install ${missing_packages[*]}${NC}"
    exit 1
fi

echo ""

# Check raw data exists
echo -e "${BLUE}[Checking Raw Data]${NC}"
if [ -d "rawdata/by_comp" ]; then
    raw_count=$(ls -1 rawdata/by_comp/*.csv 2>/dev/null | wc -l)
    if [ $raw_count -gt 0 ]; then
        echo -e "${GREEN}  ✓ Found $raw_count CSV files in rawdata/by_comp/${NC}"
    else
        echo -e "${RED}  ✗ No CSV files found in rawdata/by_comp/${NC}"
        echo -e "${YELLOW}    Please complete Step 1 (Polygon.io download) first${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ✗ Raw data directory not found: rawdata/by_comp/${NC}"
    echo -e "${YELLOW}    Please complete Step 1 (Polygon.io download) first${NC}"
    exit 1
fi

echo ""

# Ask user for execution mode
echo -e "${BLUE}[Execution Options]${NC}"
echo "  1) Full pipeline (Steps 2-5 + model training)"
echo "  2) Data processing only (Steps 2-4)"
echo "  3) Model training only (Step 5 + training)"
echo "  4) Clean start (clean + full pipeline)"
echo ""
read -p "Select option [1-4]: " option

case $option in
    1)
        mode="full"
        echo -e "${GREEN}  → Running full pipeline${NC}"
        ;;
    2)
        mode="data"
        echo -e "${GREEN}  → Running data processing only${NC}"
        ;;
    3)
        mode="model"
        echo -e "${GREEN}  → Running model training only${NC}"
        ;;
    4)
        mode="clean_full"
        echo -e "${YELLOW}  → Clean start: removing old data first${NC}"
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""

# Clean if requested
if [ "$mode" == "clean_full" ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Step 0: Cleaning Environment${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -f "clean_environment.sh" ]; then
        echo "yes" | ./clean_environment.sh
    else
        echo -e "${YELLOW}Manually cleaning directories...${NC}"
        rm -rf processed_data checkpoints results cache __pycache__
        rm -f *.h5 *.hdf5
    fi
    
    mode="full"
    echo ""
fi

# Start main pipeline
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting Pipeline Execution${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

timer_start

# Step 2: Yang-Zhang Volatility Calculation
if [ "$mode" == "full" ] || [ "$mode" == "data" ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Step 2: Yang-Zhang Volatility Estimation${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -f "2_yang_zhang_volatility_refined.py" ]; then
        echo -e "${BLUE}Running refined Yang-Zhang implementation...${NC}"
        python3 2_yang_zhang_volatility_refined.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Yang-Zhang volatility calculation complete${NC}"
        else
            echo -e "${RED}❌ Yang-Zhang calculation failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ File not found: 2_yang_zhang_volatility_refined.py${NC}"
        exit 1
    fi
    echo ""
fi

# Step 3: Create Matrix Dataset
if [ "$mode" == "full" ] || [ "$mode" == "data" ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Step 3: Creating Matrix Dataset${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -f "3_create_matrix_dataset.py" ]; then
        echo -e "${BLUE}Creating volatility and covariance matrices...${NC}"
        python3 3_create_matrix_dataset.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Matrix dataset created${NC}"
        else
            echo -e "${RED}❌ Matrix dataset creation failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ File not found: 3_create_matrix_dataset.py${NC}"
        exit 1
    fi
    echo ""
fi

# Step 4: Create Standardized Matrices
if [ "$mode" == "full" ] || [ "$mode" == "data" ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Step 4: Standardizing Matrices${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -f "4_create_standardized_mats.py" ]; then
        echo -e "${BLUE}Standardizing matrices for model input...${NC}"
        python3 4_create_standardized_mats.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Matrices standardized${NC}"
        else
            echo -e "${RED}❌ Matrix standardization failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ File not found: 4_create_standardized_mats.py${NC}"
        exit 1
    fi
    echo ""
fi

# Step 5: Create HAR Baseline (Optional)
if [ "$mode" == "full" ] || [ "$mode" == "data" ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Step 5: Creating HAR Baseline${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -f "5_create_HAR_baseline.py" ]; then
        echo -e "${BLUE}Computing HAR baseline model...${NC}"
        python3 5_create_HAR_baseline.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ HAR baseline created${NC}"
        else
            echo -e "${YELLOW}⚠️  HAR baseline failed (non-critical)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  HAR baseline script not found (skipping)${NC}"
    fi
    echo ""
fi

# Model Training
if [ "$mode" == "full" ] || [ "$mode" == "model" ]; then
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}Step 6: Model Training${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check for GPU
    echo -e "${BLUE}Checking GPU availability...${NC}"
    gpu_check=$(python3 -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
    
    if [ "$gpu_check" == "GPU" ]; then
        gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "${GREEN}  ✓ GPU available: $gpu_name${NC}"
    else
        echo -e "${YELLOW}  ⚠️  No GPU detected, will use CPU (slower)${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Select model to train:${NC}"
    echo "  1) HAR - Heterogeneous Autoregressive (baseline)"
    echo "  2) PNA - Principal Neighborhood Aggregation GNN"
    echo "  3) GAT - Graph Attention Network"
    echo "  4) GIN - Graph Isomorphism Network"
    echo "  5) All models sequentially"
    echo ""
    read -p "Select model [1-5]: " model_choice
    
    case $model_choice in
        1)
            models=("HAR")
            ;;
        2)
            models=("PNA")
            ;;
        3)
            models=("GAT")
            ;;
        4)
            models=("GIN")
            ;;
        5)
            models=("HAR" "PNA" "GAT" "GIN")
            ;;
        *)
            echo -e "${RED}Invalid model choice${NC}"
            exit 1
            ;;
    esac
    
    # Train selected models
    for model in "${models[@]}"; do
        echo ""
        echo -e "${CYAN}Training $model model...${NC}"
        echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
        
        case $model in
            "HAR")
                if [ -f "6_train_HAR.py" ]; then
                    python3 6_train_HAR.py
                else
                    echo -e "${YELLOW}⚠️  HAR training script not found${NC}"
                fi
                ;;
            "PNA")
                if [ -f "7_train_PNA.py" ]; then
                    python3 7_train_PNA.py
                else
                    echo -e "${YELLOW}⚠️  PNA training script not found${NC}"
                fi
                ;;
            "GAT")
                if [ -f "train_GAT.py" ]; then
                    python3 train_GAT.py
                else
                    echo -e "${YELLOW}⚠️  GAT training script not found${NC}"
                fi
                ;;
            "GIN")
                if [ -f "train_GIN.py" ]; then
                    python3 train_GIN.py
                else
                    echo -e "${YELLOW}⚠️  GIN training script not found${NC}"
                fi
                ;;
        esac
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ $model training complete${NC}"
        else
            echo -e "${RED}❌ $model training failed${NC}"
        fi
    done
fi

echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}Pipeline Execution Complete${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"

timer_end

# Summary
echo ""
echo -e "${BLUE}[Summary]${NC}"

# Check outputs
if [ -d "processed_data" ]; then
    echo -e "${GREEN}  ✓ Processed data created${NC}"
    
    if [ -f "processed_data/vols_mats_30min.h5" ]; then
        echo -e "${GREEN}    ✓ Volatility matrices: processed_data/vols_mats_30min.h5${NC}"
    fi
    
    if [ -f "processed_data/volvols_mats_30min.h5" ]; then
        echo -e "${GREEN}    ✓ Vol-of-vol matrices: processed_data/volvols_mats_30min.h5${NC}"
    fi
fi

if [ -f "vols_labels_30min.h5" ]; then
    echo -e "${GREEN}  ✓ Standardized data: vols_labels_30min.h5${NC}"
fi

if [ -d "checkpoints" ]; then
    echo -e "${GREEN}  ✓ Model checkpoints saved${NC}"
    ls -la checkpoints/*.pt 2>/dev/null | head -5
fi

echo ""
echo -e "${BLUE}[Next Steps]${NC}"
echo -e "  1. Review results in checkpoints/ directory"
echo -e "  2. Analyze model performance metrics"
echo -e "  3. Run utils_academic_plot.py for visualizations"
echo -e "  4. Compare model predictions with baselines"

echo ""
echo -e "${GREEN}✨ Research pipeline complete! ✨${NC}"
echo ""