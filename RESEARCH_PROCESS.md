# SpotV2Net Research Pipeline - Complete Process Guide

## Overview
This document provides the complete process for running the SpotV2Net volatility forecasting research pipeline, from raw Polygon.io data to trained Graph Neural Network models.

## Prerequisites

### System Requirements
- Python 3.8+
- 16GB+ RAM recommended
- GPU with CUDA support (optional but recommended for training)
- 50GB+ free disk space

### Python Dependencies
```bash
pip install pandas numpy torch h5py tqdm scikit-learn matplotlib seaborn
pip install torch-geometric  # For GNN models
```

## Step-by-Step Process

### Step 1: Data Download (Currently in Progress)
The raw 1-minute OHLCV data from Polygon.io is being downloaded to `rawdata/by_comp/`.
Expected files: `{SYMBOL}_201901_202507.csv` for each DJIA 30 component.

### Step 2: Clean Environment (Optional but Recommended)
For a fresh start, remove all processed data and checkpoints:
```bash
./clean_environment.sh
```

This will remove:
- `processed_data/` directory
- All `.h5` matrix files
- Model checkpoints
- Python cache files
- Debug/test files

**Note**: Raw data in `rawdata/` is preserved.

### Step 3: Run Complete Pipeline
Execute the master pipeline script:
```bash
./run_research_pipeline.sh
```

Select option based on your needs:
1. **Full pipeline** - Complete data processing + model training
2. **Data processing only** - Steps 2-4 (volatility calculation, matrix creation, standardization)
3. **Model training only** - Train models on existing processed data
4. **Clean start** - Clean environment + full pipeline

## Pipeline Components

### Data Processing Stage

#### 2. Yang-Zhang Volatility Calculation (`2_yang_zhang_volatility_refined.py`)
- **Input**: Raw 1-minute OHLCV data
- **Process**: 
  - Aggregates to 30-minute intervals
  - Calculates Yang-Zhang volatility with proper decomposition:
    - Overnight component (close-to-open gaps)
    - Opening component (auction effects)
    - Intraday component (Rogers-Satchell estimator)
  - Uses 22-day lookback for volatility estimation
  - Calculates volatility-of-volatility
- **Output**: 
  - Individual volatility series in `processed_data/vol/`
  - Vol-of-vol series in `processed_data/vol_of_vol/`

#### 3. Matrix Dataset Creation (`3_create_matrix_dataset.py`)
- **Input**: Volatility series from Step 2
- **Process**:
  - Creates 30x30 covariance matrices for each time point
  - Uses 30-day rolling windows
  - Ensures positive semi-definite matrices
- **Output**:
  - `processed_data/vols_mats_30min.h5` - Volatility covariance matrices
  - `processed_data/volvols_mats_30min.h5` - Vol-of-vol covariance matrices

#### 4. Standardization (`4_create_standardized_mats.py`)
- **Input**: HDF5 matrix files from Step 3
- **Process**:
  - Splits data into train/validation/test sets
  - Fits scaler only on training data (prevents leakage)
  - Standardizes all matrices
- **Output**: `vols_labels_30min.h5` - Standardized data ready for models

#### 5. HAR Baseline (`5_create_HAR_baseline.py`)
- Creates Heterogeneous Autoregressive baseline predictions
- Used for model performance comparison

### Model Training Stage

#### Available Models
1. **HAR** (`6_train_HAR.py`) - Traditional econometric baseline
2. **PNA** (`7_train_PNA.py`) - Principal Neighborhood Aggregation GNN
3. **GAT** (`train_GAT.py`) - Graph Attention Network
4. **GIN** (`train_GIN.py`) - Graph Isomorphism Network

Each model:
- Uses standardized matrices as input
- Implements early stopping
- Saves best checkpoint to `checkpoints/`
- Logs training metrics

## Key Algorithm Details

### Yang-Zhang Volatility Formula
```
σ²_YZ = σ²_overnight + k * σ²_open + σ²_rs

where:
- σ²_overnight = Var[ln(Open_t / Close_{t-1})]
- σ²_open = Var[ln(Close_first / Open_t)]
- σ²_rs = Rogers-Satchell intraday estimator
- k = 0.34 / (1.34 + (n+1)/(n-1)), n = lookback days
```

### Data Splits
- **Training**: 2019-2022 (4 years, ~1008 trading days)
- **Validation**: 2023 (1 year, ~252 trading days)
- **Test**: 2024-2025 (1.5+ years, remaining days)

### Matrix Structure
- 30x30 covariance matrices (DJIA 30 components)
- 30-minute interval frequency
- ~13 intervals per trading day
- Total ~1650 time points over 6.5 years

## Monitoring & Validation

### Check Data Quality
```python
# Verify volatility ranges (should be 0.10-0.80 typically)
import pandas as pd
vol_data = pd.read_csv('processed_data/vol/AAPL.csv', header=None)
print(f"Vol range: {vol_data.min().min():.3f} - {vol_data.max().max():.3f}")
```

### Monitor Training
- Watch loss convergence in terminal output
- Check `checkpoints/` for saved models
- Review validation metrics for overfitting

### Visualize Results
```bash
python utils_academic_plot.py  # Creates performance comparison plots
```

## Troubleshooting

### Common Issues

1. **Memory Error During Matrix Creation**
   - Reduce batch size in processing
   - Process symbols sequentially instead of parallel

2. **NaN Values in Volatility**
   - Check for missing data in raw files
   - Verify market hours filtering is correct

3. **Model Not Converging**
   - Adjust learning rate (typically 1e-3 to 1e-4)
   - Check data standardization
   - Verify no data leakage

4. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Use gradient accumulation
   - Switch to CPU if needed

### Quick Validation Commands
```bash
# Check raw data
ls -lh rawdata/by_comp/ | head -5

# Verify processed data
ls -lh processed_data/vol/ | wc -l  # Should be 30

# Check matrix files
h5ls processed_data/vols_mats_30min.h5 | head -10

# Validate standardized data
python -c "import h5py; f=h5py.File('vols_labels_30min.h5','r'); print(f.keys())"
```

## Expected Outputs

### After Data Processing
```
processed_data/
├── vol/              # 30 CSV files (one per symbol)
├── vol_of_vol/       # 30 CSV files
├── vols_mats_30min.h5       # ~100MB
├── volvols_mats_30min.h5    # ~100MB
└── metadata.json
```

### After Training
```
checkpoints/
├── har_best.pt      # ~10MB
├── pna_best.pt      # ~50MB
├── gat_best.pt      # ~40MB
└── gin_best.pt      # ~35MB
```

## Research Notes

### Key Improvements in This Implementation
1. **Theoretically Correct Yang-Zhang**: Proper decomposition of volatility components
2. **No Data Leakage**: Strict temporal splits, scaler fitted only on training
3. **Robust Processing**: Comprehensive NaN/zero handling
4. **Production Quality**: Progress bars, error handling, logging
5. **Reproducibility**: Fixed random seeds, documented parameters

### Performance Expectations
- HAR baseline QLIKE: ~0.8-1.2
- GNN models should improve by 10-30% over HAR
- PNA typically performs best on this task
- Training time: 1-3 hours per model (GPU), 5-10 hours (CPU)

## Contact & Support
For issues or questions about the pipeline, refer to:
- This documentation
- Code comments in individual scripts
- CLAUDE.md for implementation details

---

Last Updated: 2025
Research Pipeline Version: 2.0 (Refined Yang-Zhang)