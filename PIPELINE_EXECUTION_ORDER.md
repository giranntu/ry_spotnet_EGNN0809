# SpotV2Net Pipeline Execution Order

## Overview
This document describes the correct execution order for the SpotV2Net volatility forecasting pipeline.

## File Naming Convention
- **X_Y_name.py**: Core pipeline files (X = step, Y = substep)
- **X_Y_name_optional.py**: Optional analysis/validation files
- **X_Y_name_optuna.py**: Hyperparameter optimization versions

## 📋 Main Pipeline (Required Steps)

### Step 1: Data Collection
**File:** `1_fetch_polygon_data.py`
- **Purpose:** Download 1-minute OHLCV data from Polygon.io API
- **Input:** Polygon.io API
- **Output:** `rawdata/by_comp/SYMBOL_201901_202507.csv`
- **Time:** ~2-4 hours for 30 symbols
- **Key Features:**
  - Parallel downloading (30 workers)
  - Robust retry logic (10 attempts with exponential backoff)
  - AMZN replaces DOW for complete historical data

### Step 2: Volatility Calculation
**File:** `2_organize_prices_as_tables.py`
- **Purpose:** Calculate Yang-Zhang volatility estimator
- **Input:** `rawdata/by_comp/*.csv`
- **Output:** 
  - `processed_data/vol/*.csv` (individual volatilities)
  - `processed_data/vol_of_vol/*.csv` (volatility of volatility)
  - `processed_data/covol/*.csv` (covolatilities)
  - `processed_data/covol_of_vol/*.csv` (covolatility of volatility)
- **Time:** ~20-30 minutes
- **Key Parameters:**
  - Volatility window: 22 days
  - Covolatility window: 30 days

### Step 3: Matrix Construction
**File:** `3_1_create_matrix_dataset.py`
- **Purpose:** Create 30x30 covariance matrices from volatility data
- **Input:** `processed_data/{vol,vol_of_vol,covol,covol_of_vol}/*.csv`
- **Output:** 
  - `processed_data/vols_mats_taq.h5`
  - `processed_data/volvols_mats_taq.h5`
- **Time:** ~5-10 minutes
- **Format:** HDF5 with matrices indexed by time

### Step 4: Data Standardization
**File:** `4_1_standardize_data.py`
- **Purpose:** Standardize matrices with proper train/val/test splits
- **Input:** `processed_data/*_mats_taq.h5`
- **Output:** 
  - `processed_data/vols_mats_taq_standardized.h5`
  - `processed_data/volvols_mats_taq_standardized.h5`
  - `processed_data/*_mean_std_scalers.csv`
- **Time:** ~2-3 minutes
- **Splits:**
  - Train: matrices 0-1007 (2019-2022)
  - Validation: matrices 1008-1259 (2023)
  - Test: matrices 1260+ (2024-2025)

### Step 5: Model Training
Choose ONE of the following:

#### 5a. SpotV2Net (Graph Neural Network)
- **Basic:** `5_1_train_SpotV2Net.py`
- **Hyperparameter Tuning:** `5_2_train_SpotV2Net_optuna.py`
- **Config:** `config/GNN_param.yaml`
- **Output:** `output/MODEL_NAME/`

#### 5b. LSTM Baseline
- **Basic:** `5_3_train_LSTM.py`
- **Hyperparameter Tuning:** `5_4_train_LSTM_optuna.py`
- **Requires:** Run `4_2_prepare_lstm_data_optional.py` first
- **Output:** `output/LSTM_*/`

### Step 6: Evaluation
**File:** `6_results.ipynb`
- **Purpose:** Evaluate models and generate results
- **Benchmarks:** HAR, ARFIMA, XGBoost, LSTM, SpotV2Net
- **Metrics:** MSE, RMSE, QLIKE
- **Statistical Tests:** DM test, MCS

## 📊 Optional Analysis Files

### Data Validation
- `3_3_validate_data_pipeline_optional.py` - Validate each pipeline step
- `3_4_comprehensive_data_analysis_optional.py` - Detailed data analysis
- `6_1_validate_results_optional.py` - Final validation checks
- `6_2_compare_models_optional.py` - Model comparison analysis

### LSTM Data Preparation
- `4_2_prepare_lstm_data_optional.py` - Prepare data specifically for LSTM models

## 🚀 Quick Start Commands

### Full Pipeline Execution:
```bash
# Step 1: Download data (run once)
python 1_fetch_polygon_data.py

# Step 2: Calculate volatilities
python 2_organize_prices_as_tables.py

# Step 3: Create matrices
python 3_1_create_matrix_dataset.py

# Step 4: Standardize data
python 4_1_standardize_data.py

# Step 5: Train SpotV2Net
python 5_1_train_SpotV2Net.py

# Step 6: Evaluate results
jupyter notebook 6_results.ipynb
```

### With Hyperparameter Optimization:
```bash
# After steps 1-4, run:
python 5_2_train_SpotV2Net_optuna.py
```

### For LSTM Comparison:
```bash
# After step 4, prepare LSTM data:
python 4_2_prepare_lstm_data_optional.py

# Then train LSTM:
python 5_3_train_LSTM.py
# Or with optimization:
python 5_4_train_LSTM_optuna.py
```

## ⚠️ Important Notes

1. **Data Alignment:** All files use AMZN instead of DOW for complete historical coverage
2. **Temporal Integrity:** Scalers are fitted ONLY on training data to prevent leakage
3. **No Synthetic Data:** Pipeline uses only real market data
4. **Parallel Processing:** Step 1 uses 30 parallel workers - ensure stable internet
5. **Memory Requirements:** ~4GB RAM minimum, 8GB recommended
6. **GPU:** Optional for training, CPU mode is default

## 📁 Directory Structure

```
SpotV2Net/
├── rawdata/by_comp/           # Step 1 output
├── processed_data/
│   ├── vol/                   # Step 2 volatilities
│   ├── vol_of_vol/           # Step 2 vol-of-vol
│   ├── covol/                # Step 2 covolatilities
│   ├── covol_of_vol/         # Step 2 covol-of-vol
│   └── *.h5                  # Steps 3-4 matrices
├── output/                    # Step 5 models
└── config/
    ├── dow30_config.yaml      # Symbol list
    └── GNN_param.yaml         # Model parameters
```

## 🔧 Troubleshooting

- **API Rate Limits:** Step 1 has robust retry logic with exponential backoff
- **Memory Issues:** Reduce batch_size in GNN_param.yaml
- **Missing Data:** Check polygon_progress.json for download status
- **NaN Values:** Step 2 includes comprehensive NaN handling

## 📊 Expected Output Sizes

- Raw data: ~1-2GB (30 symbols × 6.5 years)
- Processed matrices: ~100-200MB
- Trained models: ~10-50MB
- Total disk space needed: ~3GB

## 🎯 Research Alignment

This pipeline implements the SpotV2Net architecture for multivariate volatility forecasting using:
- Yang-Zhang volatility estimator (replaces MATLAB FMVol)
- Graph Neural Networks with attention mechanisms
- Proper temporal splits for financial time series
- Multiple benchmark comparisons (HAR, LSTM, XGBoost)