# SpotV2Net - Complete Execution Instructions

## ✅ Environment Cleaned
- All training processes stopped
- All processed data removed
- All model checkpoints deleted
- All log files cleaned

---

## 📋 Step-by-Step Execution Guide

### Prerequisites Check:
```bash
# Verify raw data exists (Step 1 already completed)
ls -la rawdata/by_comp/*.csv | wc -l
# Should show 30 files

# Verify Python environment
python --version  # Should be 3.8+
```

---

## 🚀 MAIN PIPELINE EXECUTION

### Step 2: Calculate Volatilities (~20-30 minutes)
```bash
python 2_organize_prices_as_tables.py
```
**Expected Output:**
- Creates 4 folders in `processed_data/`:
  - `vol/` - 30 individual volatility files
  - `vol_of_vol/` - 30 volatility-of-volatility files
  - `covol/` - 435 covolatility files (pairs)
  - `covol_of_vol/` - 435 covolatility-of-volatility files
- Progress bars for each calculation
- Summary: "Generated X covariance matrix time points"

### Step 3: Create Matrix Datasets (~5-10 minutes)
```bash
python 3_create_matrix_dataset.py
```
**Expected Output:**
- Creates 2 HDF5 files in `processed_data/`:
  - `vols_mats_taq.h5` - Volatility matrices
  - `volvols_mats_taq.h5` - Vol-of-vol matrices
- Shows progress bars for matrix creation
- Displays "Vol has 2000 obs"

### Step 4: Standardize Data (~2-3 minutes)
```bash
python 4_standardize_data.py
```
**Expected Output:**
- Creates standardized HDF5 files:
  - `vols_mats_taq_standardized.h5`
  - `volvols_mats_taq_standardized.h5`
- Creates scaler files:
  - `vols_mean_std_scalers.csv`
  - `volvols_mean_std_scalers.csv`
- Shows temporal splits:
  - Training: 0-1008 (1008 matrices)
  - Validation: 1008-1260 (252 matrices)
  - Test: 1260-2000 (740 matrices)

---

## 🎯 MODEL TRAINING (Enhanced Versions)

### Step 5A: Train SpotV2Net with Early Stopping (~2-4 hours)
```bash
python 5_train_SpotV2Net_enhanced.py
```
**Features:**
- Automatic early stopping (patience=10 epochs)
- Checkpoints saved every 10 epochs
- Best model saved automatically
- Training curves plotted every 5 epochs

**Monitor Progress:**
- Check `output/20240525_RGNN_std_optuna_42/training_curves.png`
- Watch console for epoch updates
- Early stopping will trigger if no improvement

### Step 5B: Train LSTM (~1-2 hours)
```bash
python 5_train_LSTM_enhanced.py
```
**Features:**
- Checkpoints saved every 10 epochs
- Best model saved automatically
- Training curves plotted every 5 epochs

**Monitor Progress:**
- Check `output/LSTM_42/training_curves.png`
- Watch console for epoch updates

---

## 📊 EVALUATION

### Step 6: Evaluate All Models
```bash
python 6_evaluate_all_models.py
```
**Expected Output:**
- Comparison table with metrics:
  - RMSE, MAE, R², QLIKE
  - Mincer-Zarnowitz α and β
  - Directional Accuracy (MDA)
  - Mean Percentage Error (MPE)
- Creates `volatility_predictions_sample.png`
- Saves results to `evaluation_results_TIMESTAMP.json`

---

## 🔥 QUICK EXECUTION (All Steps)

### Copy and run this complete pipeline:
```bash
# Step 2: Volatility Calculation
echo "Starting Step 2: Volatility Calculation..."
python 2_organize_prices_as_tables.py

# Step 3: Matrix Creation
echo "Starting Step 3: Matrix Creation..."
python 3_create_matrix_dataset.py

# Step 4: Standardization
echo "Starting Step 4: Data Standardization..."
python 4_standardize_data.py

# Step 5A: Train SpotV2Net (run in background)
echo "Starting SpotV2Net Training..."
nohup python 5_train_SpotV2Net_enhanced.py > spotv2net.log 2>&1 &
echo "SpotV2Net training started in background. Check spotv2net.log"

# Step 5B: Train LSTM (run in background)
echo "Starting LSTM Training..."
nohup python 5_train_LSTM_enhanced.py > lstm.log 2>&1 &
echo "LSTM training started in background. Check lstm.log"

# Monitor training
echo "Training started. Monitor with:"
echo "  tail -f spotv2net.log"
echo "  tail -f lstm.log"
```

---

## 📈 MONITORING TRAINING

### Check Training Status:
```bash
# View SpotV2Net progress
tail -f spotv2net.log

# View LSTM progress
tail -f lstm.log

# Check if processes are running
ps aux | grep python | grep train

# View training curves (updates every 5 epochs)
ls -la output/*/training_curves.png
```

### Expected Training Times:
- SpotV2Net: ~2-4 hours (may stop early)
- LSTM: ~1-2 hours (100 epochs)

---

## ✅ VERIFICATION CHECKLIST

After each step, verify:

### After Step 2:
```bash
ls processed_data/vol/*.csv | wc -l  # Should be 30
ls processed_data/covol/*.csv | wc -l  # Should be 435
```

### After Step 3:
```bash
ls -lh processed_data/*.h5  # Should see 2 files ~15MB each
```

### After Step 4:
```bash
ls -lh processed_data/*standardized.h5  # Should see 2 standardized files
```

### During Training:
```bash
# Check for checkpoint files
ls output/*/checkpoint_*.pt
ls output/*/best_model.pt
ls output/*/training_curves.png
```

---

## 🚨 TROUBLESHOOTING

### If training stops unexpectedly:
```bash
# Check error messages
tail -100 spotv2net.log
tail -100 lstm.log

# Resume from checkpoint (models auto-load best checkpoint)
python 5_train_SpotV2Net_enhanced.py
python 5_train_LSTM_enhanced.py
```

### If out of memory:
```bash
# Edit config/GNN_param.yaml
# Reduce batch_size from 128 to 64 or 32
```

### If Step 2 shows NaN warnings:
- This is normal - Yang-Zhang estimator handles NaN robustly
- Continue to next step

---

## 📊 EXPECTED FINAL RESULTS

After full training, you should see:

### Model Performance (approximate):
- **Naive Persistence**: RMSE ~0.028-0.030
- **HAR Model**: RMSE ~0.065-0.075
- **EWMA**: RMSE ~0.040-0.050
- **LSTM**: RMSE ~0.030-0.040
- **SpotV2Net**: RMSE ~0.025-0.035

### Key Files Created:
- `output/*/best_model.pt` - Best model checkpoints
- `output/*/training_curves.png` - Training progress plots
- `evaluation_results_*.json` - Final metrics
- `volatility_predictions_sample.png` - Sample predictions

---

## 🎯 SUCCESS CRITERIA

Your pipeline is successful when:
1. ✅ All 4 data processing steps complete without errors
2. ✅ Both models train and save best checkpoints
3. ✅ Evaluation shows SpotV2Net/LSTM beating naive baseline
4. ✅ Training curves show convergence (decreasing loss)
5. ✅ No data leakage warnings in standardization

---

## 📝 NOTES

- **Total execution time**: ~4-6 hours
- **Disk space needed**: ~3-4 GB
- **RAM needed**: ~4-8 GB
- **GPU**: Not required (CPU training)

**Good luck with your research! The pipeline is ready for execution.**