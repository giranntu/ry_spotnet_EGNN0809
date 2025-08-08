# Codebase Cleanup Plan

## Files to KEEP (Core Pipeline):
1. `1_fetch_polygon_data.py` - Main data downloader ✅
2. `2_organize_prices_as_tables.py` - Yang-Zhang volatility ✅
3. `3_create_matrix_dataset.py` - Matrix construction ✅
4. `4_standardize_data.py` - Data standardization ✅
5. `5_train_SpotV2Net.py` - Main GNN training ✅
6. `5_train_LSTM.py` - Main LSTM training ✅

## Files to REMOVE (Duplicates/Unused):
1. `1_quick_validate.py` - Quick hack, remove
2. `1_validate_and_repair_data.py` - Duplicate of validate_downloaded_data
3. `1_validate_downloaded_data.py` - Keep one validator only
4. `3_3_validate_data_pipeline_optional.py` - Optional, remove
5. `3_4_comprehensive_data_analysis_optional.py` - Optional, remove
6. `4_2_prepare_lstm_data_optional.py` - Not needed, LSTM uses same data
7. `5_2_train_SpotV2Net_optuna.py` - Keep for later hyperparameter tuning
8. `5_4_train_LSTM_optuna.py` - Keep for later hyperparameter tuning
9. `6_1_validate_results_optional.py` - Optional, remove
10. `6_2_compare_models_optional.py` - Optional, remove
11. `6_evaluate_models.py` - Duplicate evaluation
12. `6_evaluate_trained_models.py` - Duplicate evaluation

## Files to CREATE/REFINE:
1. `1_validate_data.py` - Consolidate all validation
2. `5_train_SpotV2Net_enhanced.py` - Add checkpointing & early stopping
3. `5_train_LSTM_enhanced.py` - Add checkpointing
4. `6_evaluate_all_models.py` - Unified evaluation with proper metrics
5. `7_plot_results.py` - Training curves and results visualization