# SpotV2Net - 30-Minute Intraday Volatility Prediction

Graph Neural Network (GNN) and LSTM models for high-frequency volatility forecasting on DOW30 stocks using Yang-Zhang estimator.

This repository supports the paper titled **"SpotV2Net: Multivariate Intraday Spot Volatility Forecasting via Vol-of-Vol-Informed Graph Attention Networks"**, authored by **Alessio Brini** and **Giacomo Toscano**. 

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Instructions](#instructions)
3. [Data Availability](#data-availability)
4. [Computational Resources](#computational-resources)

## Repository Structure

The files in this repository are numbered in the order they should be executed. Below is the structure of the repository:

- `config/` - Configuration files.
- `rawdata/` - Data folder with example of data structure before preprocessing.
- `processed_data/` - Data folder with example of data structure after preprocessing.
- `utils/` - Utility functions for the following scripts.
- `1_downsample_TAQ_data.py` - First script to downsample TAQ data.
- `2_organize_prices_as_tables.py` - Organize prices into tables.
- `3_create_matrix_dataset.py` - Create the volatility and co-volatility of volatility matrix to be passed to the Pytorch Geometric dataset constructor.
- `4_standardize_data.py` - Standardize the data for neural network training.
- `5_train_LSTM_optuna.py` - Train LSTM model using `Optuna` (hyperparameter optimization).
- `5_train_SpotV2Net.py` - Train the RGNN model (single run, no hyperparameter optimization).
- `5_train_SpotV2Net_optuna.py` - Train RGNN using `Optuna` (hyperparameter optimization).
- `6_results.ipynb` - Jupyter notebook containing results.

## Instructions

The files are numbered to indicate the sequence in which they should be executed. Specific instructions are also embedded within each file. This section provides general guidance on what each file does and how to run them in order:

1. **First and Second Scripts**:
   - Start by running `1_fetch_polygon_data.py` followed by `2_organize_prices_as_tables.py` in sequence. These scripts now fetch 1-minute data from Polygon.io API for the 30 DJIA constituents over a 6+ year period (2019-2025).
   - The Polygon.io API key is already configured in the code for your research use.
   - The first script uses **ultra-fast parallel processing** with 30 workers per symbol and up to 5 concurrent symbols, making ~45,000+ API requests efficiently with rate limiting and retry logic.
   - Data is fetched for NYSE during market hours and stored in the folder `rawdata/polygon/`. This folder will contain 30 files, one for each company, named in the format `{COMPANY_NAME}_2019_2025.csv`.
   - The second script implements the **Yang-Zhang volatility estimator** to replace the MATLAB FMVol library, providing a more robust volatility estimation that accounts for overnight returns, opening jumps, and intraday returns.
   - The results will be saved into four structured folders:
     - `processed_data/vol/` - Univariate volatilities (30 files, one per company).
     - `processed_data/covol/` - Multivariate co-volatilities (435 files, one for each entry in a 30x30 upper triangular matrix).
     - `processed_data/vol_of_vol/` - Univariate volatility of volatilities (30 files, one per company).
     - `processed_data/covol_of_vol/` - Multivariate co-volatility of volatilities (similar to the covolatility folder, 435 files).

2. **Third Script**:
   - The next step is to run `3_create_matrix_dataset.py`. This script starts from the structured data in the four folders mentioned above. It aggregates the volatilities, co-volatilities, and volatility of volatility into sequences of matrices. These matrices will be used in later steps to construct the graph dataset, as described in the paper.

3. **Fourth Script**:
   - Run `4_standardize_data.py` to standardize the data for neural network training with proper temporal splits. This script now implements a proper train/validation/test split for the 6+ year timeframe:
     - **Training**: 2019-2025 first 4 years (~1008 trading days)  
     - **Validation**: Next 1 year (~252 trading days)
     - **Test**: Remaining 1.5+ years
   - The script saves standardization parameters fitted only on training data to prevent data leakage.

4. **Fifth Set of Scripts**:
   - There are three options for training neural network models in step 5:
     - `5_train_LSTM_optuna.py` uses the `Optuna` Python package for hyperparameter optimization of the LSTM model.
     - `5_train_SpotV2Net.py` trains the `SpotV2Net` model in a single run, without hyperparameter optimization.
     - `5_train_SpotV2Net_optuna.py` performs hyperparameter optimization on `SpotV2Net` using `Optuna`.
   - The configuration for these training processes is controlled by the YAML file in the `config/` folder. If running the `Optuna` version of `SpotV2Net`, the choice of the hyperparameter grid (specified below line 40) becomes important. For a single run, select specific hyperparameters above line 40.
   - The YAML file also specifies which data in H5 format (produced by `3_create_matrix_dataset.py`) to use for training.
   - The train scripts for `SpotV2Net` will also generate the graph-structured dataset, which may take at least 5 minutes the first time. Once generated, the dataset will be cached, making it instantaneous to reload from the second time onwards. For more information about this caching mechanism, see the [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/) to understand how caching works.


5. **Sixth Script**:
   - `6_results.ipynb` is a Jupyter notebook that performs several tasks essential for generating the results presented in the paper:
     - It fits the Multivariate HAR model used in the paper. It also includes the hyperparameter optimization and testing of the `XGBoost` model, since this process does not require `Optuna` but can be done using `scikit-learn`.
     - The notebook provides evaluations for both single-step and multi-step forecasting models.
     - Additionally, it loads the results from the neural network runs, generates the figures from the paper, and produces values for the tables on losses, MCS, and DM tests.
   - The notebook is structured into clear sections to enhance readability, and it contains instructions for modifying parameters to explore different results.

## Data Availability

The data used in this refined implementation comes from **Polygon.io API**, which provides high-quality financial market data including 1-minute aggregated bars for US equities. 

- **Polygon.io**: Premium API access configured in the code for the full 6+ year historical data span (2019-2025)
- **Alternative**: The original implementation used Trade and Quote (TAQ) database via WRDS, but this has been replaced with Polygon.io for better accessibility and real-time capabilities
- **API Configuration**: The Polygon.io API key is already embedded in the code for your research use

## Computational Resources

For training the GNN and LSTM models, we used a server equipped with an **NVIDIA GeForce RTX 2080 Ti** with 12 GB of memory. These models require substantial computational resources, and we recommend using a similar or higher-spec GPU for efficient training.

The **HAR** model and **XGBoost** can be run locally on a standard laptop without the need for extensive hardware requirements.

The required packages and their versions for running the refined code are listed below:

- `requests`: 2.31.0 (for Polygon.io API)
- `h5py`: 3.11.0
- `numpy`: 1.26.4
- `optuna`: 3.6.1
- `pandas`: 2.2.2
- `pandas_market_calendars`: 4.4.0
- `scikit-learn`: 1.2.2
- `torch`: 2.2.2
- `torch_geometric`: 2.3.0
- `tqdm`: 4.66.2
- `statsmodels`: 0.13.5

## Key Improvements

1. **Data Source**: Replaced TAQ database with Polygon.io API for more accessible and real-time data
2. **Volatility Estimation**: Implemented Yang-Zhang estimator to replace MATLAB FMVol dependency
3. **Time Period**: Extended from 3 years (2020-2023) to 6+ years (2019-2025) for better model training
4. **Data Quality**: Using 1-minute bars instead of tick-by-tick data for better computational efficiency
5. **Temporal Splits**: Proper train/validation/test splits with no data leakage
6. **Robustness**: Yang-Zhang estimator accounts for overnight returns and market microstructure effects

## Rigorous Data Alignment

### Temporal Split Consistency (Critical for Research Integrity)

All models in this project use **identical temporal splits** to ensure fair comparison:

- **Training Set**: Matrices 0-1008 (≈50.4% of data, roughly 2019-2022)
- **Validation Set**: Matrices 1008-1260 (≈12.6% of data, roughly 2023)  
- **Test Set**: Matrices 1260-2000 (≈37% of data, roughly 2024-2025)

### Alignment Implementation

1. **Script 4 (`4_standardize_data.py`)**: 
   - Fits standardization scalers **only on training data** (matrices 0-1008)
   - Applies same scalers to validation and test sets
   - Prevents data leakage from future to past

2. **Script 5 (`5_train_SpotV2Net.py`)**: 
   - Uses 3-way split: train/validation/test
   - Model selection based on **validation set** performance
   - Final evaluation on **holdout test set**
   - Sample indices adjusted for sequence length (seq_length=42)

3. **Script 5 LSTM (`5_train_LSTM.py`)**: 
   - Uses **exact same splits** as GNN for fair comparison
   - Identical batch sizes, learning rates, and epochs
   - Same early stopping criteria on validation set

### Best Practices Implemented

- **No Data Leakage**: Future data never influences past predictions
- **Validation-Based Model Selection**: Best model chosen using validation set, not test set
- **Consistent Preprocessing**: All models receive identically standardized data
- **Temporal Ordering**: Data always processed in chronological order
- **Reproducible Seeds**: Fixed random seeds for deterministic results
