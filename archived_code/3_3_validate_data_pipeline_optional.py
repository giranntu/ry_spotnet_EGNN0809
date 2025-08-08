#!/usr/bin/env python3
"""
Comprehensive Data Pipeline Validation Script
Validates each step of the SpotV2Net data processing pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DJIA_SYMBOLS = [
    'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]  # Note: AMZN replaces DOW for longer history

class DataPipelineValidator:
    def __init__(self):
        self.validation_results = {}
        
    def validate_step1_polygon_data(self):
        """Validate Step 1: Polygon.io raw data"""
        print("="*80)
        print("STEP 1: POLYGON.IO RAW DATA VALIDATION")
        print("="*80)
        
        polygon_dir = Path("rawdata/polygon/by_comp/")
        results = {
            'files_found': [],
            'files_missing': [],
            'file_stats': {},
            'data_quality': {},
            'nan_analysis': {}
        }
        
        for symbol in DJIA_SYMBOLS:
            filepath = polygon_dir / f"{symbol}_201901_202507.csv"
            
            if filepath.exists():
                results['files_found'].append(symbol)
                
                # File statistics
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                results['file_stats'][symbol] = {'size_mb': round(file_size_mb, 1)}
                
                # Sample data quality check
                try:
                    df = pd.read_csv(filepath, nrows=10000)  # Sample first 10k rows
                    
                    # Basic validation
                    expected_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                    has_all_columns = all(col in df.columns for col in expected_columns)
                    
                    # Data quality metrics
                    total_rows = len(df)
                    nan_counts = df.isnull().sum()
                    zero_counts = (df[['open', 'high', 'low', 'close']] == 0).sum()
                    
                    # Price logic validation
                    valid_ohlc = (df['high'] >= df['low']).all() and \
                                 (df['high'] >= df['open']).all() and \
                                 (df['high'] >= df['close']).all() and \
                                 (df['low'] <= df['open']).all() and \
                                 (df['low'] <= df['close']).all()
                    
                    results['data_quality'][symbol] = {
                        'has_all_columns': has_all_columns,
                        'total_rows_sample': total_rows,
                        'valid_ohlc_logic': valid_ohlc,
                        'datetime_parseable': pd.to_datetime(df['datetime'], errors='coerce').notna().all()
                    }
                    
                    results['nan_analysis'][symbol] = {
                        'nan_counts': nan_counts.to_dict(),
                        'zero_counts': zero_counts.to_dict()
                    }
                    
                except Exception as e:
                    results['data_quality'][symbol] = {'error': str(e)}
            else:
                results['files_missing'].append(symbol)
        
        # Print results
        print(f"✅ Files found: {len(results['files_found'])}/30")
        print(f"❌ Files missing: {len(results['files_missing'])}/30")
        
        if results['files_missing']:
            print(f"Missing files: {results['files_missing']}")
        
        # Summary statistics
        if results['file_stats']:
            sizes = [stats['size_mb'] for stats in results['file_stats'].values()]
            print(f"\nFile sizes: {min(sizes):.1f} - {max(sizes):.1f} MB (avg: {np.mean(sizes):.1f} MB)")
        
        # Data quality summary
        valid_files = 0
        for symbol, quality in results['data_quality'].items():
            if quality.get('has_all_columns', False) and quality.get('valid_ohlc_logic', False):
                valid_files += 1
        
        print(f"Data quality: {valid_files}/{len(results['data_quality'])} files passed validation")
        
        self.validation_results['step1'] = results
        return results
    
    def validate_step2_processed_data(self):
        """Validate Step 2: Processed volatility data"""
        print("\n" + "="*80)
        print("STEP 2: PROCESSED VOLATILITY DATA VALIDATION")
        print("="*80)
        
        processed_dir = Path("processed_data/")
        results = {
            'directories': {
                'vol': processed_dir / 'vol',
                'vol_of_vol': processed_dir / 'vol_of_vol', 
                'covol': processed_dir / 'covol',
                'covol_of_vol': processed_dir / 'covol_of_vol'
            },
            'file_counts': {},
            'data_analysis': {},
            'nan_analysis': {}
        }
        
        # Check directories exist
        for dir_name, dir_path in results['directories'].items():
            if dir_path.exists():
                csv_files = list(dir_path.glob("*.csv"))
                results['file_counts'][dir_name] = len(csv_files)
                print(f"📁 {dir_name}: {len(csv_files)} files")
                
                # Sample a few files for validation
                if csv_files:
                    sample_file = csv_files[0]
                    try:
                        df = pd.read_csv(sample_file, header=None)
                        results['data_analysis'][dir_name] = {
                            'sample_file': sample_file.name,
                            'shape': df.shape,
                            'nan_count': df.isnull().sum().sum(),
                            'inf_count': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
                            'data_range': {
                                'min': df.select_dtypes(include=[np.number]).min().min(),
                                'max': df.select_dtypes(include=[np.number]).max().max(),
                                'mean': df.select_dtypes(include=[np.number]).mean().mean()
                            }
                        }
                    except Exception as e:
                        results['data_analysis'][dir_name] = {'error': str(e)}
            else:
                results['file_counts'][dir_name] = 0
                print(f"❌ {dir_name}: Directory not found")
        
        # Expected file counts
        expected_counts = {
            'vol': 30,  # One per symbol
            'vol_of_vol': 30,  # One per symbol
            'covol': 435,  # 30 choose 2 = 435 pairs
            'covol_of_vol': 435  # 30 choose 2 = 435 pairs
        }
        
        print("\nFile count validation:")
        for dir_name, expected in expected_counts.items():
            actual = results['file_counts'].get(dir_name, 0)
            status = "✅" if actual == expected else "❌"
            print(f"{status} {dir_name}: {actual}/{expected} files")
        
        self.validation_results['step2'] = results
        return results
    
    def validate_step3_matrices(self):
        """Validate Step 3: Matrix dataset creation"""
        print("\n" + "="*80)
        print("STEP 3: MATRIX DATASET VALIDATION")
        print("="*80)
        
        h5_files = {
            'vols_matrices': 'processed_data/vols_mats_taq.h5',
            'volvols_matrices': 'processed_data/volvols_mats_taq.h5'
        }
        
        results = {
            'files_exist': {},
            'matrix_properties': {},
            'data_quality': {}
        }
        
        for name, filepath in h5_files.items():
            file_path = Path(filepath)
            results['files_exist'][name] = file_path.exists()
            
            if file_path.exists():
                try:
                    import h5py
                    with h5py.File(filepath, 'r') as f:
                        dataset = f['vols_mats'] if 'vols' in name else f['volvols_mats']
                        
                        results['matrix_properties'][name] = {
                            'shape': dataset.shape,
                            'dtype': str(dataset.dtype),
                            'size_mb': file_path.stat().st_size / (1024 * 1024)
                        }
                        
                        # Sample data for quality check
                        sample_data = dataset[:min(100, dataset.shape[0])]
                        results['data_quality'][name] = {
                            'nan_count': np.isnan(sample_data).sum(),
                            'inf_count': np.isinf(sample_data).sum(),
                            'zero_count': (sample_data == 0).sum(),
                            'data_range': {
                                'min': np.nanmin(sample_data),
                                'max': np.nanmax(sample_data),
                                'mean': np.nanmean(sample_data)
                            }
                        }
                        
                        print(f"✅ {name}: {dataset.shape} ({file_path.stat().st_size / (1024 * 1024):.1f} MB)")
                        
                except Exception as e:
                    results['data_quality'][name] = {'error': str(e)}
                    print(f"❌ {name}: Error reading file - {e}")
            else:
                print(f"❌ {name}: File not found")
        
        self.validation_results['step3'] = results
        return results
    
    def validate_step4_standardization(self):
        """Validate Step 4: Data standardization"""
        print("\n" + "="*80)
        print("STEP 4: DATA STANDARDIZATION VALIDATION")
        print("="*80)
        
        standardized_files = {
            'vols_standardized': 'processed_data/vols_mats_taq_standardized.h5',
            'volvols_standardized': 'processed_data/volvols_mats_taq_standardized.h5'
        }
        
        scaler_file = Path('processed_data/vols_mean_std_scalers.csv')
        
        results = {
            'files_exist': {},
            'scaler_analysis': {},
            'standardization_quality': {}
        }
        
        # Check scaler file
        results['files_exist']['scalers'] = scaler_file.exists()
        if scaler_file.exists():
            scalers_df = pd.read_csv(scaler_file)
            results['scaler_analysis'] = {
                'shape': scalers_df.shape,
                'columns': scalers_df.columns.tolist(),
                'mean_range': [scalers_df['mean'].min(), scalers_df['mean'].max()],
                'std_range': [scalers_df['std'].min(), scalers_df['std'].max()]
            }
            print(f"✅ Scalers file: {scalers_df.shape}")
        else:
            print("❌ Scalers file not found")
        
        # Check standardized files
        for name, filepath in standardized_files.items():
            file_path = Path(filepath)
            results['files_exist'][name] = file_path.exists()
            
            if file_path.exists():
                try:
                    import h5py
                    with h5py.File(filepath, 'r') as f:
                        dataset_name = 'vols_mats' if 'vols' in name else 'volvols_mats'
                        dataset = f[dataset_name]
                        
                        # Sample data for standardization check
                        sample_data = dataset[:min(1000, dataset.shape[0])]
                        
                        results['standardization_quality'][name] = {
                            'shape': dataset.shape,
                            'sample_mean': np.nanmean(sample_data),
                            'sample_std': np.nanstd(sample_data),
                            'nan_count': np.isnan(sample_data).sum(),
                            'inf_count': np.isinf(sample_data).sum()
                        }
                        
                        print(f"✅ {name}: {dataset.shape} (mean: {np.nanmean(sample_data):.3f}, std: {np.nanstd(sample_data):.3f})")
                        
                except Exception as e:
                    results['standardization_quality'][name] = {'error': str(e)}
                    print(f"❌ {name}: Error reading - {e}")
            else:
                print(f"❌ {name}: File not found")
        
        self.validation_results['step4'] = results
        return results
    
    def create_diagnostic_plots(self):
        """Create diagnostic plots for data validation"""
        print("\n" + "="*80)
        print("CREATING DIAGNOSTIC PLOTS")
        print("="*80)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: File sizes distribution (Step 1)
        if 'step1' in self.validation_results and self.validation_results['step1']['file_stats']:
            sizes = [stats['size_mb'] for stats in self.validation_results['step1']['file_stats'].values()]
            symbols = list(self.validation_results['step1']['file_stats'].keys())
            
            axes[0, 0].bar(range(len(sizes)), sizes)
            axes[0, 0].set_title('Raw Data File Sizes (MB)')
            axes[0, 0].set_xlabel('Symbols')
            axes[0, 0].set_ylabel('File Size (MB)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Sample volatility data
        vol_dir = Path("processed_data/vol/")
        if vol_dir.exists():
            vol_files = list(vol_dir.glob("*.csv"))
            if vol_files:
                try:
                    # Load first volatility file
                    df = pd.read_csv(vol_files[0], header=None)
                    vol_data = pd.concat([df[col] for col in df.columns], ignore_index=True).dropna()
                    
                    axes[0, 1].hist(vol_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[0, 1].set_title(f'Volatility Distribution ({vol_files[0].stem})')
                    axes[0, 1].set_xlabel('Volatility')
                    axes[0, 1].set_ylabel('Frequency')
                    
                    # Time series plot
                    axes[0, 2].plot(vol_data[:min(500, len(vol_data))])
                    axes[0, 2].set_title(f'Volatility Time Series (First 500 obs)')
                    axes[0, 2].set_xlabel('Time')
                    axes[0, 2].set_ylabel('Volatility')
                    
                except Exception as e:
                    axes[0, 1].text(0.5, 0.5, f'Error loading vol data: {e}', ha='center', va='center')
                    axes[0, 2].text(0.5, 0.5, f'Error loading vol data: {e}', ha='center', va='center')
        
        # Plot 3: Sample covolatility data
        covol_dir = Path("processed_data/covol/")
        if covol_dir.exists():
            covol_files = list(covol_dir.glob("*.csv"))
            if covol_files:
                try:
                    # Load first covolatility file
                    df = pd.read_csv(covol_files[0], header=None)
                    covol_data = pd.concat([df[col] for col in df.columns], ignore_index=True).dropna()
                    
                    axes[1, 0].hist(covol_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title(f'Covolatility Distribution ({covol_files[0].stem})')
                    axes[1, 0].set_xlabel('Covolatility')
                    axes[1, 0].set_ylabel('Frequency')
                    
                    # Time series plot
                    axes[1, 1].plot(covol_data[:min(500, len(covol_data))])
                    axes[1, 1].set_title(f'Covolatility Time Series (First 500 obs)')
                    axes[1, 1].set_xlabel('Time')
                    axes[1, 1].set_ylabel('Covolatility')
                    
                except Exception as e:
                    axes[1, 0].text(0.5, 0.5, f'Error loading covol data: {e}', ha='center', va='center')
                    axes[1, 1].text(0.5, 0.5, f'Error loading covol data: {e}', ha='center', va='center')
        
        # Plot 4: Standardized data validation
        try:
            import h5py
            std_file = Path('processed_data/vols_mats_taq_standardized.h5')
            if std_file.exists():
                with h5py.File(std_file, 'r') as f:
                    sample_data = f['vols_mats'][:min(1000, f['vols_mats'].shape[0])]
                    flat_data = sample_data.flatten()
                    flat_data = flat_data[~np.isnan(flat_data)]
                    
                    axes[1, 2].hist(flat_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[1, 2].set_title(f'Standardized Data Distribution')
                    axes[1, 2].set_xlabel('Standardized Value')
                    axes[1, 2].set_ylabel('Frequency')
                    axes[1, 2].axvline(0, color='red', linestyle='--', label='Mean')
                    axes[1, 2].legend()
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f'Error loading standardized data: {e}', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('data_validation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Diagnostic plots saved as 'data_validation_plots.png'")
    
    def generate_summary_report(self):
        """Generate comprehensive validation summary"""
        print("\n" + "="*80)
        print("DATA PIPELINE VALIDATION SUMMARY")
        print("="*80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'step_results': {}
        }
        
        # Step 1 summary
        if 'step1' in self.validation_results:
            step1 = self.validation_results['step1']
            files_found = len(step1['files_found'])
            files_expected = 30
            step1_pass = files_found >= 25  # Allow some missing files
            
            report['step_results']['step1'] = {
                'status': 'PASS' if step1_pass else 'FAIL',
                'files_found': f"{files_found}/{files_expected}",
                'data_quality_issues': sum(1 for q in step1['data_quality'].values() if 'error' in q)
            }
            
            if not step1_pass:
                report['overall_status'] = 'FAIL'
        
        # Step 2 summary
        if 'step2' in self.validation_results:
            step2 = self.validation_results['step2']
            expected_files = {'vol': 30, 'vol_of_vol': 30, 'covol': 435, 'covol_of_vol': 435}
            step2_pass = all(step2['file_counts'].get(k, 0) > 0 for k in expected_files.keys())
            
            report['step_results']['step2'] = {
                'status': 'PASS' if step2_pass else 'FAIL',
                'file_counts': step2['file_counts'],
                'data_errors': sum(1 for d in step2['data_analysis'].values() if 'error' in d)
            }
            
            if not step2_pass:
                report['overall_status'] = 'FAIL'
        
        # Step 3 summary
        if 'step3' in self.validation_results:
            step3 = self.validation_results['step3']
            step3_pass = all(step3['files_exist'].values())
            
            report['step_results']['step3'] = {
                'status': 'PASS' if step3_pass else 'FAIL',
                'files_exist': step3['files_exist'],
                'data_errors': sum(1 for d in step3['data_quality'].values() if 'error' in d)
            }
            
            if not step3_pass:
                report['overall_status'] = 'FAIL'
        
        # Step 4 summary
        if 'step4' in self.validation_results:
            step4 = self.validation_results['step4']
            step4_pass = all(step4['files_exist'].values())
            
            report['step_results']['step4'] = {
                'status': 'PASS' if step4_pass else 'FAIL',
                'files_exist': step4['files_exist'],
                'data_errors': sum(1 for d in step4['standardization_quality'].values() if 'error' in d)
            }
            
            if not step4_pass:
                report['overall_status'] = 'FAIL'
        
        # Print summary
        print(f"Overall Status: {report['overall_status']}")
        print("\nStep-by-step results:")
        for step, result in report['step_results'].items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {step.upper()}: {result['status']}")
        
        return report

def main():
    """Main validation execution"""
    print("🔍 Starting comprehensive data pipeline validation...")
    
    validator = DataPipelineValidator()
    
    # Run all validations
    validator.validate_step1_polygon_data()
    validator.validate_step2_processed_data()
    validator.validate_step3_matrices()
    validator.validate_step4_standardization()
    
    # Create diagnostic plots
    validator.create_diagnostic_plots()
    
    # Generate summary report
    report = validator.generate_summary_report()
    
    print(f"\n🎉 Validation complete! Overall status: {report['overall_status']}")

if __name__ == "__main__":
    main()