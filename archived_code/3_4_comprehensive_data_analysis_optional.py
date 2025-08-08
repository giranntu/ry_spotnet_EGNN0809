#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis and Validation
Final validation of the complete SpotV2Net data pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveDataAnalyzer:
    def __init__(self):
        self.results = {}
        
    def analyze_nan_values(self):
        """Comprehensive NaN analysis across all data"""
        print("="*80)
        print("NaN/ZERO VALUE ANALYSIS")
        print("="*80)
        
        results = {
            'processed_files': {},
            'h5_datasets': {},
            'summary': {}
        }
        
        # Check processed CSV files
        dirs_to_check = ['vol', 'vol_of_vol', 'covol', 'covol_of_vol']
        
        for dir_name in dirs_to_check:
            dir_path = Path(f"processed_data/{dir_name}")
            if dir_path.exists():
                csv_files = list(dir_path.glob("*.csv"))
                nan_counts = []
                zero_counts = []
                inf_counts = []
                
                for csv_file in csv_files[:5]:  # Sample first 5 files
                    try:
                        df = pd.read_csv(csv_file, header=None)
                        
                        # Convert to numeric, handling any string artifacts
                        numeric_df = df.apply(pd.to_numeric, errors='coerce')
                        
                        nan_counts.append(numeric_df.isnull().sum().sum())
                        zero_counts.append((numeric_df == 0).sum().sum())
                        inf_counts.append(np.isinf(numeric_df.select_dtypes(include=[np.number])).sum().sum())
                        
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                
                results['processed_files'][dir_name] = {
                    'total_files': len(csv_files),
                    'nan_counts': nan_counts,
                    'zero_counts': zero_counts,
                    'inf_counts': inf_counts,
                    'avg_nan': np.mean(nan_counts) if nan_counts else 0,
                    'avg_zero': np.mean(zero_counts) if zero_counts else 0,
                    'avg_inf': np.mean(inf_counts) if inf_counts else 0
                }
                
                print(f"📁 {dir_name}: {len(csv_files)} files")
                print(f"   Average NaN per file: {np.mean(nan_counts):.1f}")
                print(f"   Average zeros per file: {np.mean(zero_counts):.1f}")
                print(f"   Average inf per file: {np.mean(inf_counts):.1f}")
        
        # Check H5 datasets
        h5_files = {
            'vols_raw': 'processed_data/vols_mats_taq.h5',
            'vols_standardized': 'processed_data/vols_mats_taq_standardized.h5',
            'volvols_raw': 'processed_data/volvols_mats_taq.h5',
            'volvols_standardized': 'processed_data/volvols_mats_taq_standardized.h5'
        }
        
        for name, filepath in h5_files.items():
            file_path = Path(filepath)
            if file_path.exists():
                try:
                    with h5py.File(filepath, 'r') as f:
                        dataset_key = 'vols_mats' if 'vols' in name else 'volvols_mats'
                        dataset = f[dataset_key]
                        
                        # Sample data for analysis
                        sample_size = min(1000, dataset.shape[0])
                        sample_data = dataset[:sample_size]
                        
                        nan_count = np.isnan(sample_data).sum()
                        inf_count = np.isinf(sample_data).sum()
                        zero_count = (sample_data == 0).sum()
                        
                        results['h5_datasets'][name] = {
                            'shape': dataset.shape,
                            'sample_size': sample_size,
                            'nan_count': nan_count,
                            'inf_count': inf_count,
                            'zero_count': zero_count,
                            'nan_percentage': (nan_count / sample_data.size) * 100,
                            'inf_percentage': (inf_count / sample_data.size) * 100,
                            'zero_percentage': (zero_count / sample_data.size) * 100
                        }
                        
                        print(f"💾 {name}: Shape {dataset.shape}")
                        print(f"   NaN: {nan_count} ({(nan_count / sample_data.size) * 100:.2f}%)")
                        print(f"   Inf: {inf_count} ({(inf_count / sample_data.size) * 100:.2f}%)")
                        print(f"   Zero: {zero_count} ({(zero_count / sample_data.size) * 100:.2f}%)")
                        
                except Exception as e:
                    print(f"❌ Error reading {name}: {e}")
        
        self.results['nan_analysis'] = results
        return results
    
    def analyze_data_distributions(self):
        """Analyze data distributions and create plots"""
        print("\n" + "="*80)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('SpotV2Net Data Pipeline - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        results = {}
        
        # 1. Sample volatility data from CSV
        vol_dir = Path("processed_data/vol/")
        if vol_dir.exists():
            vol_files = list(vol_dir.glob("*.csv"))
            if vol_files:
                try:
                    # Load and combine first few volatility files
                    all_vol_data = []
                    for vol_file in vol_files[:5]:
                        df = pd.read_csv(vol_file, header=None)
                        vol_data = pd.concat([df[col] for col in df.columns], ignore_index=True)
                        vol_data = vol_data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                        all_vol_data.extend(vol_data.tolist())
                    
                    all_vol_data = np.array(all_vol_data)
                    
                    # Histogram
                    axes[0, 0].hist(all_vol_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('Volatility Distribution (CSV)')
                    axes[0, 0].set_xlabel('Volatility')
                    axes[0, 0].set_ylabel('Frequency')
                    
                    # Time series (sample)
                    sample_size = min(1000, len(all_vol_data))
                    axes[0, 1].plot(all_vol_data[:sample_size])
                    axes[0, 1].set_title('Volatility Time Series (Sample)')
                    axes[0, 1].set_xlabel('Time')
                    axes[0, 1].set_ylabel('Volatility')
                    
                    results['volatility_csv'] = {
                        'mean': np.mean(all_vol_data),
                        'std': np.std(all_vol_data),
                        'min': np.min(all_vol_data),
                        'max': np.max(all_vol_data),
                        'median': np.median(all_vol_data),
                        'count': len(all_vol_data)
                    }
                    
                except Exception as e:
                    axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 2. Sample covolatility data
        covol_dir = Path("processed_data/covol/")
        if covol_dir.exists():
            covol_files = list(covol_dir.glob("*.csv"))
            if covol_files:
                try:
                    # Load and combine first few covolatility files
                    all_covol_data = []
                    for covol_file in covol_files[:5]:
                        df = pd.read_csv(covol_file, header=None)
                        covol_data = pd.concat([df[col] for col in df.columns], ignore_index=True)
                        covol_data = covol_data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                        all_covol_data.extend(covol_data.tolist())
                    
                    all_covol_data = np.array(all_covol_data)
                    
                    # Histogram
                    axes[0, 2].hist(all_covol_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[0, 2].set_title('Covolatility Distribution (CSV)')
                    axes[0, 2].set_xlabel('Covolatility')
                    axes[0, 2].set_ylabel('Frequency')
                    
                    # Box plot
                    axes[0, 3].boxplot(all_covol_data)
                    axes[0, 3].set_title('Covolatility Box Plot')
                    axes[0, 3].set_ylabel('Covolatility')
                    
                    results['covolatility_csv'] = {
                        'mean': np.mean(all_covol_data),
                        'std': np.std(all_covol_data),
                        'min': np.min(all_covol_data),
                        'max': np.max(all_covol_data),
                        'median': np.median(all_covol_data),
                        'count': len(all_covol_data)
                    }
                    
                except Exception as e:
                    axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 3].transAxes)
        
        # 3. H5 raw data analysis
        try:
            with h5py.File('processed_data/vols_mats_taq.h5', 'r') as f:
                vols_data = f['vols_mats'][:1000]  # Sample first 1000 matrices
                flat_vols = vols_data.flatten()
                flat_vols = flat_vols[~np.isnan(flat_vols)]
                flat_vols = flat_vols[~np.isinf(flat_vols)]
                
                axes[1, 0].hist(flat_vols, bins=50, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Raw H5 Volatility Distribution')
                axes[1, 0].set_xlabel('Volatility')
                axes[1, 0].set_ylabel('Frequency')
                
                # Correlation matrix sample
                sample_matrix = vols_data[100]  # Sample one matrix
                if sample_matrix.shape[0] > 1:
                    corr_matrix = np.corrcoef(sample_matrix)
                    im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    axes[1, 1].set_title('Sample Correlation Matrix')
                    plt.colorbar(im, ax=axes[1, 1])
                
                results['h5_raw_vols'] = {
                    'shape': vols_data.shape,
                    'mean': np.nanmean(flat_vols),
                    'std': np.nanstd(flat_vols),
                    'min': np.nanmin(flat_vols),
                    'max': np.nanmax(flat_vols)
                }
                
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # 4. H5 standardized data analysis
        try:
            with h5py.File('processed_data/vols_mats_taq_standardized.h5', 'r') as f:
                std_vols_data = f['vols_mats'][:1000]  # Sample first 1000 matrices
                flat_std_vols = std_vols_data.flatten()
                flat_std_vols = flat_std_vols[~np.isnan(flat_std_vols)]
                flat_std_vols = flat_std_vols[~np.isinf(flat_std_vols)]
                
                axes[1, 2].hist(flat_std_vols, bins=50, alpha=0.7, edgecolor='black')
                axes[1, 2].set_title('Standardized H5 Volatility Distribution')
                axes[1, 2].set_xlabel('Standardized Volatility')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].axvline(0, color='red', linestyle='--', label='Mean=0')
                axes[1, 2].legend()
                
                # Q-Q plot for normality check
                from scipy import stats
                stats.probplot(flat_std_vols[:1000], dist="norm", plot=axes[1, 3])
                axes[1, 3].set_title('Q-Q Plot (Normality Check)')
                
                results['h5_standardized_vols'] = {
                    'shape': std_vols_data.shape,
                    'mean': np.nanmean(flat_std_vols),
                    'std': np.nanstd(flat_std_vols),
                    'min': np.nanmin(flat_std_vols),
                    'max': np.nanmax(flat_std_vols)
                }
                
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 3].transAxes)
        
        # 5. Temporal evolution analysis
        try:
            with h5py.File('processed_data/vols_mats_taq.h5', 'r') as f:
                vols_data = f['vols_mats']
                
                # Calculate mean volatility over time
                temporal_means = []
                temporal_stds = []
                
                for i in range(0, min(vols_data.shape[0], 2000), 10):
                    matrix = vols_data[i]
                    diagonal = np.diag(matrix)  # Individual volatilities
                    diagonal = diagonal[~np.isnan(diagonal)]
                    if len(diagonal) > 0:
                        temporal_means.append(np.mean(diagonal))
                        temporal_stds.append(np.std(diagonal))
                
                axes[2, 0].plot(temporal_means, label='Mean Volatility')
                axes[2, 0].fill_between(range(len(temporal_means)), 
                                      np.array(temporal_means) - np.array(temporal_stds),
                                      np.array(temporal_means) + np.array(temporal_stds),
                                      alpha=0.3)
                axes[2, 0].set_title('Temporal Evolution of Volatility')
                axes[2, 0].set_xlabel('Time (10-day intervals)')
                axes[2, 0].set_ylabel('Volatility')
                axes[2, 0].legend()
                
                # Volatility distribution across assets
                sample_matrix = vols_data[1000]  # Middle of dataset
                diagonal_vols = np.diag(sample_matrix)
                diagonal_vols = diagonal_vols[~np.isnan(diagonal_vols)]
                
                axes[2, 1].bar(range(len(diagonal_vols)), diagonal_vols)
                axes[2, 1].set_title('Individual Asset Volatilities (Sample)')
                axes[2, 1].set_xlabel('Asset Index')
                axes[2, 1].set_ylabel('Volatility')
                
                results['temporal_analysis'] = {
                    'temporal_mean_avg': np.mean(temporal_means),
                    'temporal_std_avg': np.mean(temporal_stds),
                    'individual_vol_range': [np.min(diagonal_vols), np.max(diagonal_vols)]
                }
                
        except Exception as e:
            axes[2, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 1].transAxes)
        
        # 6. Data quality summary plot
        quality_metrics = []
        quality_labels = []
        
        if 'volatility_csv' in results:
            quality_metrics.append(results['volatility_csv']['mean'])
            quality_labels.append('Vol CSV Mean')
        
        if 'h5_raw_vols' in results:
            quality_metrics.append(results['h5_raw_vols']['mean'])
            quality_labels.append('Vol H5 Raw Mean')
        
        if 'h5_standardized_vols' in results:
            quality_metrics.append(abs(results['h5_standardized_vols']['mean']))
            quality_labels.append('Vol H5 Std |Mean|')
        
        if quality_metrics:
            axes[2, 2].bar(quality_labels, quality_metrics)
            axes[2, 2].set_title('Data Quality Metrics')
            axes[2, 2].set_ylabel('Value')
            axes[2, 2].tick_params(axis='x', rotation=45)
        
        # 7. File size analysis
        try:
            file_sizes = []
            file_names = []
            
            data_dirs = ['vol', 'vol_of_vol', 'covol', 'covol_of_vol']
            for dir_name in data_dirs:
                dir_path = Path(f"processed_data/{dir_name}")
                if dir_path.exists():
                    total_size = sum(f.stat().st_size for f in dir_path.glob("*.csv"))
                    file_sizes.append(total_size / (1024 * 1024))  # MB
                    file_names.append(dir_name)
            
            # Add H5 files
            h5_files = ['vols_mats_taq.h5', 'vols_mats_taq_standardized.h5', 
                       'volvols_mats_taq.h5', 'volvols_mats_taq_standardized.h5']
            for h5_file in h5_files:
                h5_path = Path(f"processed_data/{h5_file}")
                if h5_path.exists():
                    file_sizes.append(h5_path.stat().st_size / (1024 * 1024))
                    file_names.append(h5_file.replace('.h5', ''))
            
            axes[2, 3].pie(file_sizes, labels=file_names, autopct='%1.1f%%')
            axes[2, 3].set_title('Storage Usage by Component')
            
        except Exception as e:
            axes[2, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 3].transAxes)
        
        plt.tight_layout()
        plt.savefig('comprehensive_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['distributions'] = results
        return results
    
    def validate_data_format_consistency(self):
        """Validate data format consistency across pipeline"""
        print("\n" + "="*80)
        print("DATA FORMAT CONSISTENCY VALIDATION")
        print("="*80)
        
        results = {
            'csv_format_check': {},
            'h5_format_check': {},
            'dimension_consistency': {},
            'train_val_test_splits': {}
        }
        
        # Check CSV format consistency
        dirs_to_check = ['vol', 'vol_of_vol', 'covol', 'covol_of_vol']
        
        for dir_name in dirs_to_check:
            dir_path = Path(f"processed_data/{dir_name}")
            if dir_path.exists():
                csv_files = list(dir_path.glob("*.csv"))
                shapes = []
                dtypes = []
                
                for csv_file in csv_files[:10]:  # Sample first 10 files
                    try:
                        df = pd.read_csv(csv_file, header=None)
                        shapes.append(df.shape)
                        dtypes.append(str(df.dtypes.iloc[0]))
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                
                results['csv_format_check'][dir_name] = {
                    'total_files': len(csv_files),
                    'sampled_shapes': shapes,
                    'unique_shapes': list(set(shapes)),
                    'consistent_shape': len(set(shapes)) == 1,
                    'dtypes': dtypes
                }
                
                print(f"📁 {dir_name}: {len(csv_files)} files, Shape consistency: {len(set(shapes)) == 1}")
                if shapes:
                    print(f"   Shapes found: {list(set(shapes))}")
        
        # Check H5 format consistency
        h5_files = {
            'vols_raw': 'processed_data/vols_mats_taq.h5',
            'vols_standardized': 'processed_data/vols_mats_taq_standardized.h5',
            'volvols_raw': 'processed_data/volvols_mats_taq.h5',
            'volvols_standardized': 'processed_data/volvols_mats_taq_standardized.h5'
        }
        
        for name, filepath in h5_files.items():
            file_path = Path(filepath)
            if file_path.exists():
                try:
                    with h5py.File(filepath, 'r') as f:
                        dataset_key = 'vols_mats' if 'vols' in name else 'volvols_mats'
                        dataset = f[dataset_key]
                        
                        results['h5_format_check'][name] = {
                            'shape': dataset.shape,
                            'dtype': str(dataset.dtype),
                            'chunking': dataset.chunks,
                            'compression': dataset.compression
                        }
                        
                        print(f"💾 {name}: Shape {dataset.shape}, dtype {dataset.dtype}")
                        
                except Exception as e:
                    print(f"❌ Error reading {name}: {e}")
        
        # Check train/val/test splits
        try:
            with h5py.File('processed_data/vols_mats_taq_standardized.h5', 'r') as f:
                total_matrices = f['vols_mats'].shape[0]
                
                # Expected splits based on standardization script
                train_end = 1008  # 4 years * 252 trading days
                val_end = 1260    # train_end + 252 
                
                train_size = train_end
                val_size = val_end - train_end
                test_size = total_matrices - val_end
                
                results['train_val_test_splits'] = {
                    'total_matrices': total_matrices,
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size,
                    'train_percentage': (train_size / total_matrices) * 100,
                    'val_percentage': (val_size / total_matrices) * 100,
                    'test_percentage': (test_size / total_matrices) * 100
                }
                
                print(f"📊 Dataset splits:")
                print(f"   Training: {train_size} matrices ({(train_size / total_matrices) * 100:.1f}%)")
                print(f"   Validation: {val_size} matrices ({(val_size / total_matrices) * 100:.1f}%)")
                print(f"   Test: {test_size} matrices ({(test_size / total_matrices) * 100:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error checking splits: {e}")
        
        self.results['format_validation'] = results
        return results
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("FINAL DATA QUALITY REPORT")
        print("="*80)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_status': 'PASS',
            'critical_issues': [],
            'warnings': [],
            'summary_statistics': {}
        }
        
        # Check for critical issues
        
        # 1. Check for excessive NaN values
        if 'nan_analysis' in self.results:
            nan_data = self.results['nan_analysis']
            
            for dataset_name, dataset_info in nan_data.get('h5_datasets', {}).items():
                nan_pct = dataset_info.get('nan_percentage', 0)
                if nan_pct > 5:  # More than 5% NaN is critical
                    report['critical_issues'].append(f"{dataset_name}: {nan_pct:.1f}% NaN values")
                    report['overall_status'] = 'FAIL'
                elif nan_pct > 1:  # 1-5% NaN is a warning
                    report['warnings'].append(f"{dataset_name}: {nan_pct:.1f}% NaN values")
        
        # 2. Check standardization quality
        if 'distributions' in self.results:
            dist_data = self.results['distributions']
            
            if 'h5_standardized_vols' in dist_data:
                std_mean = abs(dist_data['h5_standardized_vols']['mean'])
                std_std = dist_data['h5_standardized_vols']['std']
                
                if std_mean > 0.1:  # Mean should be close to 0
                    report['warnings'].append(f"Standardized data mean: {std_mean:.3f} (should be ~0)")
                
                if abs(std_std - 1.0) > 0.2:  # Std should be close to 1
                    report['warnings'].append(f"Standardized data std: {std_std:.3f} (should be ~1)")
        
        # 3. Check data format consistency
        if 'format_validation' in self.results:
            format_data = self.results['format_validation']
            
            for dir_name, csv_info in format_data.get('csv_format_check', {}).items():
                if not csv_info.get('consistent_shape', False):
                    report['warnings'].append(f"{dir_name}: Inconsistent CSV shapes")
        
        # 4. Check file completeness
        expected_files = {
            'vol': 29,  # Missing BA
            'vol_of_vol': 29,
            'covol': 406,  # 29 choose 2
            'covol_of_vol': 406
        }
        
        if 'format_validation' in self.results:
            for dir_name, expected_count in expected_files.items():
                actual_count = self.results['format_validation']['csv_format_check'].get(dir_name, {}).get('total_files', 0)
                if actual_count < expected_count * 0.9:  # Allow 10% missing
                    report['critical_issues'].append(f"{dir_name}: Only {actual_count}/{expected_count} files")
                    report['overall_status'] = 'FAIL'
        
        # Generate summary statistics
        if 'distributions' in self.results:
            dist_data = self.results['distributions']
            
            report['summary_statistics'] = {
                'volatility_range': f"{dist_data.get('volatility_csv', {}).get('min', 0):.3f} - {dist_data.get('volatility_csv', {}).get('max', 0):.3f}",
                'covolatility_range': f"{dist_data.get('covolatility_csv', {}).get('min', 0):.3f} - {dist_data.get('covolatility_csv', {}).get('max', 0):.3f}",
                'standardized_mean': f"{dist_data.get('h5_standardized_vols', {}).get('mean', 0):.3f}",
                'standardized_std': f"{dist_data.get('h5_standardized_vols', {}).get('std', 0):.3f}"
            }
        
        # Print report
        status_icon = "✅" if report['overall_status'] == 'PASS' else "❌"
        print(f"{status_icon} Overall Status: {report['overall_status']}")
        
        if report['critical_issues']:
            print(f"\n❌ Critical Issues ({len(report['critical_issues'])}):")
            for issue in report['critical_issues']:
                print(f"   • {issue}")
        
        if report['warnings']:
            print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"   • {warning}")
        
        if not report['critical_issues'] and not report['warnings']:
            print("✅ No issues found - data pipeline is ready for training!")
        
        print(f"\n📊 Summary Statistics:")
        for key, value in report['summary_statistics'].items():
            print(f"   {key}: {value}")
        
        return report

def main():
    """Main analysis execution"""
    print("🔬 Starting comprehensive data quality analysis...")
    
    analyzer = ComprehensiveDataAnalyzer()
    
    # Run all analyses
    analyzer.analyze_nan_values()
    analyzer.analyze_data_distributions()
    analyzer.validate_data_format_consistency()
    
    # Generate final report
    final_report = analyzer.generate_final_report()
    
    print(f"\n🎉 Analysis complete! Data pipeline status: {final_report['overall_status']}")
    print("📈 Diagnostic plots saved as 'comprehensive_data_analysis.png'")

if __name__ == "__main__":
    main()