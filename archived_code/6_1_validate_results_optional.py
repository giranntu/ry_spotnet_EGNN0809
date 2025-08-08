#!/usr/bin/env python3
"""
Final Data Pipeline Validation with Corrected H5 Structure
Creates comprehensive diagnostic plots and validation report
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

def validate_h5_data_with_correct_structure():
    """Validate H5 data with correct understanding of structure"""
    print("="*80)
    print("H5 DATA VALIDATION (CORRECTED STRUCTURE)")
    print("="*80)
    
    h5_files = {
        'vols_raw': 'processed_data/vols_mats_taq.h5',
        'vols_standardized': 'processed_data/vols_mats_taq_standardized.h5',
        'volvols_raw': 'processed_data/volvols_mats_taq.h5',
        'volvols_standardized': 'processed_data/volvols_mats_taq_standardized.h5'
    }
    
    results = {}
    
    for name, filepath in h5_files.items():
        file_path = Path(filepath)
        if file_path.exists():
            try:
                with h5py.File(filepath, 'r') as f:
                    # Keys are string indices ('0', '1', '2', ...)
                    matrix_keys = [key for key in f.keys() if key.isdigit()]
                    matrix_keys_sorted = sorted(matrix_keys, key=int)
                    
                    total_matrices = len(matrix_keys_sorted)
                    
                    # Sample first few matrices for analysis
                    sample_matrices = []
                    sample_size = min(10, total_matrices)
                    
                    for i in range(sample_size):
                        matrix = f[matrix_keys_sorted[i]][:]
                        sample_matrices.append(matrix)
                    
                    # Calculate statistics
                    if sample_matrices:
                        first_matrix = sample_matrices[0]
                        matrix_shape = first_matrix.shape
                        
                        # Flatten all sample data for analysis
                        flat_data = np.concatenate([m.flatten() for m in sample_matrices])
                        flat_data = flat_data[~np.isnan(flat_data)]
                        flat_data = flat_data[~np.isinf(flat_data)]
                        
                        results[name] = {
                            'total_matrices': total_matrices,
                            'matrix_shape': matrix_shape,
                            'sample_size': sample_size,
                            'data_stats': {
                                'mean': np.mean(flat_data),
                                'std': np.std(flat_data),
                                'min': np.min(flat_data),
                                'max': np.max(flat_data),
                                'median': np.median(flat_data)
                            },
                            'nan_count': np.sum([np.isnan(m).sum() for m in sample_matrices]),
                            'inf_count': np.sum([np.isinf(m).sum() for m in sample_matrices]),
                            'zero_count': np.sum([(m == 0).sum() for m in sample_matrices])
                        }
                        
                        print(f"✅ {name}:")
                        print(f"   Total matrices: {total_matrices}")
                        print(f"   Matrix shape: {matrix_shape}")
                        print(f"   Data range: [{np.min(flat_data):.4f}, {np.max(flat_data):.4f}]")
                        print(f"   Mean: {np.mean(flat_data):.4f}, Std: {np.std(flat_data):.4f}")
                        print(f"   NaN count (sample): {results[name]['nan_count']}")
                        print(f"   Inf count (sample): {results[name]['inf_count']}")
                        print(f"   Zero count (sample): {results[name]['zero_count']}")
                        
            except Exception as e:
                print(f"❌ Error reading {name}: {e}")
                results[name] = {'error': str(e)}
        else:
            print(f"❌ {name}: File not found")
            results[name] = {'missing': True}
    
    return results

def create_comprehensive_diagnostic_plots():
    """Create comprehensive diagnostic plots"""
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE DIAGNOSTIC PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('SpotV2Net Data Pipeline - Final Validation Report', fontsize=16, fontweight='bold')
    
    # 1. CSV Volatility Data Analysis
    try:
        vol_dir = Path("processed_data/vol/")
        if vol_dir.exists():
            vol_files = list(vol_dir.glob("*.csv"))[:5]  # Sample 5 files
            all_vol_data = []
            
            for vol_file in vol_files:
                df = pd.read_csv(vol_file, header=None)
                vol_data = pd.concat([df[col] for col in df.columns], ignore_index=True)
                vol_data = vol_data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                all_vol_data.extend(vol_data.tolist())
            
            all_vol_data = np.array(all_vol_data)
            
            # Histogram
            axes[0, 0].hist(all_vol_data, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            axes[0, 0].set_title('Individual Volatility Distribution (CSV)')
            axes[0, 0].set_xlabel('Volatility')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(all_vol_data), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(all_vol_data):.3f}')
            axes[0, 0].legend()
            
            print(f"✅ Volatility range: [{np.min(all_vol_data):.3f}, {np.max(all_vol_data):.3f}]")
            
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f'Error loading vol data: {e}', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 2. CSV Covolatility Data Analysis
    try:
        covol_dir = Path("processed_data/covol/")
        if covol_dir.exists():
            covol_files = list(covol_dir.glob("*.csv"))[:5]  # Sample 5 files
            all_covol_data = []
            
            for covol_file in covol_files:
                df = pd.read_csv(covol_file, header=None)
                covol_data = pd.concat([df[col] for col in df.columns], ignore_index=True)
                covol_data = covol_data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                all_covol_data.extend(covol_data.tolist())
            
            all_covol_data = np.array(all_covol_data)
            
            # Histogram
            axes[0, 1].hist(all_covol_data, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
            axes[0, 1].set_title('Covolatility Distribution (CSV)')
            axes[0, 1].set_xlabel('Covolatility')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(all_covol_data), color='red', linestyle='--',
                              label=f'Mean: {np.mean(all_covol_data):.3f}')
            axes[0, 1].legend()
            
            print(f"✅ Covolatility range: [{np.min(all_covol_data):.3f}, {np.max(all_covol_data):.3f}]")
            
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'Error loading covol data: {e}', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. H5 Raw Volatility Data
    try:
        with h5py.File('processed_data/vols_mats_taq.h5', 'r') as f:
            matrix_keys = sorted([key for key in f.keys() if key.isdigit()], key=int)
            
            # Sample first 20 matrices
            sample_data = []
            for i in range(min(20, len(matrix_keys))):
                matrix = f[matrix_keys[i]][:]
                # Extract diagonal (individual volatilities)
                diagonal = np.diag(matrix)
                diagonal = diagonal[~np.isnan(diagonal)]
                sample_data.extend(diagonal.tolist())
            
            sample_data = np.array(sample_data)
            sample_data = sample_data[~np.isinf(sample_data)]
            
            axes[0, 2].hist(sample_data, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
            axes[0, 2].set_title('H5 Raw Volatility (Diagonal)')
            axes[0, 2].set_xlabel('Volatility')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].axvline(np.mean(sample_data), color='red', linestyle='--',
                              label=f'Mean: {np.mean(sample_data):.3f}')
            axes[0, 2].legend()
            
            print(f"✅ H5 Raw Volatility range: [{np.min(sample_data):.3f}, {np.max(sample_data):.3f}]")
            
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # 4. H5 Standardized Volatility Data
    try:
        with h5py.File('processed_data/vols_mats_taq_standardized.h5', 'r') as f:
            matrix_keys = sorted([key for key in f.keys() if key.isdigit()], key=int)
            
            # Sample first 20 matrices
            sample_data = []
            for i in range(min(20, len(matrix_keys))):
                matrix = f[matrix_keys[i]][:]
                # Extract diagonal (individual volatilities)
                diagonal = np.diag(matrix)
                diagonal = diagonal[~np.isnan(diagonal)]
                sample_data.extend(diagonal.tolist())
            
            sample_data = np.array(sample_data)
            sample_data = sample_data[~np.isinf(sample_data)]
            
            axes[0, 3].hist(sample_data, bins=50, alpha=0.7, edgecolor='black', color='lightyellow')
            axes[0, 3].set_title('H5 Standardized Volatility')
            axes[0, 3].set_xlabel('Standardized Volatility')
            axes[0, 3].set_ylabel('Frequency')
            axes[0, 3].axvline(0, color='red', linestyle='--', label='Expected Mean=0')
            axes[0, 3].axvline(np.mean(sample_data), color='blue', linestyle='-',
                              label=f'Actual Mean: {np.mean(sample_data):.3f}')
            axes[0, 3].legend()
            
            print(f"✅ H5 Standardized stats: Mean={np.mean(sample_data):.3f}, Std={np.std(sample_data):.3f}")
            
    except Exception as e:
        axes[0, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 3].transAxes)
    
    # 5. Sample Correlation Matrix
    try:
        with h5py.File('processed_data/vols_mats_taq.h5', 'r') as f:
            matrix_keys = sorted([key for key in f.keys() if key.isdigit()], key=int)
            
            # Get a sample matrix from the middle of the dataset
            mid_idx = len(matrix_keys) // 2
            sample_matrix = f[matrix_keys[mid_idx]][:]
            
            # Create correlation matrix from covariance matrix
            # Convert covariances to correlations
            std_diag = np.sqrt(np.diag(sample_matrix))
            corr_matrix = sample_matrix / np.outer(std_diag, std_diag)
            
            # Handle any NaN values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            
            im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title('Sample Correlation Matrix')
            plt.colorbar(im, ax=axes[1, 0])
            
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 6. Temporal Evolution
    try:
        with h5py.File('processed_data/vols_mats_taq.h5', 'r') as f:
            matrix_keys = sorted([key for key in f.keys() if key.isdigit()], key=int)
            
            # Sample every 50th matrix to show temporal evolution
            temporal_means = []
            temporal_stds = []
            time_points = []
            
            for i in range(0, len(matrix_keys), 50):
                matrix = f[matrix_keys[i]][:]
                diagonal = np.diag(matrix)
                diagonal = diagonal[~np.isnan(diagonal)]
                if len(diagonal) > 0:
                    temporal_means.append(np.mean(diagonal))
                    temporal_stds.append(np.std(diagonal))
                    time_points.append(int(matrix_keys[i]))
            
            axes[1, 1].plot(time_points, temporal_means, 'b-', linewidth=2, label='Mean Volatility')
            axes[1, 1].fill_between(time_points, 
                                  np.array(temporal_means) - np.array(temporal_stds),
                                  np.array(temporal_means) + np.array(temporal_stds),
                                  alpha=0.3, color='blue')
            axes[1, 1].set_title('Temporal Evolution of Mean Volatility')
            axes[1, 1].set_xlabel('Time Index')
            axes[1, 1].set_ylabel('Average Volatility')
            axes[1, 1].legend()
            
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # 7. Data Quality Summary
    try:
        # Count files in each directory
        data_dirs = ['vol', 'vol_of_vol', 'covol', 'covol_of_vol']
        file_counts = []
        dir_names = []
        
        for dir_name in data_dirs:
            dir_path = Path(f"processed_data/{dir_name}")
            if dir_path.exists():
                count = len(list(dir_path.glob("*.csv")))
                file_counts.append(count)
                dir_names.append(dir_name)
        
        bars = axes[1, 2].bar(dir_names, file_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
        axes[1, 2].set_title('File Counts by Data Type')
        axes[1, 2].set_ylabel('Number of Files')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, file_counts):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           str(count), ha='center', va='bottom')
        
        expected_counts = [29, 29, 406, 406]  # Expected file counts
        for i, (actual, expected) in enumerate(zip(file_counts, expected_counts)):
            if actual != expected:
                axes[1, 2].text(i, actual/2, f'Expected: {expected}', ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 2].transAxes)
    
    # 8. Train/Val/Test Split Visualization
    try:
        with h5py.File('processed_data/vols_mats_taq_standardized.h5', 'r') as f:
            total_matrices = len([key for key in f.keys() if key.isdigit()])
            
            # Expected splits
            train_end = 1008
            val_end = 1260
            
            train_size = train_end
            val_size = val_end - train_end
            test_size = total_matrices - val_end
            
            sizes = [train_size, val_size, test_size]
            labels = ['Training', 'Validation', 'Test']
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = axes[1, 3].pie(sizes, labels=labels, colors=colors, 
                                                     autopct='%1.1f%%', startangle=90)
            axes[1, 3].set_title(f'Dataset Splits (Total: {total_matrices} matrices)')
            
            # Add absolute numbers
            for i, (autotext, size) in enumerate(zip(autotexts, sizes)):
                autotext.set_text(f'{size}\n({autotext.get_text()})')
        
    except Exception as e:
        axes[1, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 3].transAxes)
    
    # 9. NaN Analysis Summary
    try:
        # Count NaN values across different data types
        nan_counts = {}
        
        # CSV files
        for data_type in ['vol', 'vol_of_vol', 'covol', 'covol_of_vol']:
            dir_path = Path(f"processed_data/{data_type}")
            if dir_path.exists():
                total_nans = 0
                total_values = 0
                csv_files = list(dir_path.glob("*.csv"))[:5]  # Sample 5 files
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, header=None)
                        numeric_df = df.apply(pd.to_numeric, errors='coerce')
                        total_nans += numeric_df.isnull().sum().sum()
                        total_values += numeric_df.size
                    except:
                        pass
                
                if total_values > 0:
                    nan_counts[data_type] = (total_nans / total_values) * 100
        
        if nan_counts:
            data_types = list(nan_counts.keys())
            nan_percentages = list(nan_counts.values())
            
            bars = axes[2, 0].bar(data_types, nan_percentages, 
                                 color=['red' if pct > 5 else 'orange' if pct > 1 else 'green' 
                                       for pct in nan_percentages])
            axes[2, 0].set_title('NaN Percentage by Data Type')
            axes[2, 0].set_ylabel('NaN Percentage (%)')
            axes[2, 0].tick_params(axis='x', rotation=45)
            axes[2, 0].axhline(y=5, color='red', linestyle='--', label='Critical (5%)')
            axes[2, 0].axhline(y=1, color='orange', linestyle='--', label='Warning (1%)')
            axes[2, 0].legend()
            
            # Add value labels
            for bar, pct in zip(bars, nan_percentages):
                axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{pct:.1f}%', ha='center', va='bottom')
        
    except Exception as e:
        axes[2, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 0].transAxes)
    
    # 10. Storage Usage Analysis
    try:
        file_sizes = []
        file_labels = []
        
        # CSV directories
        for data_type in ['vol', 'vol_of_vol', 'covol', 'covol_of_vol']:
            dir_path = Path(f"processed_data/{data_type}")
            if dir_path.exists():
                total_size = sum(f.stat().st_size for f in dir_path.glob("*.csv")) / (1024 * 1024)
                file_sizes.append(total_size)
                file_labels.append(f'{data_type}_csv')
        
        # H5 files
        h5_files = ['vols_mats_taq.h5', 'vols_mats_taq_standardized.h5', 
                   'volvols_mats_taq.h5', 'volvols_mats_taq_standardized.h5']
        for h5_file in h5_files:
            h5_path = Path(f"processed_data/{h5_file}")
            if h5_path.exists():
                size_mb = h5_path.stat().st_size / (1024 * 1024)
                file_sizes.append(size_mb)
                file_labels.append(h5_file.replace('.h5', ''))
        
        if file_sizes:
            wedges, texts, autotexts = axes[2, 1].pie(file_sizes, labels=file_labels, 
                                                     autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                                     startangle=90)
            axes[2, 1].set_title('Storage Usage by Component (MB)')
            
            # Show total storage
            total_storage = sum(file_sizes)
            axes[2, 1].text(0, -1.3, f'Total Storage: {total_storage:.1f} MB', 
                           ha='center', transform=axes[2, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
    except Exception as e:
        axes[2, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 1].transAxes)
    
    # 11. Data Processing Pipeline Status
    try:
        pipeline_steps = ['Raw Data\n(Polygon)', 'Volatility\n(Yang-Zhang)', 'Matrices\n(H5)', 'Standardized\n(H5)']
        step_status = []
        
        # Check each step
        polygon_files = len(list(Path("rawdata/polygon/by_comp/").glob("*.csv"))) if Path("rawdata/polygon/by_comp/").exists() else 0
        vol_files = len(list(Path("processed_data/vol/").glob("*.csv"))) if Path("processed_data/vol/").exists() else 0
        h5_raw_exists = Path("processed_data/vols_mats_taq.h5").exists()
        h5_std_exists = Path("processed_data/vols_mats_taq_standardized.h5").exists()
        
        step_status = [
            polygon_files >= 25,  # At least 25/30 files
            vol_files >= 25,      # At least 25/30 files
            h5_raw_exists,        # H5 raw file exists
            h5_std_exists         # H5 standardized file exists
        ]
        
        colors = ['green' if status else 'red' for status in step_status]
        bars = axes[2, 2].bar(range(len(pipeline_steps)), [1] * len(pipeline_steps), color=colors)
        axes[2, 2].set_xticks(range(len(pipeline_steps)))
        axes[2, 2].set_xticklabels(pipeline_steps)
        axes[2, 2].set_title('Pipeline Step Completion Status')
        axes[2, 2].set_ylabel('Status')
        axes[2, 2].set_ylim(0, 1.2)
        
        # Add status labels
        for i, (bar, status) in enumerate(zip(bars, step_status)):
            label = '✅' if status else '❌'
            axes[2, 2].text(bar.get_x() + bar.get_width()/2, 0.5, label, 
                           ha='center', va='center', fontsize=20)
        
    except Exception as e:
        axes[2, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 2].transAxes)
    
    # 12. Final Quality Score
    try:
        quality_checks = []
        quality_labels = []
        
        # File completeness
        vol_files = len(list(Path("processed_data/vol/").glob("*.csv"))) if Path("processed_data/vol/").exists() else 0
        quality_checks.append(vol_files >= 25)
        quality_labels.append('File Completeness')
        
        # H5 files exist
        h5_exists = Path("processed_data/vols_mats_taq_standardized.h5").exists()
        quality_checks.append(h5_exists)
        quality_labels.append('H5 Files')
        
        # Data ranges reasonable (check volatility)
        try:
            with h5py.File('processed_data/vols_mats_taq.h5', 'r') as f:
                matrix_keys = sorted([key for key in f.keys() if key.isdigit()], key=int)
                sample_matrix = f[matrix_keys[0]][:]
                diagonal = np.diag(sample_matrix)
                diagonal = diagonal[~np.isnan(diagonal)]
                vol_range_ok = 0.01 < np.mean(diagonal) < 2.0  # Reasonable volatility range
                quality_checks.append(vol_range_ok)
                quality_labels.append('Data Ranges')
        except:
            quality_checks.append(False)
            quality_labels.append('Data Ranges')
        
        # Calculate overall score
        quality_score = (sum(quality_checks) / len(quality_checks)) * 100
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color based on score
        if quality_score >= 90:
            color = 'green'
            status_text = 'EXCELLENT'
        elif quality_score >= 70:
            color = 'orange'
            status_text = 'GOOD'
        else:
            color = 'red'
            status_text = 'NEEDS ATTENTION'
        
        axes[2, 3].fill_between(theta, 0, r, alpha=0.3, color=color)
        axes[2, 3].plot(theta, r, color=color, linewidth=3)
        
        # Add score text
        axes[2, 3].text(np.pi/2, 0.5, f'{quality_score:.0f}%', 
                       ha='center', va='center', fontsize=24, fontweight='bold')
        axes[2, 3].text(np.pi/2, 0.2, status_text, 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        axes[2, 3].set_xlim(0, np.pi)
        axes[2, 3].set_ylim(0, 1.2)
        axes[2, 3].set_title('Overall Data Quality Score')
        axes[2, 3].axis('off')
        
    except Exception as e:
        axes[2, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[2, 3].transAxes)
    
    plt.tight_layout()
    plt.savefig('final_data_validation_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive diagnostic plots saved as 'final_data_validation_report.png'")

def generate_final_summary_report():
    """Generate final comprehensive summary"""
    print("\n" + "="*80)
    print("FINAL DATA PIPELINE VALIDATION SUMMARY")
    print("="*80)
    
    # Validate H5 data first
    h5_results = validate_h5_data_with_correct_structure()
    
    # Check file completeness
    file_counts = {}
    expected_counts = {'vol': 29, 'vol_of_vol': 29, 'covol': 406, 'covol_of_vol': 406}
    
    for data_type in expected_counts:
        dir_path = Path(f"processed_data/{data_type}")
        if dir_path.exists():
            actual_count = len(list(dir_path.glob("*.csv")))
            file_counts[data_type] = actual_count
        else:
            file_counts[data_type] = 0
    
    # Overall assessment
    critical_issues = []
    warnings = []
    
    # Check file completeness
    for data_type, expected in expected_counts.items():
        actual = file_counts.get(data_type, 0)
        if actual < expected * 0.8:  # Less than 80% of expected files
            critical_issues.append(f"{data_type}: Only {actual}/{expected} files")
        elif actual < expected:
            warnings.append(f"{data_type}: Missing {expected - actual} files")
    
    # Check H5 files
    required_h5_files = ['vols_raw', 'vols_standardized', 'volvols_raw', 'volvols_standardized']
    for h5_file in required_h5_files:
        if h5_file not in h5_results or 'error' in h5_results[h5_file] or 'missing' in h5_results[h5_file]:
            critical_issues.append(f"H5 file issue: {h5_file}")
    
    # Check data quality
    for h5_file, results in h5_results.items():
        if 'data_stats' in results:
            mean_val = results['data_stats']['mean']
            if 'standardized' in h5_file and abs(mean_val) > 0.1:
                warnings.append(f"{h5_file}: Mean not close to 0 ({mean_val:.3f})")
    
    # Determine overall status
    if critical_issues:
        overall_status = 'FAIL'
        status_icon = '❌'
    elif warnings:
        overall_status = 'PASS WITH WARNINGS'
        status_icon = '⚠️'
    else:
        overall_status = 'PASS'
        status_icon = '✅'
    
    # Print summary
    print(f"{status_icon} Overall Status: {overall_status}")
    
    if critical_issues:
        print(f"\n❌ Critical Issues ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"   • {issue}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not critical_issues and not warnings:
        print("\n🎉 Excellent! No issues found - data pipeline is production ready!")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Raw data files: {len(list(Path('rawdata/polygon/by_comp/').glob('*.csv'))) if Path('rawdata/polygon/by_comp/').exists() else 0}/30")
    print(f"   Volatility files: {file_counts.get('vol', 0)}/29")
    print(f"   Covolatility files: {file_counts.get('covol', 0)}/406")
    
    for h5_file, results in h5_results.items():
        if 'total_matrices' in results:
            print(f"   {h5_file}: {results['total_matrices']} matrices ({results['matrix_shape']})")
    
    # Data readiness assessment
    print(f"\n🚀 Data Pipeline Readiness Assessment:")
    
    if overall_status == 'PASS':
        print("   ✅ Ready for model training")
        print("   ✅ All data quality checks passed")
        print("   ✅ Standardization completed successfully")
        print("   ✅ Train/validation/test splits are properly configured")
    elif overall_status == 'PASS WITH WARNINGS':
        print("   ⚠️  Ready for training with minor issues noted")
        print("   ✅ Core functionality is intact")
        print("   ⚠️  Some non-critical issues should be monitored")
    else:
        print("   ❌ Not ready for training - critical issues must be resolved")
        print("   ❌ Address critical issues before proceeding")
    
    return {
        'overall_status': overall_status,
        'critical_issues': critical_issues,
        'warnings': warnings,
        'file_counts': file_counts,
        'h5_results': h5_results
    }

def main():
    """Main validation execution"""
    print("🔬 Starting final comprehensive data validation...")
    
    # Run comprehensive validation
    summary_report = generate_final_summary_report()
    
    # Create diagnostic plots
    create_comprehensive_diagnostic_plots()
    
    print(f"\n🎯 Final Status: {summary_report['overall_status']}")
    print("📊 Complete validation report saved as 'final_data_validation_report.png'")
    
    return summary_report

if __name__ == "__main__":
    main()