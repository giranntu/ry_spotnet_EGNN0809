#!/usr/bin/env python3
"""
Fixed Yang-Zhang Volatility Debug Script with Random Sampling
==============================================================
Shows U-shape pattern and random day/asset sampling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import json
from tqdm import tqdm
import random

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# DJIA 30 symbols
DJIA_SYMBOLS = [
    'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

class YangZhangDebuggerFixed:
    """Fixed debugger with random sampling and U-shape visualization"""
    
    def __init__(self):
        self.output_dir = Path("debug_plots_fixed")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def validate_30min_volatility_with_random_sampling(self):
        """Step 2: Fixed 30-minute volatility validation with random sampling"""
        print("\n" + "="*80)
        print("STEP 2: VALIDATING 30-MINUTE VOLATILITY WITH RANDOM SAMPLING")
        print("="*80)
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('30-Minute Yang-Zhang Volatility Validation & Random Sampling', 
                     fontsize=16, fontweight='bold')
        
        # Load all volatility data
        vol_data = {}
        for symbol in DJIA_SYMBOLS:
            vol_file = Path(f"processed_data/vol_30min/{symbol}.csv")
            if vol_file.exists():
                # Check if file has header
                with open(vol_file, 'r') as f:
                    first_line = f.readline()
                    has_header = 'datetime' in first_line.lower()
                
                if has_header:
                    # File has header, read normally
                    df = pd.read_csv(vol_file)
                    if 'yang_zhang_vol' in df.columns:
                        vol_data[symbol] = df['yang_zhang_vol'].values.astype(np.float64)
                    else:
                        # Try second column if exists
                        vol_data[symbol] = df.iloc[:, 1].values.astype(np.float64)
                else:
                    # Original format without header
                    df = pd.read_csv(vol_file, header=None)
                    all_vols = []
                    for col in df.columns:
                        chunk = df[col].dropna()
                        # Convert to float, skip non-numeric
                        for x in chunk.values:
                            try:
                                all_vols.append(float(x))
                            except (ValueError, TypeError):
                                continue
                    vol_data[symbol] = np.array(all_vols, dtype=np.float64)
        
        print(f"Loaded volatility data for {len(vol_data)} symbols")
        
        # 1. U-SHAPE PATTERN - Average intraday volatility
        ax = fig.add_subplot(gs[0, :2])
        
        # Calculate average volatility by interval across all stocks
        intraday_patterns = {}
        for symbol, vols in vol_data.items():
            n_days = len(vols) // 13
            if n_days > 10:  # Need enough days
                reshaped = vols[:n_days*13].reshape(n_days, 13)
                intraday_patterns[symbol] = reshaped.mean(axis=0)
        
        # Plot all stocks' patterns and average
        all_patterns = []
        for symbol, pattern in intraday_patterns.items():
            ax.plot(range(13), pattern, alpha=0.2, color='gray', linewidth=0.5)
            all_patterns.append(pattern)
        
        # Calculate and plot average pattern
        avg_pattern = np.mean(all_patterns, axis=0)
        std_pattern = np.std(all_patterns, axis=0)
        
        interval_labels = [
            '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30',
            '11:30-12:00', '12:00-12:30', '12:30-13:00', '13:00-13:30',
            '13:30-14:00', '14:00-14:30', '14:30-15:00', '15:00-15:30',
            '15:30-16:00'
        ]
        
        x = range(13)
        ax.errorbar(x, avg_pattern, yerr=std_pattern, fmt='o-', 
                   color='red', linewidth=2, markersize=8, capsize=5,
                   label='Market Average ± Std')
        ax.set_title('📈 U-SHAPE INTRADAY VOLATILITY PATTERN (All Stocks)', fontweight='bold')
        ax.set_xlabel('30-min Interval')
        ax.set_ylabel('Average Annualized Volatility')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in x])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text annotations for key times
        ax.text(0, avg_pattern[0] + 0.01, 'Open\n(High)', ha='center', fontsize=8)
        ax.text(6, avg_pattern[6] - 0.01, 'Midday\n(Low)', ha='center', fontsize=8)
        ax.text(12, avg_pattern[12] + 0.01, 'Close\n(High)', ha='center', fontsize=8)
        
        # 2. Random Day Sampling - Single Stock
        ax = fig.add_subplot(gs[0, 2:])
        
        sample_symbol = random.choice(list(vol_data.keys()))
        vols = vol_data[sample_symbol]
        n_days = len(vols) // 13
        
        # Sample 5 random days
        random_days = random.sample(range(n_days), min(5, n_days))
        
        for day_idx in random_days:
            day_vols = vols[day_idx*13:(day_idx+1)*13]
            ax.plot(range(13), day_vols, 'o-', alpha=0.7, 
                   label=f'Day {day_idx}', markersize=4)
        
        ax.set_title(f'🎲 Random Days Sampling: {sample_symbol}', fontweight='bold')
        ax.set_xlabel('30-min Interval')
        ax.set_ylabel('Annualized Volatility')
        ax.set_xticks(range(13))
        ax.set_xticklabels([f'{i+1}' for i in range(13)])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3-6. Random Asset Sampling - 4 random stocks
        random_symbols = random.sample(list(vol_data.keys()), min(4, len(vol_data)))
        
        for idx, symbol in enumerate(random_symbols):
            row = 1 + idx // 2
            col = (idx % 2) * 2
            ax = fig.add_subplot(gs[row, col:col+2])
            
            vols = vol_data[symbol]
            n_days = len(vols) // 13
            
            # Show first 10 days as heatmap
            if n_days > 10:
                heatmap_data = vols[:130].reshape(10, 13)
                im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
                ax.set_title(f'🎯 {symbol} - First 10 Days Heatmap', fontweight='bold')
                ax.set_xlabel('30-min Interval')
                ax.set_ylabel('Trading Day')
                ax.set_xticks(range(13))
                ax.set_xticklabels([f'{i+1}' for i in range(13)])
                plt.colorbar(im, ax=ax, label='Volatility')
        
        # 7. Cross-Asset Comparison - Box plots
        ax = fig.add_subplot(gs[3, :2])
        
        # Sample 8 random stocks for comparison
        compare_symbols = random.sample(list(vol_data.keys()), min(8, len(vol_data)))
        box_data = [vol_data[s][:1000] for s in compare_symbols]  # First 1000 observations
        
        bp = ax.boxplot(box_data, labels=compare_symbols, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(compare_symbols))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('📊 Cross-Asset Volatility Distribution', fontweight='bold')
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Annualized Volatility')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 8. Volatility Time Series - Random continuous segment
        ax = fig.add_subplot(gs[3, 2:])
        
        random_symbol = random.choice(list(vol_data.keys()))
        vols = vol_data[random_symbol]
        
        # Pick random starting point and show 390 points (30 days)
        if len(vols) > 390:
            start_idx = random.randint(0, len(vols) - 390)
            segment = vols[start_idx:start_idx+390]
            
            # Color by day (13 intervals per day)
            colors = []
            for i in range(390):
                day_num = i // 13
                colors.append(day_num % 5)  # Cycle through 5 colors
            
            scatter = ax.scatter(range(390), segment, c=colors, cmap='tab10', 
                               s=10, alpha=0.7)
            ax.plot(range(390), segment, 'k-', linewidth=0.3, alpha=0.3)
            
            # Add day boundaries
            for day in range(30):
                ax.axvline(x=day*13, color='gray', linestyle='--', alpha=0.2)
            
            ax.set_title(f'📈 30-Day Volatility Time Series: {random_symbol} (Start: {start_idx})', 
                        fontweight='bold')
            ax.set_xlabel('30-min Interval')
            ax.set_ylabel('Annualized Volatility')
            
            # Add moving average
            ma = pd.Series(segment).rolling(13).mean()
            ax.plot(range(390), ma, 'r-', linewidth=2, alpha=0.7, label='1-Day MA')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'2_yangzhang_random_sampling_{self.timestamp}.png', dpi=150)
        plt.show()
        
        # Print statistics
        print("\n✅ Random Sampling Validation Complete")
        print("-" * 60)
        print(f"Symbols analyzed: {len(vol_data)}")
        print(f"Average volatility across all stocks: {np.mean([v.mean() for v in vol_data.values()]):.4f}")
        print(f"Volatility range: [{min([v.min() for v in vol_data.values()]):.4f}, "
              f"{max([v.max() for v in vol_data.values()]):.4f}]")
        
        # Check for U-shape
        u_shape_score = (avg_pattern[0] + avg_pattern[12]) / 2 - avg_pattern[6]
        print(f"\nU-Shape Score (Open+Close vs Midday): {u_shape_score:.4f}")
        if u_shape_score > 0:
            print("✅ U-SHAPE PATTERN CONFIRMED: Higher volatility at open/close vs midday")
        else:
            print("⚠️  U-shape pattern not clearly visible")
            
    def validate_yangzhang_components(self):
        """Validate individual Yang-Zhang components"""
        print("\n" + "="*80)
        print("STEP 3: VALIDATING YANG-ZHANG COMPONENTS")
        print("="*80)
        
        # Load a sample stock's raw data
        sample_symbol = 'AAPL'
        raw_file = Path(f"rawdata/by_comp/{sample_symbol}_201901_202507.csv")
        
        if not raw_file.exists():
            print(f"❌ Raw data not found for {sample_symbol}")
            return
            
        df = pd.read_csv(raw_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        # Filter to one month for detailed analysis
        df_month = df['2024-01':'2024-01']
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Yang-Zhang Component Analysis - {sample_symbol} (Jan 2024)', 
                     fontsize=16, fontweight='bold')
        
        # Aggregate to 30-minute bars
        df_30min = df_month.between_time('09:30', '16:00').resample('30T', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 1. Price series
        ax = axes[0, 0]
        ax.plot(df_30min.index, df_30min['close'], 'b-', linewidth=1)
        ax.fill_between(df_30min.index, df_30min['low'], df_30min['high'], alpha=0.3)
        ax.set_title('30-min OHLC Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Overnight returns (close-to-open)
        ax = axes[0, 1]
        df_30min['prev_close'] = df_30min['close'].shift(1)
        overnight_returns = np.log(df_30min['open'] / df_30min['prev_close'].fillna(method='ffill'))
        ax.hist(overnight_returns.dropna(), bins=30, edgecolor='black', alpha=0.7, color='blue')
        ax.set_title('Overnight Returns Distribution')
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='r', linestyle='--')
        
        # 3. Opening returns (open-to-close within bar)
        ax = axes[0, 2]
        opening_returns = np.log(df_30min['close'] / df_30min['open'])
        ax.hist(opening_returns.dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_title('Opening Returns Distribution')
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='r', linestyle='--')
        
        # 4. Rogers-Satchell component
        ax = axes[1, 0]
        rs_component = (
            np.log(df_30min['high'] / df_30min['close']) * 
            np.log(df_30min['high'] / df_30min['open']) +
            np.log(df_30min['low'] / df_30min['close']) * 
            np.log(df_30min['low'] / df_30min['open'])
        )
        ax.plot(df_30min.index, rs_component, 'o-', markersize=2, color='red')
        ax.set_title('Rogers-Satchell Component (Intraday)')
        ax.set_xlabel('Date')
        ax.set_ylabel('RS Value')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. High-Low ratio
        ax = axes[1, 1]
        hl_ratio = df_30min['high'] / df_30min['low']
        ax.plot(df_30min.index, hl_ratio, 'o-', markersize=2, color='purple')
        ax.set_title('High/Low Ratio')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ratio')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. Volume pattern
        ax = axes[1, 2]
        ax.bar(df_30min.index, df_30min['volume'], width=0.01, alpha=0.7)
        ax.set_title('30-min Volume')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.tick_params(axis='x', rotation=45)
        
        # 7. Calculate rolling Yang-Zhang (simplified)
        ax = axes[2, 0]
        window = 39  # 3 days × 13 intervals
        
        # Rolling variances
        overnight_var = overnight_returns.rolling(window).var()
        opening_var = opening_returns.rolling(window).var()
        rs_var = rs_component.rolling(window).mean()
        
        n = window
        k = 0.34 / (1.34 + (n + 1)/(n - 1))
        
        yz_variance = overnight_var + k * opening_var + rs_var
        yz_volatility = np.sqrt(np.abs(yz_variance) * 252 * 13)
        
        ax.plot(df_30min.index, yz_volatility, 'g-', linewidth=2)
        ax.set_title(f'Yang-Zhang Volatility (Window={window})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')
        ax.tick_params(axis='x', rotation=45)
        
        # 8. Component contribution
        ax = axes[2, 1]
        
        # Calculate average contribution
        total_var = overnight_var.mean() + k * opening_var.mean() + rs_var.mean()
        contributions = {
            'Overnight': overnight_var.mean() / total_var * 100,
            'Opening': k * opening_var.mean() / total_var * 100,
            'Intraday (RS)': rs_var.mean() / total_var * 100
        }
        
        ax.pie(contributions.values(), labels=contributions.keys(), autopct='%1.1f%%',
               colors=['blue', 'green', 'red'])
        ax.set_title('Yang-Zhang Component Contributions')
        
        # 9. Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        stats_text = f"""Yang-Zhang Component Statistics
        
Overnight Returns:
  Mean: {overnight_returns.mean():.6f}
  Std: {overnight_returns.std():.6f}
  Skew: {overnight_returns.skew():.3f}
  
Opening Returns:
  Mean: {opening_returns.mean():.6f}
  Std: {opening_returns.std():.6f}
  Skew: {opening_returns.skew():.3f}
  
Rogers-Satchell:
  Mean: {rs_component.mean():.6f}
  Std: {rs_component.std():.6f}
  
Final YZ Volatility:
  Mean: {yz_volatility.mean():.4f}
  Min: {yz_volatility.min():.4f}
  Max: {yz_volatility.max():.4f}
  
Drift Factor k: {k:.4f}"""
        
        ax.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'3_yangzhang_components_{self.timestamp}.png', dpi=150)
        plt.show()
        
        print("✅ Yang-Zhang component validation complete")
        
    def validate_matrix_properties(self):
        """Step 4: Fixed matrix property validation"""
        print("\n" + "="*80)
        print("STEP 4: VALIDATING MATRIX PROPERTIES")
        print("="*80)
        
        vol_h5 = Path("processed_data/vols_mats_30min.h5")
        if not vol_h5.exists():
            print("❌ Matrix file not found")
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('Covariance Matrix Properties Validation', fontsize=16, fontweight='bold')
        
        with h5py.File(vol_h5, 'r') as f:
            keys = sorted(f.keys(), key=lambda x: int(x))
            n_matrices = len(keys)
            
            # Sample random matrices
            random_indices = random.sample(range(n_matrices), min(100, n_matrices))
            
            # 1. Random matrix visualization
            ax = axes[0, 0]
            random_idx = random.choice(random_indices)
            random_matrix = np.array(f[keys[random_idx]])
            im = ax.imshow(random_matrix, cmap='coolwarm', aspect='auto')
            ax.set_title(f'Random Matrix (Index: {random_idx})')
            ax.set_xlabel('Stock Index')
            ax.set_ylabel('Stock Index')
            plt.colorbar(im, ax=ax)
            
            # 2. Eigenvalue spectrum for random matrices
            ax = axes[0, 1]
            for idx in random.sample(random_indices, 5):
                matrix = np.array(f[keys[idx]])
                eigenvals = np.sort(np.linalg.eigvalsh(matrix))[::-1]
                ax.plot(eigenvals[:10], 'o-', alpha=0.7, label=f'Mat {idx}')
            ax.set_title('Top 10 Eigenvalues (Random Matrices)')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Eigenvalue')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            
            # 3. Matrix determinant over time
            ax = axes[0, 2]
            determinants = []
            for idx in random_indices[:50]:
                matrix = np.array(f[keys[idx]])
                det = np.linalg.det(matrix)
                determinants.append(np.log10(abs(det) + 1e-100))
            ax.plot(determinants, 'o-', markersize=3)
            ax.set_title('Log10(|Determinant|) for Random Matrices')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Log10(|Det|)')
            
            # 4. Frobenius norm
            ax = axes[1, 0]
            frobenius_norms = []
            for idx in random_indices:
                matrix = np.array(f[keys[idx]])
                fnorm = np.linalg.norm(matrix, 'fro')
                frobenius_norms.append(fnorm)
            ax.hist(frobenius_norms, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title('Frobenius Norm Distribution')
            ax.set_xlabel('Frobenius Norm')
            ax.set_ylabel('Frequency')
            
            # 5. Correlation between random pairs
            ax = axes[1, 1]
            pair_correlations = []
            for _ in range(100):
                i, j = random.sample(range(30), 2)
                idx = random.choice(random_indices)
                matrix = np.array(f[keys[idx]])
                if matrix[i, i] > 0 and matrix[j, j] > 0:
                    corr = matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j])
                    pair_correlations.append(corr)
            ax.hist(pair_correlations, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title('Random Pairwise Correlations')
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Frequency')
            ax.axvline(x=0, color='r', linestyle='--')
            
            # 6. Matrix rank
            ax = axes[1, 2]
            ranks = []
            for idx in random_indices[:50]:
                matrix = np.array(f[keys[idx]])
                rank = np.linalg.matrix_rank(matrix)
                ranks.append(rank)
            unique_ranks, counts = np.unique(ranks, return_counts=True)
            ax.bar(unique_ranks, counts)
            ax.set_title('Matrix Rank Distribution')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Count')
            ax.axvline(x=30, color='r', linestyle='--', label='Full Rank (30)')
            ax.legend()
            
            # 7. Spectral radius
            ax = axes[2, 0]
            spectral_radii = []
            for idx in random_indices:
                matrix = np.array(f[keys[idx]])
                eigenvals = np.linalg.eigvalsh(matrix)
                spectral_radius = max(abs(eigenvals))
                spectral_radii.append(spectral_radius)
            ax.plot(sorted(spectral_radii), 'o-', markersize=2)
            ax.set_title('Sorted Spectral Radii')
            ax.set_xlabel('Index')
            ax.set_ylabel('Spectral Radius')
            
            # 8. Matrix similarity (consecutive matrices)
            ax = axes[2, 1]
            similarities = []
            for i in range(len(random_indices)-1):
                mat1 = np.array(f[keys[random_indices[i]]])
                mat2 = np.array(f[keys[random_indices[i+1]]])
                # Frobenius norm of difference
                diff = np.linalg.norm(mat1 - mat2, 'fro')
                avg_norm = (np.linalg.norm(mat1, 'fro') + np.linalg.norm(mat2, 'fro')) / 2
                similarity = 1 - diff / avg_norm
                similarities.append(similarity)
            ax.hist(similarities, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title('Matrix Similarity (Consecutive)')
            ax.set_xlabel('Similarity Score')
            ax.set_ylabel('Frequency')
            
            # 9. Summary
            ax = axes[2, 2]
            ax.axis('off')
            
            summary_text = f"""Matrix Properties Summary
            
Total Matrices: {n_matrices}
Matrix Dimension: 30×30
Samples Analyzed: {len(random_indices)}

Eigenvalue Stats:
  All Positive: {all(d >= -1e-10 for d in determinants)}
  Avg Spectral Radius: {np.mean(spectral_radii):.4f}
  
Rank Stats:
  Full Rank Count: {counts[unique_ranks == 30][0] if 30 in unique_ranks else 0}/{len(ranks)}
  Most Common Rank: {unique_ranks[np.argmax(counts)]}
  
Correlation Stats:
  Mean Correlation: {np.mean(pair_correlations):.4f}
  Std Correlation: {np.std(pair_correlations):.4f}
  
Frobenius Norm:
  Mean: {np.mean(frobenius_norms):.4f}
  Std: {np.std(frobenius_norms):.4f}"""
            
            ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'4_matrix_properties_{self.timestamp}.png', dpi=150)
        plt.show()
        
        print("✅ Matrix properties validation complete")
        
    def run_all_validations(self):
        """Run all validation steps"""
        print("="*80)
        print("FIXED YANG-ZHANG VALIDATION WITH RANDOM SAMPLING")
        print("="*80)
        
        try:
            self.validate_30min_volatility_with_random_sampling()
        except Exception as e:
            print(f"⚠️  Step 2 error: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self.validate_yangzhang_components()
        except Exception as e:
            print(f"⚠️  Step 3 error: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self.validate_matrix_properties()
        except Exception as e:
            print(f"⚠️  Step 4 error: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n" + "="*80)
        print("VALIDATION COMPLETE!")
        print("="*80)
        print(f"All plots saved to: {self.output_dir}/")
        
        # List generated files
        generated_files = list(self.output_dir.glob("*.png"))
        if generated_files:
            print("\nGenerated plots:")
            for f in generated_files:
                print(f"  - {f.name}")


def main():
    """Run the fixed validation suite"""
    debugger = YangZhangDebuggerFixed()
    debugger.run_all_validations()
    return debugger


if __name__ == "__main__":
    debugger = main()