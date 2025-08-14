#!/usr/bin/env python3
"""
Detailed U-Shape Validation for Enhanced Realized Variance Data
==============================================================
Comprehensive analysis to validate intraday volatility patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import random
from datetime import datetime, timedelta
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('default')
sns.set_palette("husl")

class UShapeValidator:
    def __init__(self, vol_file='processed_data/vols_mats_30min.h5'):
        self.vol_file = vol_file
        self.intervals_per_day = 13
        self.symbols = [
            'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
            'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
            'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
        ]
        
        # Trading hours mapping (30-min intervals from 9:30 to 16:00)
        self.time_labels = [
            '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
            '13:00', '13:30', '14:00', '14:30', '15:00', '15:30'
        ]
        
        print(f"Initialized U-Shape validator for {vol_file}")
        print(f"Expected intervals per day: {self.intervals_per_day}")
        
    def load_data_sample(self, max_days=50):
        """Load a sample of volatility data for analysis"""
        with h5py.File(self.vol_file, 'r') as f:
            total_matrices = len(f.keys())
            print(f"Total matrices available: {total_matrices}")
            
            # Calculate total trading days
            total_days = total_matrices // self.intervals_per_day
            print(f"Estimated total trading days: {total_days}")
            
            # Sample some random days
            sample_days = min(max_days, total_days - 10)  # Leave some buffer
            random_days = sorted(random.sample(range(10, total_days - 10), sample_days))
            
            volatility_data = []
            metadata = []
            
            for day_idx in tqdm(random_days, desc="Loading sample data"):
                day_start = day_idx * self.intervals_per_day
                day_end = day_start + self.intervals_per_day
                
                # Check if we have complete day
                if day_end > total_matrices:
                    continue
                    
                day_vols = []
                day_valid = True
                
                for interval_idx in range(day_start, day_end):
                    matrix_key = str(interval_idx)
                    if matrix_key in f:
                        matrix = f[matrix_key][:]
                        diagonal_vols = np.diag(matrix)
                        
                        # Check for valid volatilities
                        if np.any(diagonal_vols <= 0) or np.any(np.isnan(diagonal_vols)):
                            day_valid = False
                            break
                            
                        day_vols.append(diagonal_vols)
                    else:
                        day_valid = False
                        break
                
                if day_valid and len(day_vols) == self.intervals_per_day:
                    volatility_data.append(np.array(day_vols))  # Shape: [13, 30]
                    metadata.append({
                        'day_idx': day_idx,
                        'start_matrix': day_start,
                        'end_matrix': day_end - 1
                    })
        
        print(f"Loaded {len(volatility_data)} complete trading days")
        return np.array(volatility_data), metadata
    
    def analyze_u_shape_by_asset(self, volatility_data, metadata, num_assets=6):
        """Analyze U-shape for individual assets"""
        n_days, n_intervals, n_assets = volatility_data.shape
        
        # Select random assets for detailed analysis
        selected_assets = random.sample(range(n_assets), num_assets)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, asset_idx in enumerate(selected_assets):
            ax = axes[i]
            
            # Get volatility for this asset across all days and intervals
            asset_vols = volatility_data[:, :, asset_idx]  # Shape: [n_days, 13]
            
            # Calculate statistics
            mean_pattern = np.mean(asset_vols, axis=0)
            std_pattern = np.std(asset_vols, axis=0)
            median_pattern = np.median(asset_vols, axis=0)
            
            # Plot individual days (light lines)
            for day_idx in range(min(20, n_days)):  # Show max 20 days
                ax.plot(range(n_intervals), asset_vols[day_idx], 
                       color='lightblue', alpha=0.3, linewidth=0.5)
            
            # Plot statistics
            ax.plot(range(n_intervals), mean_pattern, 'b-', linewidth=3, 
                   label='Mean', marker='o', markersize=6)
            ax.plot(range(n_intervals), median_pattern, 'r--', linewidth=2, 
                   label='Median', marker='s', markersize=4)
            
            # Add confidence bands
            ax.fill_between(range(n_intervals), 
                          mean_pattern - std_pattern,
                          mean_pattern + std_pattern,
                          alpha=0.2, color='blue', label='±1 Std')
            
            # Formatting
            ax.set_title(f'{self.symbols[asset_idx]} - Intraday Volatility Pattern\n'
                        f'({n_days} trading days)', fontsize=12, fontweight='bold')
            ax.set_xlabel('30-Min Interval')
            ax.set_ylabel('Realized Volatility')
            ax.set_xticks(range(n_intervals))
            ax.set_xticklabels(self.time_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Annotate U-shape characteristics
            morning_vol = mean_pattern[0]
            midday_vol = np.min(mean_pattern[3:10])  # Avoid first/last few intervals
            afternoon_vol = mean_pattern[-1]
            
            u_ratio = (morning_vol + afternoon_vol) / (2 * midday_vol)
            ax.text(0.02, 0.98, f'U-Ratio: {u_ratio:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('validation_results/u_shape_by_asset_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return selected_assets
    
    def analyze_u_shape_aggregate(self, volatility_data, metadata):
        """Analyze aggregate U-shape pattern"""
        n_days, n_intervals, n_assets = volatility_data.shape
        
        # Calculate cross-sectional average volatility for each interval
        avg_vols = np.mean(volatility_data, axis=2)  # Average across assets: [n_days, 13]
        
        # Calculate statistics across days
        mean_pattern = np.mean(avg_vols, axis=0)
        std_pattern = np.std(avg_vols, axis=0)
        median_pattern = np.median(avg_vols, axis=0)
        q25_pattern = np.percentile(avg_vols, 25, axis=0)
        q75_pattern = np.percentile(avg_vols, 75, axis=0)
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Individual days + aggregate pattern
        for day_idx in range(min(30, n_days)):
            ax1.plot(range(n_intervals), avg_vols[day_idx], 
                    color='lightgray', alpha=0.4, linewidth=0.8)
        
        ax1.plot(range(n_intervals), mean_pattern, 'b-', linewidth=4, 
                label='Mean Pattern', marker='o', markersize=8)
        ax1.plot(range(n_intervals), median_pattern, 'r--', linewidth=3, 
                label='Median Pattern', marker='s', markersize=6)
        
        ax1.fill_between(range(n_intervals), q25_pattern, q75_pattern,
                        alpha=0.3, color='green', label='IQR (25%-75%)')
        
        ax1.set_title(f'Aggregate Intraday Volatility Pattern\n{n_days} Trading Days, {n_assets} Assets', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('30-Min Interval')
        ax1.set_ylabel('Average Realized Volatility')
        ax1.set_xticks(range(n_intervals))
        ax1.set_xticklabels(self.time_labels, rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Normalized U-shape
        # Normalize each day's pattern to have mean=1
        normalized_patterns = []
        for day_idx in range(n_days):
            day_pattern = avg_vols[day_idx]
            if np.mean(day_pattern) > 0:
                normalized = day_pattern / np.mean(day_pattern)
                normalized_patterns.append(normalized)
        
        normalized_patterns = np.array(normalized_patterns)
        norm_mean = np.mean(normalized_patterns, axis=0)
        norm_std = np.std(normalized_patterns, axis=0)
        
        ax2.plot(range(n_intervals), norm_mean, 'purple', linewidth=4, 
                marker='o', markersize=8, label='Normalized Mean')
        ax2.fill_between(range(n_intervals), 
                        norm_mean - norm_std, norm_mean + norm_std,
                        alpha=0.3, color='purple')
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Daily Average')
        
        ax2.set_title('Normalized Intraday Pattern (Daily Mean = 1)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('30-Min Interval')
        ax2.set_ylabel('Relative Volatility')
        ax2.set_xticks(range(n_intervals))
        ax2.set_xticklabels(self.time_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: U-shape metrics by day
        u_ratios = []
        morning_afternoon_ratios = []
        
        for day_idx in range(n_days):
            pattern = avg_vols[day_idx]
            morning_vol = pattern[0]
            afternoon_vol = pattern[-1]
            midday_vol = np.min(pattern[3:10])  # Middle intervals
            
            if midday_vol > 0:
                u_ratio = (morning_vol + afternoon_vol) / (2 * midday_vol)
                ma_ratio = (morning_vol + afternoon_vol) / np.mean(pattern)
                u_ratios.append(u_ratio)
                morning_afternoon_ratios.append(ma_ratio)
        
        ax3.hist(u_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(u_ratios), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(u_ratios):.2f}')
        ax3.axvline(1.0, color='gray', linestyle=':', alpha=0.7, label='No U-shape')
        ax3.set_title('Distribution of U-Shape Ratios\n(Morning + Afternoon) / (2 × Midday Min)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('U-Shape Ratio')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volatility by time period
        time_periods = {
            'Morning Open (9:30-10:30)': [0, 1, 2],
            'Mid-Morning (11:00-12:00)': [3, 4, 5],
            'Midday (12:30-13:30)': [6, 7],
            'Mid-Afternoon (14:00-15:00)': [8, 9, 10],
            'Close (15:30-16:00)': [11, 12]
        }
        
        period_vols = {}
        for period_name, intervals in time_periods.items():
            period_vol = np.mean(avg_vols[:, intervals], axis=1)
            period_vols[period_name] = period_vol
        
        box_data = [period_vols[period] for period in time_periods.keys()]
        bp = ax4.boxplot(box_data, labels=list(time_periods.keys()), patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.set_title('Volatility Distribution by Time Period', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Realized Volatility')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('validation_results/u_shape_aggregate_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("U-SHAPE VALIDATION SUMMARY")
        print("="*60)
        print(f"Sample size: {n_days} trading days, {n_assets} assets")
        print(f"Average U-ratio: {np.mean(u_ratios):.3f} (>1.0 indicates U-shape)")
        print(f"U-ratio std: {np.std(u_ratios):.3f}")
        print(f"% of days with U-ratio > 1.0: {100 * np.mean(np.array(u_ratios) > 1.0):.1f}%")
        
        # Analyze specific intervals
        morning_avg = np.mean(mean_pattern[:3])
        midday_avg = np.mean(mean_pattern[5:8])
        afternoon_avg = np.mean(mean_pattern[-3:])
        
        print(f"\nInterval Analysis:")
        print(f"Morning (9:30-11:00): {morning_avg:.6f}")
        print(f"Midday (12:00-13:00): {midday_avg:.6f}")
        print(f"Afternoon (14:30-16:00): {afternoon_avg:.6f}")
        print(f"Morning/Midday ratio: {morning_avg/midday_avg:.2f}")
        print(f"Afternoon/Midday ratio: {afternoon_avg/midday_avg:.2f}")
        
        return {
            'u_ratios': u_ratios,
            'mean_pattern': mean_pattern,
            'morning_avg': morning_avg,
            'midday_avg': midday_avg,
            'afternoon_avg': afternoon_avg
        }
    
    def analyze_random_periods(self, volatility_data, metadata, num_periods=6):
        """Analyze random time periods in detail"""
        n_days, n_intervals, n_assets = volatility_data.shape
        
        # Select random periods (each period = 5 consecutive days)
        period_length = 5
        max_start = n_days - period_length
        period_starts = random.sample(range(max_start), num_periods)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, start_day in enumerate(period_starts):
            ax = axes[i]
            end_day = start_day + period_length
            
            # Get data for this period
            period_data = volatility_data[start_day:end_day]  # [5, 13, 30]
            
            # Calculate average across assets for each day
            daily_patterns = np.mean(period_data, axis=2)  # [5, 13]
            
            # Plot each day in the period
            colors = plt.cm.Set3(np.linspace(0, 1, period_length))
            for day_idx in range(period_length):
                ax.plot(range(n_intervals), daily_patterns[day_idx], 
                       color=colors[day_idx], linewidth=2, marker='o', 
                       label=f'Day {start_day + day_idx + 1}')
            
            # Plot period average
            period_avg = np.mean(daily_patterns, axis=0)
            ax.plot(range(n_intervals), period_avg, 'black', linewidth=4, 
                   marker='s', markersize=8, label='Period Average')
            
            # Calculate U-ratio for period
            morning_vol = period_avg[0]
            afternoon_vol = period_avg[-1]
            midday_vol = np.min(period_avg[3:10])
            u_ratio = (morning_vol + afternoon_vol) / (2 * midday_vol)
            
            ax.set_title(f'Period {i+1}: Days {start_day+1}-{end_day}\n'
                        f'U-Ratio: {u_ratio:.2f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('30-Min Interval')
            ax.set_ylabel('Average Volatility')
            ax.set_xticks(range(n_intervals))
            ax.set_xticklabels(self.time_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('validation_results/u_shape_random_periods.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main validation function"""
    print("="*80)
    print("ENHANCED REALIZED VARIANCE U-SHAPE VALIDATION")
    print("="*80)
    
    # Create output directory
    import os
    os.makedirs('validation_results', exist_ok=True)
    
    # Initialize validator
    validator = UShapeValidator()
    
    # Load sample data
    print("\nLoading volatility data...")
    volatility_data, metadata = validator.load_data_sample(max_days=100)
    
    if len(volatility_data) == 0:
        print("❌ No valid data found for analysis")
        return
    
    print(f"✅ Loaded data shape: {volatility_data.shape}")
    print(f"   {volatility_data.shape[0]} trading days")
    print(f"   {volatility_data.shape[1]} intervals per day")  
    print(f"   {volatility_data.shape[2]} assets")
    
    # Run analyses
    print("\n1. Analyzing U-shape by individual assets...")
    selected_assets = validator.analyze_u_shape_by_asset(volatility_data, metadata)
    
    print("\n2. Analyzing aggregate U-shape pattern...")
    summary = validator.analyze_u_shape_aggregate(volatility_data, metadata)
    
    print("\n3. Analyzing random time periods...")
    validator.analyze_random_periods(volatility_data, metadata)
    
    print("\n" + "="*80)
    print("✅ U-SHAPE VALIDATION COMPLETE")
    print("="*80)
    print("Generated plots:")
    print("  - validation_results/u_shape_by_asset_detailed.png")
    print("  - validation_results/u_shape_aggregate_analysis.png")
    print("  - validation_results/u_shape_random_periods.png")
    print("\nCheck these plots to validate the U-shape pattern in your data!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()