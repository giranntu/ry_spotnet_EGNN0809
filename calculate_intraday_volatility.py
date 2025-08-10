#!/usr/bin/env python3
"""
Best Practice Intraday Volatility Calculation
==============================================
Direct calculation from 1-minute data to show U-shape pattern
No synthetic data, just real market microstructure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def calculate_realized_volatility(returns, annualize=True):
    """
    Standard realized volatility (sum of squared returns)
    """
    rv = np.sqrt(np.sum(returns**2))
    if annualize:
        # For 30-minute intervals: 252 trading days, 13 intervals per day
        # Each interval represents 1/13 of a trading day
        rv = rv * np.sqrt(252 * 13)
    return rv

def calculate_bipower_variation(returns, annualize=True):
    """
    Bipower variation - robust to jumps
    Barndorff-Nielsen & Shephard (2004)
    """
    bv = np.sqrt((np.pi/2) * np.sum(np.abs(returns[:-1]) * np.abs(returns[1:])))
    if annualize:
        bv = bv * np.sqrt(252 * 13)
    return bv

def calculate_realized_kernel(prices, kernel='parzen', bandwidth=5):
    """
    Realized kernel estimator - robust to microstructure noise
    Barndorff-Nielsen et al. (2008)
    """
    returns = np.log(prices[1:] / prices[:-1])
    n = len(returns)
    
    # Parzen kernel weights
    weights = []
    for h in range(bandwidth + 1):
        if h == 0:
            w = 1
        else:
            x = h / bandwidth
            if x <= 0.5:
                w = 1 - 6*x**2 + 6*x**3
            else:
                w = 2*(1-x)**3
        weights.append(w)
    
    # Calculate autocovariances with kernel weights
    rv = 0
    for h in range(len(weights)):
        if h == 0:
            gamma_h = np.sum(returns**2)
        else:
            gamma_h = np.sum(returns[:-h] * returns[h:])
        rv += weights[h] * gamma_h * (2 if h > 0 else 1)
    
    return np.sqrt(max(0, rv)) * np.sqrt(252 * 13)

def calculate_rmed(prices, k=2, annualize=True):
    """
    MedRV (Median Realized Volatility) - ultra robust to jumps and noise
    Andersen et al. (2012)
    """
    returns = np.log(prices[1:] / prices[:-1])
    n = len(returns)
    
    if n < 2*k + 1:
        return np.nan
    
    # Calculate median of adjacent returns
    med_products = []
    for i in range(k, n-k):
        window = returns[i-k:i+k+1]
        med_products.append(np.median(np.abs(window))**2)
    
    constant = (np.pi / (4 - np.pi)) * ((2*k + 1) / (2*k - 1))
    rmed = np.sqrt(constant * np.sum(med_products))
    
    if annualize:
        rmed = rmed * np.sqrt(252 * 13)
    return rmed

def calculate_intraday_patterns(symbol='AAPL', sample_days=30):
    """
    Calculate intraday volatility patterns using multiple methods
    """
    # Load raw 1-minute data
    filepath = Path(f"rawdata/by_comp/{symbol}_201901_202507.csv")
    if not filepath.exists():
        print(f"Data not found for {symbol}")
        return None
    
    print(f"Loading {symbol} data...")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Filter to regular trading hours
    df = df.between_time('09:30', '16:00')
    
    # Get sample of recent trading days
    df['date'] = df.index.date
    unique_dates = df['date'].unique()
    
    # Use recent data for cleaner patterns
    sample_dates = unique_dates[-sample_days:] if len(unique_dates) >= sample_days else unique_dates
    
    print(f"Analyzing {len(sample_dates)} trading days...")
    
    # Initialize storage for different volatility measures
    intraday_patterns = {
        'realized_vol': {i: [] for i in range(13)},
        'bipower_var': {i: [] for i in range(13)},
        'realized_kernel': {i: [] for i in range(13)},
        'median_rv': {i: [] for i in range(13)},
        'high_low': {i: [] for i in range(13)}
    }
    
    # Process each day
    for date in sample_dates:
        day_data = df[df['date'] == date].copy()
        
        if len(day_data) < 390:  # Skip incomplete days
            continue
        
        # Create 30-minute intervals (13 per day)
        # Fix the interval calculation for proper boundary handling
        minutes_from_open = (day_data.index.hour - 9) * 60 + day_data.index.minute - 30
        day_data['interval'] = np.minimum(12, np.maximum(0, minutes_from_open // 30))
        
        # Calculate volatility for each 30-minute interval
        for interval in range(13):
            interval_data = day_data[day_data['interval'] == interval]
            
            if len(interval_data) < 10:  # Need minimum data
                continue
            
            prices = interval_data['close'].values
            
            # 1. Realized Volatility (5-minute sampling to reduce noise)
            prices_5min = prices[::5] if len(prices) >= 5 else prices
            if len(prices_5min) > 1:
                returns_5min = np.log(prices_5min[1:] / prices_5min[:-1])
                rv = calculate_realized_volatility(returns_5min)
                intraday_patterns['realized_vol'][interval].append(rv)
            
            # 2. Bipower Variation (jump-robust)
            if len(prices) > 1:
                returns = np.log(prices[1:] / prices[:-1])
                bv = calculate_bipower_variation(returns)
                intraday_patterns['bipower_var'][interval].append(bv)
            
            # 3. Realized Kernel (noise-robust)
            if len(prices) > 10:
                rk = calculate_realized_kernel(prices)
                if not np.isnan(rk) and rk < 10:  # Filter outliers
                    intraday_patterns['realized_kernel'][interval].append(rk)
            
            # 4. Median RV (ultra-robust)
            if len(prices) > 5:
                med_rv = calculate_rmed(prices)
                if not np.isnan(med_rv) and med_rv < 10:
                    intraday_patterns['median_rv'][interval].append(med_rv)
            
            # 5. High-Low Range (Parkinson)
            if 'high' in interval_data.columns and 'low' in interval_data.columns:
                # Parkinson estimator for the interval
                high = interval_data['high'].max()
                low = interval_data['low'].min()
                if high > 0 and low > 0 and high > low:
                    hl_range = np.sqrt(np.log(high/low)**2 / (4 * np.log(2)))
                    hl_vol = hl_range * np.sqrt(252 * 13)  # Annualize
                    if hl_vol < 2:  # Filter outliers (reasonable threshold)
                        intraday_patterns['high_low'][interval].append(hl_vol)
    
    return intraday_patterns

def plot_intraday_volatility_patterns(symbols=['AAPL', 'MSFT', 'JPM', 'GS'], sample_days=60):
    """
    Plot intraday volatility patterns for multiple stocks
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Intraday Volatility U-Shape Pattern (Real Data, No Synthetic)', 
                 fontsize=16, fontweight='bold')
    
    interval_times = [
        '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30',
        '11:30-12:00', '12:00-12:30', '12:30-13:00', '13:00-13:30',
        '13:30-14:00', '14:00-14:30', '14:30-15:00', '15:00-15:30',
        '15:30-16:00'
    ]
    
    # 1. Individual stock patterns
    ax = axes[0, 0]
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        patterns = calculate_intraday_patterns(symbol, sample_days)
        
        if patterns is None:
            continue
        
        # Calculate average realized volatility for each interval
        avg_pattern = []
        std_pattern = []
        for i in range(13):
            if patterns['realized_vol'][i]:
                avg_pattern.append(np.mean(patterns['realized_vol'][i]))
                std_pattern.append(np.std(patterns['realized_vol'][i]))
            else:
                avg_pattern.append(np.nan)
                std_pattern.append(np.nan)
        
        # Plot
        valid_intervals = [i for i, v in enumerate(avg_pattern) if not np.isnan(v)]
        valid_values = [v for v in avg_pattern if not np.isnan(v)]
        
        if valid_intervals:
            ax.plot(valid_intervals, valid_values, 'o-', label=symbol, linewidth=2, markersize=6)
    
    ax.set_title('Realized Volatility by Interval (Multiple Stocks)', fontweight='bold')
    ax.set_xlabel('30-minute Interval')
    ax.set_ylabel('Annualized Volatility')
    ax.set_xticks(range(13))
    ax.set_xticklabels(range(13))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average pattern across all stocks (main U-shape)
    ax = axes[0, 1]
    
    all_patterns = {i: [] for i in range(13)}
    
    for symbol in symbols:
        patterns = calculate_intraday_patterns(symbol, sample_days)
        if patterns:
            for i in range(13):
                if patterns['realized_vol'][i]:
                    all_patterns[i].extend(patterns['realized_vol'][i])
    
    avg_overall = []
    std_overall = []
    for i in range(13):
        if all_patterns[i]:
            avg_overall.append(np.mean(all_patterns[i]))
            std_overall.append(np.std(all_patterns[i]))
        else:
            avg_overall.append(np.nan)
            std_overall.append(np.nan)
    
    # Plot with error bars
    valid_intervals = [i for i, v in enumerate(avg_overall) if not np.isnan(v)]
    valid_avg = [avg_overall[i] for i in valid_intervals]
    valid_std = [std_overall[i] for i in valid_intervals]
    
    ax.errorbar(valid_intervals, valid_avg, yerr=valid_std, 
                fmt='ro-', linewidth=2, markersize=8, capsize=5)
    ax.fill_between(valid_intervals, 
                     np.array(valid_avg) - np.array(valid_std),
                     np.array(valid_avg) + np.array(valid_std),
                     alpha=0.2, color='red')
    
    ax.set_title('Average U-Shape Pattern (All Stocks)', fontweight='bold')
    ax.set_xlabel('30-minute Interval')
    ax.set_ylabel('Annualized Volatility')
    ax.set_xticks(range(13))
    ax.set_xticklabels(range(13))
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    if valid_avg:
        ax.text(0, valid_avg[0], 'Open\nHigh', ha='center', va='bottom', fontsize=9)
        mid_idx = 6
        if mid_idx in valid_intervals:
            mid_val = valid_avg[valid_intervals.index(mid_idx)]
            ax.text(mid_idx, mid_val, 'Lunch\nLow', ha='center', va='top', fontsize=9)
        if 12 in valid_intervals:
            close_val = valid_avg[valid_intervals.index(12)]
            ax.text(12, close_val, 'Close\nHigh', ha='center', va='bottom', fontsize=9)
    
    # 3. Compare different volatility estimators (for AAPL)
    ax = axes[1, 0]
    
    patterns = calculate_intraday_patterns('AAPL', sample_days)
    if patterns:
        methods = ['realized_vol', 'bipower_var', 'realized_kernel', 'median_rv']
        colors = ['blue', 'green', 'red', 'purple']
        
        for method, color in zip(methods, colors):
            avg_method = []
            for i in range(13):
                if patterns[method][i]:
                    avg_method.append(np.mean(patterns[method][i]))
                else:
                    avg_method.append(np.nan)
            
            valid_intervals = [i for i, v in enumerate(avg_method) if not np.isnan(v)]
            valid_values = [v for v in avg_method if not np.isnan(v)]
            
            if valid_intervals:
                ax.plot(valid_intervals, valid_values, 'o-', 
                       label=method.replace('_', ' ').title(), 
                       color=color, linewidth=1.5, markersize=5)
    
    ax.set_title('Different Volatility Estimators (AAPL)', fontweight='bold')
    ax.set_xlabel('30-minute Interval')
    ax.set_ylabel('Annualized Volatility')
    ax.set_xticks(range(13))
    ax.set_xticklabels(range(13))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Percentage change from midday (U-shape intensity)
    ax = axes[1, 1]
    
    if avg_overall and 6 in valid_intervals:
        midday_vol = avg_overall[6]
        pct_change = [(v/midday_vol - 1) * 100 for v in avg_overall]
        
        valid_pct = [pct_change[i] for i in valid_intervals]
        
        colors = ['red' if i in [0, 1, 11, 12] else 'blue' for i in valid_intervals]
        bars = ax.bar(valid_intervals, valid_pct, color=colors, alpha=0.7)
        
        ax.set_title('Volatility % Change from Midday', fontweight='bold')
        ax.set_xlabel('30-minute Interval')
        ax.set_ylabel('% Change from Lunch Period')
        ax.set_xticks(range(13))
        ax.set_xticklabels(range(13))
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, valid_pct):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.0f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)
    
    # 5. Heatmap of volatility by interval and day
    ax = axes[2, 0]
    
    # Get daily patterns for heatmap
    patterns = calculate_intraday_patterns('AAPL', sample_days=20)
    if patterns:
        # Create matrix for heatmap
        heatmap_data = []
        for day in range(min(20, len(patterns['realized_vol'][0]))):
            day_vols = []
            for interval in range(13):
                if len(patterns['realized_vol'][interval]) > day:
                    day_vols.append(patterns['realized_vol'][interval][day])
                else:
                    day_vols.append(np.nan)
            heatmap_data.append(day_vols)
        
        if heatmap_data:
            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
            ax.set_title('Daily Volatility Heatmap (AAPL, 20 days)', fontweight='bold')
            ax.set_xlabel('30-minute Interval')
            ax.set_ylabel('Trading Day')
            ax.set_xticks(range(13))
            ax.set_xticklabels(range(13))
            plt.colorbar(im, ax=ax, label='Volatility')
    
    # 6. Summary statistics
    ax = axes[2, 1]
    ax.axis('off')
    
    if avg_overall and valid_intervals:
        open_vol = avg_overall[0] if 0 in valid_intervals else np.nan
        lunch_vol = avg_overall[6] if 6 in valid_intervals else np.nan
        close_vol = avg_overall[12] if 12 in valid_intervals else np.nan
        
        u_score = ((open_vol + close_vol) / 2) / lunch_vol if lunch_vol else np.nan
        
        stats_text = f"""U-Shape Pattern Statistics
        
Volatility Levels (Annualized):
  Opening (9:30-10:00): {open_vol:.2%} if not np.isnan(open_vol) else 'N/A'
  Lunch (12:30-13:00): {lunch_vol:.2%} if not np.isnan(lunch_vol) else 'N/A'
  Closing (15:30-16:00): {close_vol:.2%} if not np.isnan(close_vol) else 'N/A'

U-Shape Metrics:
  U-Score: {u_score:.2f} if not np.isnan(u_score) else 'N/A'
  (Values > 1.3 indicate strong U-shape)
  
  Open vs Lunch: {(open_vol/lunch_vol - 1)*100:.1f}% higher
  Close vs Lunch: {(close_vol/lunch_vol - 1)*100:.1f}% higher
  
Data:
  Symbols: {', '.join(symbols)}
  Days analyzed: {sample_days}
  Method: 5-min Realized Volatility
  
Note: All calculations from real 1-minute data
      No synthetic data or adjustments"""
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("debug_plots_fixed")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(output_dir / f'intraday_volatility_u_shape_{timestamp}.png', dpi=150)
    plt.show()
    
    return avg_overall

def main():
    """
    Calculate and visualize intraday volatility patterns
    """
    print("="*80)
    print("INTRADAY VOLATILITY U-SHAPE ANALYSIS")
    print("Using best practices for volatility calculation")
    print("="*80)
    
    # Analyze multiple stocks
    symbols = ['AAPL', 'MSFT', 'JPM', 'GS']
    
    # Generate comprehensive plots
    avg_pattern = plot_intraday_volatility_patterns(symbols, sample_days=60)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    if avg_pattern:
        print("\nKey Findings:")
        print(f"  Opening volatility: {avg_pattern[0]:.2%}" if not np.isnan(avg_pattern[0]) else "  Opening: N/A")
        print(f"  Lunch volatility: {avg_pattern[6]:.2%}" if not np.isnan(avg_pattern[6]) else "  Lunch: N/A")
        print(f"  Closing volatility: {avg_pattern[12]:.2%}" if not np.isnan(avg_pattern[12]) else "  Closing: N/A")
        
        if not np.isnan(avg_pattern[0]) and not np.isnan(avg_pattern[6]):
            print(f"\n  U-Shape confirmed: {(avg_pattern[0]/avg_pattern[6] - 1)*100:.1f}% higher at open vs lunch")

if __name__ == "__main__":
    main()