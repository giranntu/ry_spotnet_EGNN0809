#!/usr/bin/env python3
"""
Fixed Intraday Volatility Calculation with Proper Scaling
==========================================================
Correctly calculates intraday volatility with realistic values
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

def calculate_intraday_volatility_correct(symbol='AAPL', sample_days=60):
    """
    Calculate intraday volatility correctly for each 30-minute interval
    """
    # Load raw 1-minute data
    filepath = Path(f"rawdata/by_comp/{symbol}_201901_202507.csv")
    if not filepath.exists():
        print(f"Data not found for {symbol}")
        return None
    
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Filter to regular trading hours only
    df = df.between_time('09:30', '16:00')
    
    # Add date column
    df['date'] = df.index.date
    unique_dates = df['date'].unique()
    
    # Use recent data
    sample_dates = unique_dates[-sample_days:] if len(unique_dates) >= sample_days else unique_dates
    
    # Store volatilities for each interval
    interval_volatilities = {i: [] for i in range(13)}
    
    for date in sample_dates:
        day_data = df[df['date'] == date].copy()
        
        if len(day_data) < 390:  # Skip incomplete days
            continue
        
        # Split into 30-minute intervals
        # Interval 0: 9:30-10:00, Interval 1: 10:00-10:30, etc.
        for interval_idx in range(13):
            start_minute = interval_idx * 30
            end_minute = (interval_idx + 1) * 30
            
            # Get the time boundaries
            start_time = pd.Timestamp(f"{date} 09:30:00") + pd.Timedelta(minutes=start_minute)
            end_time = pd.Timestamp(f"{date} 09:30:00") + pd.Timedelta(minutes=end_minute)
            
            # Get data for this interval
            interval_data = day_data[(day_data.index >= start_time) & (day_data.index < end_time)]
            
            if len(interval_data) < 10:  # Need minimum data points
                continue
            
            # Calculate 1-minute returns
            returns = np.log(interval_data['close'].values[1:] / interval_data['close'].values[:-1])
            
            # Remove any infinite or NaN values
            returns = returns[np.isfinite(returns)]
            
            if len(returns) < 5:
                continue
            
            # Calculate realized volatility for this 30-minute interval
            # Standard deviation of 1-minute returns
            interval_vol = np.std(returns)
            
            # Annualize correctly:
            # interval_vol is the std of 1-minute returns
            # To annualize 1-minute volatility:
            # There are 390 minutes per trading day
            # There are 252 trading days per year
            # So total minutes per year = 252 * 390
            annualized_vol = interval_vol * np.sqrt(252 * 390)
            
            interval_volatilities[interval_idx].append(annualized_vol)
    
    return interval_volatilities

def create_comprehensive_plots(symbols=['AAPL', 'MSFT', 'JPM'], sample_days=60):
    """
    Create comprehensive plots for multiple assets
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Intraday Volatility U-Shape Pattern Analysis (Corrected)', 
                 fontsize=16, fontweight='bold')
    
    # Time labels
    time_labels = [
        '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30',
        '11:30-12:00', '12:00-12:30', '12:30-13:00', '13:00-13:30',
        '13:30-14:00', '14:00-14:30', '14:30-15:00', '15:00-15:30',
        '15:30-16:00'
    ]
    
    all_data = {}
    
    # Process each symbol
    for symbol in symbols:
        print(f"Processing {symbol}...")
        vol_data = calculate_intraday_volatility_correct(symbol, sample_days)
        if vol_data:
            all_data[symbol] = vol_data
    
    # Plot 1: Individual asset patterns (3 separate plots)
    for idx, symbol in enumerate(symbols):
        ax = fig.add_subplot(gs[0, idx])
        
        if symbol not in all_data:
            continue
        
        vol_data = all_data[symbol]
        
        # Calculate mean and std for each interval
        means = []
        stds = []
        for i in range(13):
            if vol_data[i]:
                means.append(np.mean(vol_data[i]))
                stds.append(np.std(vol_data[i]))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        # Plot with error bars
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_means = [means[i] for i in valid_idx]
        valid_stds = [stds[i] for i in valid_idx]
        
        ax.errorbar(valid_idx, valid_means, yerr=valid_stds, 
                   fmt='o-', capsize=5, linewidth=2, markersize=6)
        
        ax.set_title(f'{symbol} Intraday Volatility', fontweight='bold')
        ax.set_xlabel('30-min Interval')
        ax.set_ylabel('Annualized Vol (%)')
        ax.set_xticks(range(13))
        ax.set_xticklabels(range(13), fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
        
        # Add U-shape score
        if len(valid_means) >= 10:
            open_vol = means[0] if not np.isnan(means[0]) else 0
            lunch_vol = means[6] if not np.isnan(means[6]) else 1
            close_vol = means[12] if not np.isnan(means[12]) else 0
            u_score = ((open_vol + close_vol) / 2) / lunch_vol if lunch_vol > 0 else 0
            ax.text(0.95, 0.95, f'U-Score: {u_score:.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Combined comparison
    ax = fig.add_subplot(gs[1, :])
    
    for symbol in symbols:
        if symbol not in all_data:
            continue
        
        vol_data = all_data[symbol]
        means = []
        for i in range(13):
            if vol_data[i]:
                means.append(np.mean(vol_data[i]))
            else:
                means.append(np.nan)
        
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_means = [means[i] for i in valid_idx]
        
        ax.plot(valid_idx, valid_means, 'o-', label=symbol, linewidth=2, markersize=6)
    
    ax.set_title('Volatility Comparison Across Assets', fontweight='bold')
    ax.set_xlabel('30-minute Interval')
    ax.set_ylabel('Annualized Volatility (%)')
    ax.set_xticks(range(13))
    ax.set_xticklabels([f'{i}' for i in range(13)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    
    # Add time labels on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xticks(range(13))
    ax2.set_xticklabels(time_labels, rotation=45, fontsize=8)
    ax2.set_xlabel('Time of Day', fontsize=10)
    
    # Plot 3: Average pattern across all assets
    ax = fig.add_subplot(gs[2, 0])
    
    # Combine all data
    combined_intervals = {i: [] for i in range(13)}
    for symbol in all_data:
        for i in range(13):
            combined_intervals[i].extend(all_data[symbol][i])
    
    avg_means = []
    avg_stds = []
    for i in range(13):
        if combined_intervals[i]:
            avg_means.append(np.mean(combined_intervals[i]))
            avg_stds.append(np.std(combined_intervals[i]))
        else:
            avg_means.append(np.nan)
            avg_stds.append(np.nan)
    
    valid_idx = [i for i, m in enumerate(avg_means) if not np.isnan(m)]
    valid_means = [avg_means[i] for i in valid_idx]
    valid_stds = [avg_stds[i] for i in valid_idx]
    
    ax.errorbar(valid_idx, valid_means, yerr=valid_stds,
               fmt='ro-', capsize=5, linewidth=2, markersize=8)
    ax.fill_between(valid_idx,
                    np.array(valid_means) - np.array(valid_stds),
                    np.array(valid_means) + np.array(valid_stds),
                    alpha=0.2, color='red')
    
    ax.set_title('Average U-Shape Pattern', fontweight='bold')
    ax.set_xlabel('30-minute Interval')
    ax.set_ylabel('Annualized Vol (%)')
    ax.set_xticks(range(13))
    ax.set_xticklabels(range(13))
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    
    # Annotate key points
    if valid_means:
        ax.annotate('Open', xy=(0, valid_means[0]), xytext=(0, valid_means[0]+0.002),
                   ha='center', fontsize=8, color='red')
        if 6 in valid_idx:
            lunch_idx = valid_idx.index(6)
            ax.annotate('Lunch', xy=(6, valid_means[lunch_idx]), 
                       xytext=(6, valid_means[lunch_idx]-0.002),
                       ha='center', fontsize=8, color='blue')
        if 12 in valid_idx:
            close_idx = valid_idx.index(12)
            ax.annotate('Close', xy=(12, valid_means[close_idx]),
                       xytext=(12, valid_means[close_idx]+0.002),
                       ha='center', fontsize=8, color='red')
    
    # Plot 4: Percentage change from midday
    ax = fig.add_subplot(gs[2, 1])
    
    if avg_means and not np.isnan(avg_means[6]):
        lunch_vol = avg_means[6]
        pct_changes = [(m/lunch_vol - 1) * 100 if not np.isnan(m) else 0 for m in avg_means]
        
        colors = ['red' if i in [0, 1, 11, 12] else 'blue' for i in range(13)]
        bars = ax.bar(range(13), pct_changes, color=colors, alpha=0.7)
        
        ax.set_title('% Change from Lunch Volatility', fontweight='bold')
        ax.set_xlabel('30-minute Interval')
        ax.set_ylabel('% Change')
        ax.set_xticks(range(13))
        ax.set_xticklabels(range(13))
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pct in zip(bars, pct_changes):
            if pct != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{pct:.0f}%', ha='center', 
                       va='bottom' if pct > 0 else 'top', fontsize=8)
    
    # Plot 5: Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Calculate statistics
    stats_text = "Summary Statistics\n" + "="*30 + "\n\n"
    
    for symbol in symbols:
        if symbol not in all_data:
            continue
        
        vol_data = all_data[symbol]
        means = [np.mean(vol_data[i]) if vol_data[i] else np.nan for i in range(13)]
        
        open_vol = means[0] if not np.isnan(means[0]) else 0
        lunch_vol = means[6] if not np.isnan(means[6]) else 0
        close_vol = means[12] if not np.isnan(means[12]) else 0
        
        stats_text += f"{symbol}:\n"
        stats_text += f"  Open:  {open_vol*100:5.2f}%\n"
        stats_text += f"  Lunch: {lunch_vol*100:5.2f}%\n"
        stats_text += f"  Close: {close_vol*100:5.2f}%\n"
        
        if lunch_vol > 0:
            u_score = ((open_vol + close_vol) / 2) / lunch_vol
            stats_text += f"  U-Score: {u_score:.2f}\n"
        stats_text += "\n"
    
    # Add overall statistics
    if combined_intervals[0] and combined_intervals[6] and combined_intervals[12]:
        overall_open = np.mean(combined_intervals[0])
        overall_lunch = np.mean(combined_intervals[6])
        overall_close = np.mean(combined_intervals[12])
        
        stats_text += "Overall Average:\n"
        stats_text += f"  Open:  {overall_open*100:5.2f}%\n"
        stats_text += f"  Lunch: {overall_lunch*100:5.2f}%\n"
        stats_text += f"  Close: {overall_close*100:5.2f}%\n"
        
        if overall_lunch > 0:
            overall_u = ((overall_open + overall_close) / 2) / overall_lunch
            stats_text += f"  U-Score: {overall_u:.2f}\n"
            stats_text += f"\nOpen vs Lunch: {(overall_open/overall_lunch-1)*100:.1f}% higher"
            stats_text += f"\nClose vs Lunch: {(overall_close/overall_lunch-1)*100:.1f}% higher"
    
    ax.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("debug_plots_fixed")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f'intraday_volatility_corrected_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    
    print(f"\nPlot saved to: {filename}")
    
    return all_data

def main():
    """
    Main execution with 3 different assets
    """
    print("="*80)
    print("CORRECTED INTRADAY VOLATILITY ANALYSIS")
    print("="*80)
    
    # Analyze 3 different types of assets
    # Tech stocks, Financial, Consumer
    symbols = ['AAPL', 'JPM', 'WMT']  
    
    print(f"\nAnalyzing: {', '.join(symbols)}")
    print("Using 60 days of recent data")
    print("Calculating volatility for 30-minute intervals")
    print("-"*80)
    
    # Generate plots
    all_data = create_comprehensive_plots(symbols, sample_days=60)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Print final summary
    for symbol in symbols:
        if symbol in all_data:
            vol_data = all_data[symbol]
            means = [np.mean(vol_data[i]) if vol_data[i] else np.nan for i in range(13)]
            
            open_vol = means[0] if not np.isnan(means[0]) else 0
            lunch_vol = means[6] if not np.isnan(means[6]) else 0
            close_vol = means[12] if not np.isnan(means[12]) else 0
            
            print(f"\n{symbol} Results:")
            print(f"  Opening (9:30-10:00):  {open_vol*100:.2f}%")
            print(f"  Lunch (12:30-13:00):   {lunch_vol*100:.2f}%")
            print(f"  Closing (15:30-16:00): {close_vol*100:.2f}%")
            
            if lunch_vol > 0:
                print(f"  Open vs Lunch: {(open_vol/lunch_vol - 1)*100:.1f}% higher")
                print(f"  U-Score: {((open_vol + close_vol) / 2) / lunch_vol:.2f}")

if __name__ == "__main__":
    main()