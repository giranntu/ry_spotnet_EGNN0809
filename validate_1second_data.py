#!/usr/bin/env python3
"""
Comprehensive validation script for 1-second data
Checks completeness, continuity, and data quality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def validate_1second_data(filepath: str):
    """
    Comprehensive validation of 1-second data
    """
    print("="*80)
    print("1-SECOND DATA VALIDATION REPORT")
    print("="*80)
    print(f"File: {filepath}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(filepath, compression='gzip', parse_dates=['datetime'])
    
    # Basic statistics
    print("📊 BASIC STATISTICS:")
    print(f"  Total records: {len(df):,}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Unique dates: {df['datetime'].dt.date.nunique()}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for unique trading days
    unique_days = df['datetime'].dt.date.unique()
    print(f"  Trading days: {', '.join([str(d) for d in sorted(unique_days)])}")
    
    print("\n" + "="*80)
    print("📅 PER-DAY ANALYSIS:")
    print("="*80)
    
    all_issues = []
    
    for day in sorted(unique_days):
        day_df = df[df['datetime'].dt.date == day].copy()
        day_str = str(day)
        
        print(f"\n📆 Date: {day_str}")
        print("-"*40)
        
        # Expected trading hours (9:30 AM - 4:00 PM ET)
        expected_start = pd.Timestamp(f"{day} 09:30:00")
        expected_end = pd.Timestamp(f"{day} 15:59:59")
        
        # Actual time range
        actual_start = day_df['datetime'].min()
        actual_end = day_df['datetime'].max()
        
        print(f"  Expected time range: {expected_start} to {expected_end}")
        print(f"  Actual time range:   {actual_start} to {actual_end}")
        
        # Check record count
        expected_seconds = 23400  # 6.5 hours * 60 minutes * 60 seconds
        actual_seconds = len(day_df)
        
        print(f"  Expected records: {expected_seconds:,}")
        print(f"  Actual records:   {actual_seconds:,}")
        
        if actual_seconds != expected_seconds:
            diff = actual_seconds - expected_seconds
            print(f"  ⚠️ MISMATCH: {diff:+,} records ({diff/expected_seconds*100:+.2f}%)")
            all_issues.append(f"{day_str}: Record count mismatch ({diff:+,})")
        else:
            print(f"  ✅ Record count matches exactly!")
        
        # Check for time continuity (should have exactly 1 second between consecutive records)
        day_df['time_diff'] = day_df['datetime'].diff()
        
        # Find gaps (where time difference is not 1 second)
        gaps = day_df[day_df['time_diff'] > pd.Timedelta(seconds=1)]
        if len(gaps) > 0:
            print(f"\n  ⚠️ TIME GAPS FOUND: {len(gaps)} gaps")
            for idx, gap in gaps.head(5).iterrows():
                prev_time = day_df.loc[idx-1, 'datetime'] if idx > 0 else None
                print(f"    Gap at {gap['datetime']}: {gap['time_diff'].total_seconds():.0f} seconds")
                if prev_time:
                    print(f"      Previous: {prev_time}")
            if len(gaps) > 5:
                print(f"    ... and {len(gaps)-5} more gaps")
            all_issues.append(f"{day_str}: {len(gaps)} time gaps")
        
        # Check for duplicates
        duplicates = day_df[day_df['time_diff'] == pd.Timedelta(seconds=0)]
        if len(duplicates) > 0:
            print(f"\n  ⚠️ DUPLICATE TIMESTAMPS: {len(duplicates)} duplicates")
            for idx, dup in duplicates.head(5).iterrows():
                print(f"    Duplicate at {dup['datetime']}")
            all_issues.append(f"{day_str}: {len(duplicates)} duplicate timestamps")
        
        # Check for missing seconds
        full_range = pd.date_range(start=expected_start, end=expected_end, freq='1s')
        actual_times = set(day_df['datetime'])
        expected_times = set(full_range)
        missing_times = expected_times - actual_times
        
        if missing_times:
            print(f"\n  ⚠️ MISSING TIMESTAMPS: {len(missing_times)} seconds missing")
            missing_sorted = sorted(list(missing_times))[:10]
            for mt in missing_sorted[:5]:
                print(f"    Missing: {mt}")
            if len(missing_sorted) > 5:
                print(f"    ... and {len(missing_times)-5} more")
            all_issues.append(f"{day_str}: {len(missing_times)} missing seconds")
        else:
            print(f"  ✅ No missing timestamps!")
        
        # Data quality checks
        print("\n  📈 DATA QUALITY:")
        
        # Check for NaN values
        nan_cols = day_df.columns[day_df.isna().any()].tolist()
        if nan_cols:
            print(f"    ⚠️ NaN values in columns: {nan_cols}")
            for col in nan_cols:
                nan_count = day_df[col].isna().sum()
                print(f"      {col}: {nan_count} NaN values")
            all_issues.append(f"{day_str}: NaN values in {nan_cols}")
        else:
            print(f"    ✅ No NaN values")
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (day_df[col] <= 0).any():
                zero_count = (day_df[col] <= 0).sum()
                print(f"    ⚠️ {col}: {zero_count} zero/negative values")
                all_issues.append(f"{day_str}: {zero_count} zero/negative {col} values")
        
        # Check OHLC consistency (High >= Low, High >= Open/Close, Low <= Open/Close)
        ohlc_issues = []
        if (day_df['high'] < day_df['low']).any():
            count = (day_df['high'] < day_df['low']).sum()
            ohlc_issues.append(f"High < Low ({count} records)")
        if (day_df['high'] < day_df['open']).any():
            count = (day_df['high'] < day_df['open']).sum()
            ohlc_issues.append(f"High < Open ({count} records)")
        if (day_df['high'] < day_df['close']).any():
            count = (day_df['high'] < day_df['close']).sum()
            ohlc_issues.append(f"High < Close ({count} records)")
        if (day_df['low'] > day_df['open']).any():
            count = (day_df['low'] > day_df['open']).sum()
            ohlc_issues.append(f"Low > Open ({count} records)")
        if (day_df['low'] > day_df['close']).any():
            count = (day_df['low'] > day_df['close']).sum()
            ohlc_issues.append(f"Low > Close ({count} records)")
        
        if ohlc_issues:
            print(f"    ⚠️ OHLC consistency issues:")
            for issue in ohlc_issues:
                print(f"      - {issue}")
            all_issues.append(f"{day_str}: OHLC consistency issues")
        else:
            print(f"    ✅ OHLC values are consistent")
        
        # Check for zero volume
        zero_vol = (day_df['volume'] == 0).sum()
        if zero_vol > 0:
            print(f"    ℹ️ Zero volume: {zero_vol} records ({zero_vol/len(day_df)*100:.1f}%)")
            # This is actually normal for some seconds
        
        # Price statistics
        print(f"\n  💰 PRICE STATISTICS:")
        print(f"    Open:  Min={day_df['open'].min():.2f}, Max={day_df['open'].max():.2f}, Mean={day_df['open'].mean():.2f}")
        print(f"    High:  Min={day_df['high'].min():.2f}, Max={day_df['high'].max():.2f}, Mean={day_df['high'].mean():.2f}")
        print(f"    Low:   Min={day_df['low'].min():.2f}, Max={day_df['low'].max():.2f}, Mean={day_df['low'].mean():.2f}")
        print(f"    Close: Min={day_df['close'].min():.2f}, Max={day_df['close'].max():.2f}, Mean={day_df['close'].mean():.2f}")
        print(f"    Volume: Total={day_df['volume'].sum():,.0f}, Mean={day_df['volume'].mean():,.0f}")
        
        # Check for extreme price movements (> 1% in 1 second)
        day_df['return'] = day_df['close'].pct_change()
        extreme_moves = day_df[abs(day_df['return']) > 0.01]  # 1% threshold
        if len(extreme_moves) > 0:
            print(f"\n  📊 EXTREME MOVEMENTS (>1% in 1 second): {len(extreme_moves)} occurrences")
            for idx, move in extreme_moves.head(3).iterrows():
                print(f"    {move['datetime']}: {move['return']*100:.2f}% move")
    
    # Overall summary
    print("\n" + "="*80)
    print("📋 OVERALL SUMMARY:")
    print("="*80)
    
    if all_issues:
        print(f"\n⚠️ ISSUES FOUND ({len(all_issues)} total):")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ NO ISSUES FOUND - DATA IS COMPLETE AND VALID!")
    
    # Additional statistics
    print(f"\n📊 AGGREGATE STATISTICS:")
    print(f"  Total records: {len(df):,}")
    print(f"  Expected total: {len(unique_days) * 23400:,}")
    print(f"  Completeness: {len(df) / (len(unique_days) * 23400) * 100:.2f}%")
    
    # Check time spacing distribution
    all_diffs = df['datetime'].diff().dropna()
    time_dist = all_diffs.value_counts()
    
    print(f"\n⏱️ TIME SPACING DISTRIBUTION:")
    for td, count in time_dist.head(5).items():
        seconds = td.total_seconds()
        print(f"  {seconds:.0f} second(s): {count:,} occurrences ({count/len(all_diffs)*100:.1f}%)")
    
    # Check if data is properly sorted
    is_sorted = df['datetime'].is_monotonic_increasing
    print(f"\n🔄 TIME ORDERING:")
    if is_sorted:
        print(f"  ✅ Data is properly sorted by time")
    else:
        print(f"  ⚠️ Data is NOT properly sorted by time!")
        all_issues.append("Data not sorted by time")
    
    # Final verdict
    print("\n" + "="*80)
    print("🏁 FINAL VERDICT:")
    print("="*80)
    
    if not all_issues:
        print("✅ DATA PASSES ALL VALIDATION CHECKS!")
        print("The 1-second data is complete, continuous, and ready for analysis.")
    else:
        print(f"⚠️ DATA HAS {len(all_issues)} ISSUES THAT MAY NEED ATTENTION")
        print("Review the issues above to determine if they impact your analysis.")
    
    return df, all_issues


if __name__ == "__main__":
    # Validate the AAPL 2-day test data
    filepath = "rawdata/by_comp_1second/AAPL_201901_202507_1second.csv.gz"
    
    print("Starting validation of 1-second data...")
    df, issues = validate_1second_data(filepath)
    
    print("\n" + "="*80)
    print("Validation complete!")
    
    # Save validation report
    report_file = "1second_validation_report.txt"
    with open(report_file, 'w') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        validate_1second_data(filepath)
        sys.stdout = original_stdout
    
    print(f"Full report saved to: {report_file}")