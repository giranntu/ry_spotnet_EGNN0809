#!/usr/bin/env python3
"""
Quick validation script to check data quality without repair
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def quick_validate():
    """Quick validation of all downloaded files"""
    data_dir = Path("rawdata/by_comp")
    
    symbols = [
        'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
    ]
    
    print("=" * 80)
    print("🔍 QUICK DATA VALIDATION CHECK")
    print("=" * 80)
    
    results = {}
    total_nan_symbols = 0
    total_nan_count = 0
    
    for symbol in symbols:
        filepath = data_dir / f"{symbol}_201901_202507.csv"
        
        if not filepath.exists():
            results[symbol] = "MISSING"
            print(f"❌ {symbol}: File not found")
            continue
        
        try:
            # Quick check for NaN values
            df = pd.read_csv(filepath)
            nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
            
            if nan_count > 0:
                results[symbol] = f"NaN: {int(nan_count)}"
                total_nan_symbols += 1
                total_nan_count += int(nan_count)
                print(f"⚠️  {symbol}: {int(nan_count)} NaN values")
            else:
                results[symbol] = "OK"
                print(f"✅ {symbol}: Clean")
                
        except Exception as e:
            results[symbol] = f"ERROR: {e}"
            print(f"❌ {symbol}: Error reading file")
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total symbols: {len(symbols)}")
    print(f"Clean files: {sum(1 for v in results.values() if v == 'OK')}")
    print(f"Files with NaN: {total_nan_symbols}")
    print(f"Total NaN values: {total_nan_count}")
    print(f"Missing files: {sum(1 for v in results.values() if v == 'MISSING')}")
    
    # Save results
    with open("quick_validation_report.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total': len(symbols),
                'clean': sum(1 for v in results.values() if v == 'OK'),
                'with_nan': total_nan_symbols,
                'missing': sum(1 for v in results.values() if v == 'MISSING'),
                'total_nan_values': total_nan_count
            }
        }, f, indent=2)
    
    print(f"\n📊 Report saved to: quick_validation_report.json")
    
    return results

if __name__ == "__main__":
    quick_validate()