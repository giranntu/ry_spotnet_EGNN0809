#!/usr/bin/env python3
"""
Data Validation Script for Step 1 - Polygon Data Download
Checks for data completeness, missing days, and quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """Comprehensive validator for downloaded 1-minute OHLCV data"""
    
    def __init__(self):
        self.data_dir = Path("rawdata/by_comp")
        self.report_file = Path("data_quality_report.json")
        self.issues_file = Path("data_issues_detailed.csv")
        
        # Expected parameters
        self.expected_bars_per_day = 390  # 9:30 AM to 4:00 PM = 6.5 hours * 60 minutes
        self.start_date = '2019-01-01'
        self.end_date = '2025-07-30'
        
        # Load symbol list
        self.symbols = self.load_symbols()
        
        # Get NYSE trading calendar
        self.nyse = mcal.get_calendar('NYSE')
        self.trading_dates = self.get_trading_dates()
        
    def load_symbols(self):
        """Load DOW30 symbols from config"""
        try:
            import yaml
            with open('config/dow30_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                return config['dow30_symbols']
        except:
            # Fallback to hardcoded list
            return [
                'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
                'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
                'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
            ]
    
    def get_trading_dates(self):
        """Get all NYSE trading dates in the period"""
        trading_days = self.nyse.valid_days(start_date=self.start_date, end_date=self.end_date)
        return [date.strftime('%Y-%m-%d') for date in trading_days.date]
    
    def check_file_exists(self, symbol):
        """Check if file exists for a symbol"""
        filepath = self.data_dir / f"{symbol}_201901_202507.csv"
        return filepath.exists(), filepath
    
    def validate_single_file(self, symbol):
        """Validate a single symbol's data file"""
        exists, filepath = self.check_file_exists(symbol)
        
        if not exists:
            return {
                'symbol': symbol,
                'status': 'MISSING',
                'file_exists': False,
                'issues': ['File not found']
            }
        
        try:
            # Load the data
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            
            # Initialize validation result
            result = {
                'symbol': symbol,
                'status': 'OK',
                'file_exists': True,
                'file_size_mb': round(filepath.stat().st_size / (1024 * 1024), 2),
                'total_records': len(df),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'issues': [],
                'warnings': [],
                'stats': {}
            }
            
            # 1. Check date range
            actual_start = df['date'].min()
            actual_end = df['date'].max()
            
            # Special handling for AMZN (which replaced DOW)
            if symbol == 'AMZN':
                expected_start = pd.Timestamp('2019-01-02').date()  # First trading day of 2019
                if actual_start > expected_start:
                    result['issues'].append(f"Late start: {actual_start} (expected {expected_start})")
            
            # Special handling for BA
            if symbol == 'BA':
                expected_end = pd.Timestamp('2025-07-30').date()
                if actual_end != expected_end:
                    result['issues'].append(f"Unexpected end date: {actual_end} (expected {expected_end})")
            
            # 2. Check for missing trading days
            actual_dates = set(df['date'].unique())
            expected_dates = set()
            
            for date_str in self.trading_dates:
                date = pd.Timestamp(date_str).date()
                # Skip dates before AMZN should have data
                if symbol == 'AMZN' and date < pd.Timestamp('2019-01-02').date():
                    continue
                # Skip dates after BA constraint
                if symbol == 'BA' and date > pd.Timestamp('2025-07-30').date():
                    continue
                # Skip all symbols for dates beyond BA constraint
                if date > pd.Timestamp('2025-07-30').date():
                    continue
                    
                if actual_start <= date <= actual_end:
                    expected_dates.add(date)
            
            missing_dates = expected_dates - actual_dates
            if missing_dates:
                result['issues'].append(f"Missing {len(missing_dates)} trading days")
                result['missing_dates'] = sorted([str(d) for d in missing_dates])[:10]  # Show first 10
            
            # 3. Check bars per day
            bars_per_day = df.groupby('date').size()
            days_with_wrong_bars = bars_per_day[bars_per_day != self.expected_bars_per_day]
            
            if len(days_with_wrong_bars) > 0:
                result['warnings'].append(f"{len(days_with_wrong_bars)} days with != {self.expected_bars_per_day} bars")
                result['wrong_bar_days'] = {
                    str(date): count for date, count in days_with_wrong_bars.head(10).items()
                }
            
            # 4. Check for NaN values
            nan_counts = df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
            if nan_counts.any():
                result['issues'].append(f"NaN values found: {nan_counts.to_dict()}")
            
            # 5. Check for zero prices (invalid)
            zero_prices = (df[['open', 'high', 'low', 'close']] == 0).any(axis=1).sum()
            if zero_prices > 0:
                result['issues'].append(f"{zero_prices} records with zero prices")
            
            # 6. Check for negative prices (impossible)
            negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()
            if negative_prices > 0:
                result['issues'].append(f"{negative_prices} records with negative prices")
            
            # 7. Check OHLC consistency (High >= Low, High >= Open/Close, Low <= Open/Close)
            inconsistent_ohlc = 0
            inconsistent_ohlc += (df['high'] < df['low']).sum()
            inconsistent_ohlc += (df['high'] < df['open']).sum()
            inconsistent_ohlc += (df['high'] < df['close']).sum()
            inconsistent_ohlc += (df['low'] > df['open']).sum()
            inconsistent_ohlc += (df['low'] > df['close']).sum()
            
            if inconsistent_ohlc > 0:
                result['issues'].append(f"{inconsistent_ohlc} OHLC consistency violations")
            
            # 8. Check for duplicate timestamps
            duplicates = df.duplicated(subset=['datetime']).sum()
            if duplicates > 0:
                result['issues'].append(f"{duplicates} duplicate timestamps")
            
            # 9. Statistics
            result['stats'] = {
                'total_days': df['date'].nunique(),
                'expected_days': len(expected_dates),
                'avg_bars_per_day': round(bars_per_day.mean(), 2),
                'min_bars_per_day': int(bars_per_day.min()),
                'max_bars_per_day': int(bars_per_day.max()),
                'avg_volume': round(df['volume'].mean(), 0),
                'avg_price': round(df['close'].mean(), 2),
                'price_range': f"${df['close'].min():.2f} - ${df['close'].max():.2f}"
            }
            
            # Set overall status
            if result['issues']:
                result['status'] = 'FAIL'
            elif result['warnings']:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'PASS'
            
            return result
            
        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'ERROR',
                'file_exists': True,
                'issues': [f"Error reading file: {str(e)}"]
            }
    
    def generate_summary(self, results):
        """Generate summary statistics"""
        summary = {
            'total_symbols': len(self.symbols),
            'files_found': sum(1 for r in results if r['file_exists']),
            'files_missing': sum(1 for r in results if not r['file_exists']),
            'passed': sum(1 for r in results if r['status'] == 'PASS'),
            'warnings': sum(1 for r in results if r['status'] == 'WARNING'),
            'failed': sum(1 for r in results if r['status'] == 'FAIL'),
            'errors': sum(1 for r in results if r['status'] == 'ERROR'),
            'total_size_mb': sum(r.get('file_size_mb', 0) for r in results),
            'total_records': sum(r.get('total_records', 0) for r in results)
        }
        return summary
    
    def save_detailed_issues(self, results):
        """Save detailed issues to CSV for easy review"""
        issues_data = []
        
        for result in results:
            if result['status'] != 'PASS':
                for issue in result.get('issues', []):
                    issues_data.append({
                        'symbol': result['symbol'],
                        'status': result['status'],
                        'issue_type': 'ERROR',
                        'description': issue
                    })
                for warning in result.get('warnings', []):
                    issues_data.append({
                        'symbol': result['symbol'],
                        'status': result['status'],
                        'issue_type': 'WARNING',
                        'description': warning
                    })
        
        if issues_data:
            df_issues = pd.DataFrame(issues_data)
            df_issues.to_csv(self.issues_file, index=False)
            print(f"💾 Detailed issues saved to: {self.issues_file}")
    
    def run_validation(self):
        """Run complete validation on all files"""
        print("=" * 80)
        print("🔍 DATA VALIDATION FOR POLYGON DOWNLOADS")
        print("=" * 80)
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Symbols to check: {len(self.symbols)}")
        print(f"📅 Expected period: {self.start_date} to {self.end_date}")
        print(f"📈 Expected bars per day: {self.expected_bars_per_day}")
        print("=" * 80)
        
        # Validate each symbol
        results = []
        for symbol in tqdm(self.symbols, desc="Validating files"):
            result = self.validate_single_file(symbol)
            results.append(result)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        # Create final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'data_directory': str(self.data_dir),
                'expected_period': f"{self.start_date} to {self.end_date}",
                'expected_bars_per_day': self.expected_bars_per_day,
                'total_symbols': len(self.symbols)
            },
            'summary': summary,
            'symbols': {r['symbol']: r for r in results}
        }
        
        # Save report
        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed issues
        self.save_detailed_issues(results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        print(f"✅ PASS: {summary['passed']}/{summary['total_symbols']}")
        print(f"⚠️  WARNING: {summary['warnings']}/{summary['total_symbols']}")
        print(f"❌ FAIL: {summary['failed']}/{summary['total_symbols']}")
        print(f"🚫 ERROR: {summary['errors']}/{summary['total_symbols']}")
        print(f"📂 Missing files: {summary['files_missing']}/{summary['total_symbols']}")
        print(f"💾 Total size: {summary['total_size_mb']:.1f} MB")
        print(f"📈 Total records: {summary['total_records']:,}")
        print("=" * 80)
        
        # Print symbols with issues
        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\n🚨 SYMBOLS WITH CRITICAL ISSUES:")
            for result in results:
                if result['status'] in ['FAIL', 'ERROR']:
                    print(f"  {result['symbol']}: {', '.join(result.get('issues', []))}")
        
        if summary['warnings'] > 0:
            print("\n⚠️  SYMBOLS WITH WARNINGS:")
            for result in results:
                if result['status'] == 'WARNING':
                    print(f"  {result['symbol']}: {', '.join(result.get('warnings', []))}")
        
        print(f"\n📊 Full report saved to: {self.report_file}")
        print(f"📋 Issues details saved to: {self.issues_file}")
        
        # Return success based on critical issues
        return summary['failed'] == 0 and summary['errors'] == 0 and summary['files_missing'] == 0


def main():
    """Main entry point"""
    validator = DataValidator()
    success = validator.run_validation()
    
    if success:
        print("\n✅ All data files passed validation! Ready for Step 2.")
        return 0
    else:
        print("\n❌ Some files have issues. Please review the reports and consider re-downloading.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())