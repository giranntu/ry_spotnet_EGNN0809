#!/usr/bin/env python3
"""
Data Validation and Repair Script for Step 1 - Polygon Data Download
Identifies issues and automatically repairs them by re-fetching problematic days
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from tqdm import tqdm
import warnings
import requests
import time
import sys
warnings.filterwarnings('ignore')

class DataValidatorAndRepairer:
    """Validator and repairer for downloaded 1-minute OHLCV data"""
    
    def __init__(self):
        self.data_dir = Path("rawdata/by_comp")
        self.report_file = Path("data_quality_report.json")
        self.repair_log_file = Path("data_repair_log.json")
        
        # Polygon API configuration
        self.API_KEY = "W4WJfLUGGO5vdylrgw9RKKyD9LANdWRu"
        self.BASE_URL = "https://api.polygon.io"
        
        # Expected parameters
        self.expected_bars_per_day = 390  # 9:30 AM to 4:00 PM = 6.5 hours * 60 minutes
        self.start_date = '2019-01-01'
        self.end_date = '2025-07-30'
        
        # Load symbol list
        self.symbols = self.load_symbols()
        
        # Get NYSE trading calendar
        self.nyse = mcal.get_calendar('NYSE')
        self.trading_dates = self.get_trading_dates()
        
        # Track repairs
        self.repair_log = []
        
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
    
    def fetch_day_data_with_retry(self, symbol, date, max_retries=10):
        """Fetch 1-minute data for a specific day with retry logic"""
        for attempt in range(max_retries):
            try:
                url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
                params = {
                    'apiKey': self.API_KEY,
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50000
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff capped at 60s
                    print(f"    Rate limited, waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    if attempt == max_retries - 1:
                        print(f"    API error {response.status_code} for {symbol} {date}")
                    return pd.DataFrame()
                
                data = response.json()
                
                if data.get('status') != 'OK' or not data.get('results'):
                    # No data available for this date
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(data['results'])
                
                # Convert timestamp to Eastern Time
                df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)
                
                # Map to OHLCV format
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                df['ticker'] = symbol
                
                # Filter to regular trading hours only (9:30 AM - 4:00 PM)
                market_open = pd.Timestamp(f"{date} 09:30:00")
                market_close = pd.Timestamp(f"{date} 16:00:00")
                df = df[(df['datetime'] >= market_open) & (df['datetime'] < market_close)]
                
                # Select and sort
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
                df = df.sort_values('datetime').reset_index(drop=True)
                
                return df
                
            except requests.exceptions.Timeout:
                wait_time = 2
                if attempt < max_retries - 1:
                    print(f"    Timeout (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                wait_time = 2
                if attempt < max_retries - 1:
                    print(f"    Connection error (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    Error: {e} (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)
        
        return pd.DataFrame()
    
    def create_aligned_intervals(self, df, date):
        """Create aligned 1-minute intervals for regular trading hours"""
        if df.empty:
            return pd.DataFrame()
        
        # Create complete 1-minute index for regular trading hours (9:30-4:00)
        market_open = pd.Timestamp(f"{date} 09:30:00")
        market_close = pd.Timestamp(f"{date} 15:59:00")  # Last minute is 15:59
        complete_index = pd.date_range(start=market_open, end=market_close, freq='1min')
        
        # Reindex with forward fill for missing minutes during trading hours
        df_indexed = df.set_index('datetime')
        df_aligned = df_indexed.reindex(complete_index)
        
        # Forward fill prices (standard practice for missing minutes)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_aligned[col] = df_aligned[col].ffill()
        
        # Volume should be 0 for missing bars, not forward filled
        df_aligned['volume'] = df_aligned['volume'].fillna(0)
        df_aligned['ticker'] = df['ticker'].iloc[0] if not df.empty else None
        
        # Reset index
        df_aligned = df_aligned.reset_index().rename(columns={'index': 'datetime'})
        
        return df_aligned[['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
    
    def identify_problematic_days(self, symbol):
        """Identify days with issues in the data"""
        filepath = self.data_dir / f"{symbol}_201901_202507.csv"
        
        if not filepath.exists():
            return None, []
        
        try:
            # Load the data
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            
            problematic_days = []
            
            # 1. Find days with NaN values
            nan_mask = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
            if nan_mask.any():
                nan_dates = df[nan_mask]['date'].unique()
                for date in nan_dates:
                    problematic_days.append({
                        'date': str(date),
                        'issue': 'NaN values',
                        'count': int(nan_mask[df['date'] == date].sum())
                    })
            
            # 2. Find days with wrong number of bars
            bars_per_day = df.groupby('date').size()
            wrong_bars = bars_per_day[bars_per_day != self.expected_bars_per_day]
            for date, count in wrong_bars.items():
                problematic_days.append({
                    'date': str(date),
                    'issue': f'Wrong bar count ({count} instead of {self.expected_bars_per_day})',
                    'count': int(count)
                })
            
            # 3. Find days with zero prices
            zero_mask = (df[['open', 'high', 'low', 'close']] == 0).any(axis=1)
            if zero_mask.any():
                zero_dates = df[zero_mask]['date'].unique()
                for date in zero_dates:
                    if str(date) not in [d['date'] for d in problematic_days]:
                        problematic_days.append({
                            'date': str(date),
                            'issue': 'Zero prices',
                            'count': int(zero_mask[df['date'] == date].sum())
                        })
            
            # 4. Find missing days
            actual_dates = set(df['date'].unique())
            expected_dates = set()
            
            for date_str in self.trading_dates:
                date = pd.Timestamp(date_str).date()
                # Skip dates based on symbol constraints
                if symbol == 'AMZN' and date < pd.Timestamp('2019-01-02').date():
                    continue
                if symbol == 'BA' and date > pd.Timestamp('2025-07-30').date():
                    continue
                if date > pd.Timestamp('2025-07-30').date():
                    continue
                    
                min_date = df['date'].min()
                max_date = df['date'].max()
                if min_date <= date <= max_date:
                    expected_dates.add(date)
            
            missing_dates = expected_dates - actual_dates
            for date in missing_dates:
                problematic_days.append({
                    'date': str(date),
                    'issue': 'Missing day',
                    'count': 0
                })
            
            return df, problematic_days
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None, []
    
    def repair_symbol_data(self, symbol, problematic_days, existing_df):
        """Repair problematic days by re-fetching data"""
        if not problematic_days:
            return 0, 0
        
        print(f"\n  🔧 Repairing {symbol}: {len(problematic_days)} problematic days found")
        
        repaired_count = 0
        failed_count = 0
        
        # Group problematic days by date for efficient processing
        dates_to_repair = list(set([d['date'] for d in problematic_days]))
        
        for date_str in tqdm(dates_to_repair, desc=f"  Repairing {symbol}", leave=False):
            # Re-fetch data for this day
            print(f"    Re-fetching {date_str}...")
            new_data = self.fetch_day_data_with_retry(symbol, date_str)
            
            if not new_data.empty:
                # Create aligned intervals
                aligned_data = self.create_aligned_intervals(new_data, date_str)
                
                if not aligned_data.empty and len(aligned_data) == self.expected_bars_per_day:
                    # Remove old data for this date
                    date_obj = pd.Timestamp(date_str).date()
                    existing_df = existing_df[existing_df['date'] != date_obj]
                    
                    # Add new data
                    aligned_data['date'] = aligned_data['datetime'].dt.date
                    existing_df = pd.concat([existing_df, aligned_data], ignore_index=True)
                    
                    repaired_count += 1
                    self.repair_log.append({
                        'symbol': symbol,
                        'date': date_str,
                        'status': 'repaired',
                        'bars': len(aligned_data)
                    })
                    print(f"    ✅ Repaired {date_str}: {len(aligned_data)} bars")
                else:
                    failed_count += 1
                    self.repair_log.append({
                        'symbol': symbol,
                        'date': date_str,
                        'status': 'failed',
                        'reason': 'Incomplete data after fetch'
                    })
                    print(f"    ❌ Failed to repair {date_str}: Incomplete data")
            else:
                failed_count += 1
                self.repair_log.append({
                    'symbol': symbol,
                    'date': date_str,
                    'status': 'failed',
                    'reason': 'No data from API'
                })
                print(f"    ❌ Failed to repair {date_str}: No data available")
            
            # Rate limiting
            time.sleep(0.1)  # 100ms between requests
        
        # Save the repaired data if any repairs were successful
        if repaired_count > 0:
            # Sort by datetime and save
            existing_df = existing_df.sort_values('datetime').reset_index(drop=True)
            
            # Keep only required columns
            columns_to_keep = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']
            existing_df = existing_df[columns_to_keep]
            
            # Save to file
            filepath = self.data_dir / f"{symbol}_201901_202507.csv"
            existing_df.to_csv(filepath, index=False)
            print(f"  💾 Saved repaired data for {symbol}")
        
        return repaired_count, failed_count
    
    def validate_single_file(self, symbol, repair=False):
        """Validate and optionally repair a single symbol's data file"""
        filepath = self.data_dir / f"{symbol}_201901_202507.csv"
        
        if not filepath.exists():
            return {
                'symbol': symbol,
                'status': 'MISSING',
                'file_exists': False,
                'issues': ['File not found'],
                'repaired': False
            }
        
        # Identify problematic days
        existing_df, problematic_days = self.identify_problematic_days(symbol)
        
        if existing_df is None:
            return {
                'symbol': symbol,
                'status': 'ERROR',
                'file_exists': True,
                'issues': ['Error reading file'],
                'repaired': False
            }
        
        # Repair if requested and issues found
        repaired_count = 0
        failed_count = 0
        if repair and problematic_days:
            repaired_count, failed_count = self.repair_symbol_data(symbol, problematic_days, existing_df)
            
            # Re-validate after repair
            existing_df, problematic_days = self.identify_problematic_days(symbol)
        
        # Final validation
        result = {
            'symbol': symbol,
            'file_exists': True,
            'file_size_mb': round(filepath.stat().st_size / (1024 * 1024), 2),
            'total_records': len(existing_df),
            'date_range': f"{existing_df['date'].min()} to {existing_df['date'].max()}",
            'issues': [],
            'problematic_days': len(problematic_days),
            'repaired': repaired_count > 0,
            'repairs_successful': repaired_count,
            'repairs_failed': failed_count
        }
        
        # Check for remaining issues
        nan_counts = existing_df[['open', 'high', 'low', 'close']].isna().sum()
        if nan_counts.any():
            result['issues'].append(f"NaN values: {nan_counts.to_dict()}")
        
        # Set status
        if not result['issues'] and result['problematic_days'] == 0:
            result['status'] = 'PASS'
        elif result['repaired'] and result['problematic_days'] < 5:
            result['status'] = 'REPAIRED'
        else:
            result['status'] = 'FAIL'
        
        return result
    
    def run_validation_and_repair(self, repair=True, batch_size=5):
        """Run validation and optional repair on all files"""
        print("=" * 80)
        print("🔍 DATA VALIDATION AND REPAIR FOR POLYGON DOWNLOADS")
        print("=" * 80)
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Symbols to check: {len(self.symbols)}")
        print(f"🔧 Repair mode: {'ON' if repair else 'OFF'}")
        print(f"📦 Batch size: {batch_size} symbols at a time")
        print("=" * 80)
        
        results = []
        
        # Process in batches to avoid timeout
        for batch_start in range(0, len(self.symbols), batch_size):
            batch_end = min(batch_start + batch_size, len(self.symbols))
            batch_symbols = self.symbols[batch_start:batch_end]
            
            print(f"\n📦 Processing batch {batch_start//batch_size + 1}/{(len(self.symbols) + batch_size - 1)//batch_size}")
            print(f"   Symbols: {', '.join(batch_symbols)}")
            
            for symbol in tqdm(batch_symbols, desc="Processing batch"):
                result = self.validate_single_file(symbol, repair=repair)
                results.append(result)
            
            # Save intermediate progress
            if repair and self.repair_log:
                with open(self.repair_log_file, 'w') as f:
                    json.dump(self.repair_log, f, indent=2, default=str)
                print(f"   💾 Saved intermediate repair log with {len(self.repair_log)} entries")
        
        # Generate summary
        summary = {
            'total_symbols': len(self.symbols),
            'passed': sum(1 for r in results if r['status'] == 'PASS'),
            'repaired': sum(1 for r in results if r['status'] == 'REPAIRED'),
            'failed': sum(1 for r in results if r['status'] == 'FAIL'),
            'missing': sum(1 for r in results if r['status'] == 'MISSING'),
            'total_repairs': sum(r.get('repairs_successful', 0) for r in results),
            'total_failures': sum(r.get('repairs_failed', 0) for r in results)
        }
        
        # Save reports
        report = {
            'timestamp': datetime.now().isoformat(),
            'repair_mode': repair,
            'summary': summary,
            'symbols': {r['symbol']: r for r in results}
        }
        
        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.repair_log:
            with open(self.repair_log_file, 'w') as f:
                json.dump(self.repair_log, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 VALIDATION AND REPAIR SUMMARY")
        print("=" * 80)
        print(f"✅ PASS: {summary['passed']}/{summary['total_symbols']}")
        print(f"🔧 REPAIRED: {summary['repaired']}/{summary['total_symbols']}")
        print(f"❌ FAIL: {summary['failed']}/{summary['total_symbols']}")
        print(f"📂 MISSING: {summary['missing']}/{summary['total_symbols']}")
        
        if repair:
            print(f"\n🔧 Repair Statistics:")
            print(f"  Successful repairs: {summary['total_repairs']}")
            print(f"  Failed repairs: {summary['total_failures']}")
        
        print(f"\n📊 Full report saved to: {self.report_file}")
        if self.repair_log:
            print(f"🔧 Repair log saved to: {self.repair_log_file}")
        
        # Return success if all passed or repaired
        return summary['failed'] == 0 and summary['missing'] == 0


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and repair Polygon data')
    parser.add_argument('--no-repair', action='store_true', 
                       help='Only validate without repairing')
    args = parser.parse_args()
    
    validator = DataValidatorAndRepairer()
    success = validator.run_validation_and_repair(repair=not args.no_repair)
    
    if success:
        print("\n✅ All data files are valid or have been repaired! Ready for Step 2.")
        return 0
    else:
        print("\n⚠️ Some files still have issues. Review the report and consider manual intervention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())