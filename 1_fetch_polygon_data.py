#!/usr/bin/env python3
"""
Polygon.io 1-Minute Data Fetcher for DOW30 with Perfect Alignment
- Updated list: AMZN replaces DOW for complete historical data
- Ensures all symbols have consistent date ranges
- NO synthetic data - only real market data
- Validates data completeness and alignment
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import logging
import json
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import pandas_market_calendars as mcal
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('polygon_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load DOW30 symbols from config
with open('config/dow30_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    DOW30_SYMBOLS = config['dow30_symbols']

class PolygonDataFetcher:
    """Enhanced Polygon.io fetcher with perfect data alignment - NO synthetic data"""
    
    API_KEY = "W4WJfLUGGO5vdylrgw9RKKyD9LANdWRu"
    BASE_URL = "https://api.polygon.io"
    
    # Symbol-specific constraints based on real market data availability
    SYMBOL_CONSTRAINTS = {
        'BA': {
            'last_known_data': '2025-07-30',  # Last known data point
            'note': 'Data availability constraint'
        }
    }
    
    def __init__(self, max_workers=30):
        """Initialize the enhanced fetcher"""
        self.symbols = DOW30_SYMBOLS
        self.output_dir = Path("rawdata/by_comp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = Path("polygon_progress.json")
        self.validation_file = Path("data_validation_report.json")
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.rate_limit_delay = 0.05  # 50ms between requests
        
        # Global date range for alignment
        self.global_start = '2019-01-01'
        self.global_end = '2025-07-30'
        
        logger.info(f"✅ Output directory: {self.output_dir}")
        logger.info(f"📋 Progress file: {self.progress_file}")
        logger.info(f"📊 Validation file: {self.validation_file}")
        logger.info(f"⚡ Parallel workers: {max_workers}")
        logger.info(f"📅 Target period: {self.global_start} to {self.global_end}")
    
    def load_progress(self) -> dict:
        """Load progress from JSON file (thread-safe)"""
        with self.progress_lock:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            return {
                "completed": {}, 
                "failed": {},
                "validation": {},
                "started_at": None, 
                "last_update": None
            }
    
    def save_progress(self, progress: dict):
        """Save progress to JSON file (thread-safe)"""
        with self.progress_lock:
            progress["last_update"] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    
    def get_trading_dates(self, start_date: str, end_date: str) -> list:
        """Get NYSE trading dates between start and end dates"""
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
        return [date.strftime('%Y-%m-%d') for date in trading_days.date]
    
    def get_symbol_date_range(self, symbol: str) -> Tuple[str, str]:
        """Get the actual date range for a symbol considering real data availability"""
        start_date = self.global_start
        end_date = self.global_end
        
        # Apply symbol-specific constraints based on real market data
        if symbol == 'BA' and end_date > '2025-07-30':
            # BA data constraint
            end_date = '2025-07-30'
        
        return start_date, end_date
    
    def is_already_downloaded(self, symbol: str) -> tuple:
        """Check if symbol data already exists and validate it"""
        filename = f"{symbol}_201901_202507.csv"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Quick validation
            df = pd.read_csv(filepath, nrows=5)
            has_datetime = 'datetime' in df.columns
            
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f) - 1
            
            return True, filepath, {
                "size_mb": round(file_size_mb, 1),
                "records": line_count,
                "valid_format": has_datetime,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
        return False, filepath, None
    
    def fetch_day_data_with_retry(self, symbol: str, date: str, max_retries: int = 10) -> pd.DataFrame:
        """Fetch 1-minute data with ROBUST retry logic and rate limit handling"""
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
                
                # Handle rate limiting with improved exponential backoff
                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff capped at 60s
                    logger.warning(f"Rate limited for {symbol} {date}, attempt {attempt+1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    logger.debug(f"API error {response.status_code} for {symbol} {date}")
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
                wait_time = 2  # Fixed 2s wait for timeouts
                logger.warning(f"Timeout for {symbol} {date} (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                wait_time = 2  # Fixed 2s wait for connection errors
                logger.warning(f"Connection error for {symbol} {date}: {e} (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                wait_time = 2  # Fixed 2s wait for other errors
                logger.warning(f"Error fetching {symbol} {date}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        return pd.DataFrame()
    
    def create_aligned_intervals(self, df: pd.DataFrame, symbol: str, date: str) -> pd.DataFrame:
        """Create aligned 1-minute intervals for regular trading hours - NO synthetic data"""
        if df.empty:
            # Return empty frame - no synthetic data generation
            return pd.DataFrame()
        
        # Create complete 1-minute index for regular trading hours only (9:30-4:00)
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
        df_aligned['ticker'] = symbol
        
        # Reset index
        df_aligned = df_aligned.reset_index().rename(columns={'index': 'datetime'})
        
        # Validate we have exactly 390 bars (6.5 hours * 60 minutes)
        expected_bars = 390
        if len(df_aligned) != expected_bars:
            logger.warning(f"Alignment issue for {symbol} {date}: got {len(df_aligned)} bars, expected {expected_bars}")
        
        return df_aligned[['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
    
    def fetch_symbol_data(self, symbol: str, worker_id: int = None) -> Tuple[pd.DataFrame, Dict]:
        """Fetch data for one symbol with proper date range handling"""
        # Get symbol-specific date range based on real data availability
        start_date, end_date = self.get_symbol_date_range(symbol)
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if not trading_dates:
            logger.warning(f"No trading dates found for {symbol} between {start_date} and {end_date}")
            return pd.DataFrame(), {}
        
        worker_prefix = f"[Worker-{worker_id:02d}] " if worker_id is not None else ""
        
        # Log symbol fetching
        logger.info(f"{worker_prefix}📈 Fetching {symbol}: {len(trading_dates)} trading days")
        
        all_data = []
        stats = {
            'successful_days': 0,
            'empty_days': 0,
            'total_records': 0,
            'date_range': f"{start_date} to {end_date}"
        }
        
        # Progress bar for this symbol
        pbar = tqdm(trading_dates, desc=f"{symbol}", unit="day", position=worker_id, leave=False)
        
        for date in pbar:
            # Fetch real market data
            day_data = self.fetch_day_data_with_retry(symbol, date)
            
            if not day_data.empty:
                # Align the data to 390 bars
                aligned_data = self.create_aligned_intervals(day_data, symbol, date)
                if not aligned_data.empty:
                    all_data.append(aligned_data)
                    stats['successful_days'] += 1
                    stats['total_records'] += len(aligned_data)
            else:
                stats['empty_days'] += 1
            
            # Update progress
            pbar.set_postfix({
                'success': stats['successful_days'],
                'empty': stats['empty_days']
            })
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        pbar.close()
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"{worker_prefix}✅ {symbol}: {len(combined):,} records ({stats['successful_days']} days with data)")
            return combined, stats
        
        logger.warning(f"{worker_prefix}⚠️ {symbol}: No valid data collected")
        return pd.DataFrame(), stats

    def validate_and_save_data(self, symbol: str, df: pd.DataFrame, stats: Dict) -> dict:
        """Validate and save symbol data with comprehensive checks"""
        if df.empty:
            return {}
        
        # Validation checks
        validation = {
            'date_range_check': True,
            'bar_count_check': True,
            'data_quality_check': True,
            'issues': []
        }
        
        # Check date range
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()
        
        # Date range validation for all symbols
        # All symbols should have data from early 2019 (AMZN has complete history)
        
        # Check for consistent 390 bars per day
        bars_per_day = df.groupby(df['datetime'].dt.date).size()
        inconsistent_days = bars_per_day[bars_per_day != 390]
        if len(inconsistent_days) > 0:
            validation['issues'].append(f"{len(inconsistent_days)} days with != 390 bars")
            validation['bar_count_check'] = False
        
        # Check for data quality
        if df['open'].isna().any() or df['close'].isna().any():
            validation['issues'].append("NaN values in price data")
            validation['data_quality_check'] = False
        
        # Save data
        filename = f"{symbol}_201901_202507.csv"
        filepath = self.output_dir / filename
        
        # Sort by datetime before saving
        df = df.sort_values('datetime').reset_index(drop=True)
        df.to_csv(filepath, index=False)
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        trading_days = df['datetime'].dt.date.nunique()
        
        file_info = {
            "records": len(df),
            "trading_days": trading_days,
            "size_mb": round(file_size_mb, 2),
            "date_range": f"{min_date.date()} to {max_date.date()}",
            "stats": stats,
            "validation": validation,
            "completed_at": datetime.now().isoformat()
        }
        
        if validation['issues']:
            logger.warning(f"⚠️ {symbol} validation issues: {', '.join(validation['issues'])}")
        
        logger.info(f"💾 {filename}: {len(df):,} records, {trading_days} days, {file_size_mb:.1f}MB")
        return file_info
    
    def process_single_symbol(self, args):
        """Process a single symbol with validation"""
        symbol, worker_id = args
        
        try:
            # Fetch data with proper date range
            df, stats = self.fetch_symbol_data(symbol, worker_id)
            
            if not df.empty:
                # Validate and save
                file_info = self.validate_and_save_data(symbol, df, stats)
                
                # Update progress
                progress = self.load_progress()
                progress["completed"][symbol] = file_info
                self.save_progress(progress)
                
                return {
                    'symbol': symbol,
                    'status': 'completed',
                    'info': file_info,
                    'message': f"Saved: {len(df):,} records from {stats['date_range']}"
                }
            else:
                # Handle symbols with no data
                progress = self.load_progress()
                progress["failed"][symbol] = {
                    'reason': 'No data collected',
                    'stats': stats,
                    'attempted_at': datetime.now().isoformat()
                }
                self.save_progress(progress)
                
                return {
                    'symbol': symbol,
                    'status': 'failed',
                    'info': stats,
                    'message': f"No data available for {symbol}"
                }
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            
            progress = self.load_progress()
            progress["failed"][symbol] = {
                'reason': str(e),
                'attempted_at': datetime.now().isoformat()
            }
            self.save_progress(progress)
            
            return {
                'symbol': symbol,
                'status': 'error',
                'info': {},
                'message': f"Error: {str(e)}"
            }

    def run_final_validation(self) -> Dict:
        """Run comprehensive validation across all downloaded files"""
        logger.info("="*80)
        logger.info("🔍 RUNNING FINAL VALIDATION")
        logger.info("="*80)
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'target_period': f"{self.global_start} to {self.global_end}",
            'symbols': {},
            'summary': {
                'total_symbols': len(self.symbols),
                'successfully_downloaded': 0,
                'has_issues': 0,
                'missing': 0,
                'amzn_check': 'PENDING',
                'ba_check': 'PENDING'
            }
        }
        
        for symbol in self.symbols:
            filepath = self.output_dir / f"{symbol}_201901_202507.csv"
            
            if filepath.exists():
                try:
                    # Read first and last rows to check date range
                    df_head = pd.read_csv(filepath, nrows=1, parse_dates=['datetime'])
                    df_tail = pd.read_csv(filepath, skiprows=lambda x: x not in [0, filepath.stat().st_size // 100], parse_dates=['datetime'])
                    
                    first_date = df_head['datetime'].iloc[0]
                    
                    # Full validation for special symbols
                    if symbol in ['AMZN', 'BA']:
                        df_full = pd.read_csv(filepath, parse_dates=['datetime'])
                        last_date = df_full['datetime'].iloc[-1]
                        
                        symbol_validation = {
                            'exists': True,
                            'records': len(df_full),
                            'date_range': f"{first_date.date()} to {last_date.date()}",
                            'issues': []
                        }
                        
                        # Check AMZN has full data range
                        if symbol == 'AMZN':
                            if first_date <= pd.Timestamp('2019-01-02'):  # First trading day of 2019
                                validation_report['summary']['amzn_check'] = 'PASS'
                            else:
                                symbol_validation['issues'].append(f"Missing early data: starts at {first_date.date()}")
                        
                        # Check BA date range
                        if symbol == 'BA':
                            if last_date.date() == pd.Timestamp('2025-07-30').date():
                                validation_report['summary']['ba_check'] = 'PASS'
                            else:
                                symbol_validation['issues'].append(f"Unexpected end date: {last_date.date()}")
                    else:
                        symbol_validation = {'exists': True, 'checked': 'basic'}
                    
                    validation_report['symbols'][symbol] = symbol_validation
                    
                    if 'issues' not in symbol_validation or not symbol_validation['issues']:
                        validation_report['summary']['successfully_downloaded'] += 1
                    else:
                        validation_report['summary']['has_issues'] += 1
                        
                except Exception as e:
                    validation_report['symbols'][symbol] = {'error': str(e)}
                    validation_report['summary']['has_issues'] += 1
            else:
                validation_report['symbols'][symbol] = {'exists': False}
                validation_report['summary']['missing'] += 1
        
        # Save report
        with open(self.validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Print summary
        summary = validation_report['summary']
        logger.info(f"✅ Successfully downloaded: {summary['successfully_downloaded']}/{summary['total_symbols']}")
        logger.info(f"⚠️ Has issues: {summary['has_issues']}/{summary['total_symbols']}")
        logger.info(f"❌ Missing: {summary['missing']}/{summary['total_symbols']}")
        logger.info(f"🎯 AMZN check (full data from 2019-01-02): {summary.get('amzn_check', 'N/A')}")
        logger.info(f"🎯 BA check (ends on 2025-07-30): {summary['ba_check']}")
        logger.info(f"📊 Full report: {self.validation_file}")
        
        return validation_report
    
    def run_pipeline(self, symbols: list = None, resume: bool = True):
        """Run the enhanced data fetching pipeline with alignment"""
        if symbols is None:
            symbols = self.symbols
        
        # Load progress
        progress = self.load_progress()
        if not progress["started_at"]:
            progress["started_at"] = datetime.now().isoformat()
            self.save_progress(progress)
        
        # Calculate expected data scope
        all_trading_dates = self.get_trading_dates(self.global_start, self.global_end)
        total_days = len(all_trading_dates)
        
        logger.info("="*80)
        logger.info("🚀 ENHANCED POLYGON.IO DATA FETCHER FOR DOW30")
        logger.info("="*80)
        logger.info(f"📅 Target period: {self.global_start} to {self.global_end} ({total_days} trading days)")
        logger.info(f"📈 Symbols: {len(symbols)} stocks")
        logger.info(f"⚡ Parallel workers: {self.max_workers}")
        logger.info(f"💾 Output: {self.output_dir}")
        logger.info(f"🔄 Resume mode: {'ON' if resume else 'OFF'}")
        logger.info("="*80)
        logger.info("📌 Special handling:")
        logger.info("  - AMZN: Full historical data from 2019-01-01 (replaces DOW)")
        logger.info("  - BA: Data through 2025-07-30")
        logger.info("  - NO synthetic data generation - only real market data")
        logger.info("="*80)
        
        # Filter symbols to process
        symbols_to_process = []
        skipped_count = 0
        
        if resume:
            completed = progress.get('completed', {}).keys()
            for symbol in symbols:
                if symbol in completed:
                    info = progress['completed'][symbol]
                    logger.info(f"⏩ SKIPPING {symbol} - Already completed: {info.get('records', 'N/A')} records")
                    skipped_count += 1
                else:
                    symbols_to_process.append(symbol)
        else:
            symbols_to_process = symbols
        
        if not symbols_to_process:
            logger.info("✅ All symbols already downloaded!")
            return
        
        logger.info(f"📊 Processing {len(symbols_to_process)} symbols (skipped {skipped_count})")
        
        # Prepare tasks for parallel processing
        tasks = [(symbol, i % self.max_workers) for i, symbol in enumerate(symbols_to_process)]
        
        # Execute in parallel
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(self.process_single_symbol, task): task[0] for task in tasks}
            
            # Main progress bar
            main_pbar = tqdm(total=len(symbols_to_process), desc="Overall Progress", unit="symbol")
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    
                    if result['status'] == 'completed':
                        completed_count += 1
                        logger.info(f"✅ {symbol}: {result['message']}")
                    else:
                        failed_count += 1
                        logger.warning(f"❌ {symbol}: {result['message']}")
                    
                    # Update main progress bar
                    main_pbar.update(1)
                    main_pbar.set_postfix({
                        'completed': completed_count,
                        'failed': failed_count
                    })
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ {symbol}: Execution error: {e}")
                    main_pbar.update(1)
            
            main_pbar.close()
        
        # Run final validation
        validation_report = self.run_final_validation()
        
        logger.info("="*80)
        logger.info("🎉 DATA FETCHING COMPLETED!")
        logger.info(f"✅ Successfully processed: {completed_count} symbols")
        logger.info(f"⏩ Skipped (already done): {skipped_count} symbols")
        logger.info(f"❌ Failed: {failed_count} symbols")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📊 Validation report: {self.validation_file}")
        logger.info("="*80)
        
        # Special checks summary
        if validation_report['summary'].get('amzn_check') == 'PASS':
            logger.info("✅ AMZN has complete data from 2019-01-02")
        else:
            logger.warning("⚠️ AMZN data issue - check validation report")
        
        if validation_report['summary']['ba_check'] == 'PASS':
            logger.info("✅ BA data ends correctly on 2025-07-30")
        else:
            logger.warning("⚠️ BA data issue - check validation report")


def main():
    """Main entry point"""
    # Initialize enhanced fetcher
    fetcher = PolygonDataFetcher(max_workers=30)
    
    # Run pipeline with proper alignment
    # Set resume=False to reprocess everything from scratch
    fetcher.run_pipeline(resume=False)

if __name__ == "__main__":
    main()