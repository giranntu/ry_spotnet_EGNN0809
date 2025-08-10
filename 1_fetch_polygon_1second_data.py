#!/usr/bin/env python3
"""
Polygon.io 1-SECOND Data Fetcher for DOW30 with Perfect Alignment
- Downloads 1-second resolution data instead of 1-minute
- Handles much larger data volumes (60x more data)
- Optimized for memory and storage efficiency
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
        logging.FileHandler('polygon_1second_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load DOW30 symbols from config
with open('config/dow30_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    DOW30_SYMBOLS = config['dow30_symbols']

class PolygonSecondDataFetcher:
    """Enhanced Polygon.io fetcher for 1-SECOND resolution data"""
    
    API_KEY = "W4WJfLUGGO5vdylrgw9RKKyD9LANdWRu"
    BASE_URL = "https://api.polygon.io"
    
    # Symbol-specific constraints based on real market data availability
    SYMBOL_CONSTRAINTS = {
        'BA': {
            'last_known_data': '2025-07-30',
            'note': 'Data availability constraint'
        }
    }
    
    def __init__(self, max_workers=10, test_days=None):  # Reduced workers due to higher data volume
        """Initialize the 1-second data fetcher"""
        self.symbols = DOW30_SYMBOLS
        self.output_dir = Path("rawdata/by_comp_1second")  # Different directory for 1-second data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = Path("polygon_1second_progress.json")
        self.validation_file = Path("data_1second_validation_report.json")
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.rate_limit_delay = 0.1  # Increased delay for larger requests
        self.test_days = test_days  # Limit days for testing
        
        # Global date range for alignment
        self.global_start = '2019-01-01'
        self.global_end = '2025-07-30'
        
        # 1-second specific settings
        self.seconds_per_day = 23400  # 6.5 hours * 60 minutes * 60 seconds
        self.chunk_size = 3600  # Process 1 hour at a time for memory efficiency
        
        logger.info(f"✅ Output directory: {self.output_dir}")
        logger.info(f"📋 Progress file: {self.progress_file}")
        logger.info(f"📊 Validation file: {self.validation_file}")
        logger.info(f"⚡ Parallel workers: {max_workers}")
        logger.info(f"📅 Target period: {self.global_start} to {self.global_end}")
        logger.info(f"⏱️ Resolution: 1-SECOND (23,400 bars per day)")
        if self.test_days:
            logger.info(f"🧪 TEST MODE: Limited to last {self.test_days} trading days per symbol")
    
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
        filename = f"{symbol}_201901_202507_1second.csv.gz"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Quick validation (handle gzip)
            df = pd.read_csv(filepath, nrows=5, compression='gzip')
            has_datetime = 'datetime' in df.columns
            
            # For large files, estimate line count
            with open(filepath, 'rb') as f:
                # Read first 1MB to estimate
                sample = f.read(1024*1024)
                lines_in_sample = sample.count(b'\n')
                total_size = filepath.stat().st_size
                estimated_lines = int(lines_in_sample * total_size / (1024*1024))
            
            return True, filepath, {
                "size_mb": round(file_size_mb, 1),
                "estimated_records": estimated_lines,
                "valid_format": has_datetime,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
        return False, filepath, None
    
    def fetch_hour_data_with_retry(self, symbol: str, date: str, hour_start: int, hour_end: int, max_retries: int = 10) -> pd.DataFrame:
        """Fetch 1-second data for a specific hour range with retry logic"""
        for attempt in range(max_retries):
            try:
                # Construct time range for the hour
                start_time = f"{date}T{hour_start:02d}:00:00"
                end_time = f"{date}T{hour_end:02d}:00:00"
                
                # Use v2 aggregates endpoint with second resolution
                url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/second/{date}/{date}"
                params = {
                    'apiKey': self.API_KEY,
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50000,  # Max limit per request
                    'timestamp.gte': start_time,
                    'timestamp.lt': end_time
                }
                
                response = requests.get(url, params=params, timeout=60)  # Increased timeout for larger data
                
                # Handle rate limiting with improved exponential backoff
                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(f"Rate limited for {symbol} {date} hour {hour_start}, attempt {attempt+1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    logger.debug(f"API error {response.status_code} for {symbol} {date} hour {hour_start}")
                    return pd.DataFrame()
                
                data = response.json()
                
                if data.get('status') != 'OK' or not data.get('results'):
                    # No data available for this hour
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(data['results'])
                
                # Convert timestamp to Eastern Time
                df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)
                
                # Map to OHLCV format
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                df['ticker'] = symbol
                
                # Select and sort
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
                df = df.sort_values('datetime').reset_index(drop=True)
                
                return df
                
            except requests.exceptions.Timeout:
                wait_time = 5  # Longer wait for timeouts on larger requests
                logger.warning(f"Timeout for {symbol} {date} hour {hour_start} (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                wait_time = 5
                logger.warning(f"Connection error for {symbol} {date} hour {hour_start}: {e} (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                wait_time = 5
                logger.warning(f"Error fetching {symbol} {date} hour {hour_start}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        return pd.DataFrame()
    
    def fetch_day_data_with_retry(self, symbol: str, date: str, max_retries: int = 10) -> pd.DataFrame:
        """Fetch 1-second data for entire day by fetching hourly chunks"""
        all_data = []
        
        # Trading hours: 9:30 AM - 4:00 PM ET
        # Fetch data in hourly chunks to avoid memory issues
        hours = [
            (9, 10),   # 9:30-10:00 (partial)
            (10, 11),  # 10:00-11:00
            (11, 12),  # 11:00-12:00
            (12, 13),  # 12:00-13:00
            (13, 14),  # 13:00-14:00
            (14, 15),  # 14:00-15:00
            (15, 16),  # 15:00-16:00 (includes close)
        ]
        
        for hour_start, hour_end in hours:
            hour_data = self.fetch_hour_data_with_retry(symbol, date, hour_start, hour_end, max_retries)
            if not hour_data.empty:
                # Filter to regular trading hours only
                market_open = pd.Timestamp(f"{date} 09:30:00")
                market_close = pd.Timestamp(f"{date} 16:00:00")
                hour_data = hour_data[(hour_data['datetime'] >= market_open) & (hour_data['datetime'] < market_close)]
                
                if not hour_data.empty:
                    all_data.append(hour_data)
            
            # Small delay between hour requests
            time.sleep(0.5)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values('datetime').reset_index(drop=True)
            return combined
        
        return pd.DataFrame()
    
    def create_aligned_intervals(self, df: pd.DataFrame, symbol: str, date: str) -> pd.DataFrame:
        """Create aligned 1-second intervals for regular trading hours"""
        if df.empty:
            return pd.DataFrame()
        
        # Handle duplicate timestamps by keeping last value (most recent trade)
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # Create complete 1-second index for regular trading hours (9:30-4:00)
        market_open = pd.Timestamp(f"{date} 09:30:00")
        market_close = pd.Timestamp(f"{date} 15:59:59")  # Last second
        complete_index = pd.date_range(start=market_open, end=market_close, freq='1s')
        
        # Reindex with forward fill for missing seconds
        df_indexed = df.set_index('datetime')
        
        # Check for any remaining duplicates in the index
        if df_indexed.index.duplicated().any():
            logger.warning(f"Found {df_indexed.index.duplicated().sum()} duplicate timestamps for {symbol} on {date}")
            # Group by index and take mean of duplicates
            df_indexed = df_indexed.groupby(level=0).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum',
                'ticker': 'first'
            })
        
        try:
            df_aligned = df_indexed.reindex(complete_index)
        except Exception as e:
            logger.error(f"Failed to reindex {symbol} on {date}: {e}")
            return pd.DataFrame()
        
        # Forward fill prices (standard practice for missing seconds)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_aligned[col] = df_aligned[col].ffill()
        
        # Volume should be 0 for missing bars
        df_aligned['volume'] = df_aligned['volume'].fillna(0)
        df_aligned['ticker'] = symbol
        
        # Reset index
        df_aligned = df_aligned.reset_index().rename(columns={'index': 'datetime'})
        
        # Validate we have exactly 23,400 bars (6.5 hours * 60 minutes * 60 seconds)
        expected_bars = 23400
        if len(df_aligned) != expected_bars:
            logger.warning(f"Alignment issue for {symbol} {date}: got {len(df_aligned)} bars, expected {expected_bars}")
        
        return df_aligned[['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
    
    def fetch_symbol_data(self, symbol: str, worker_id: int = None, limit_days: int = None) -> Tuple[pd.DataFrame, Dict]:
        """Fetch 1-second data for one symbol"""
        # Get symbol-specific date range
        start_date, end_date = self.get_symbol_date_range(symbol)
        
        # For 1-second data, we might want to limit the date range initially
        # to avoid overwhelming storage - e.g., start with last 3 months
        # Uncomment below to limit date range:
        # start_date = '2025-05-01'  # Last 3 months only for testing
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if not trading_dates:
            logger.warning(f"No trading dates found for {symbol} between {start_date} and {end_date}")
            return pd.DataFrame(), {}
        
        # Limit number of days if specified (for testing)
        if limit_days:
            trading_dates = trading_dates[-limit_days:]  # Get most recent days
            logger.info(f"Limiting to last {limit_days} trading days for testing")
        
        worker_prefix = f"[Worker-{worker_id:02d}] " if worker_id is not None else ""
        
        # Log symbol fetching
        logger.info(f"{worker_prefix}📈 Fetching {symbol} (1-SECOND): {len(trading_dates)} trading days")
        logger.info(f"{worker_prefix}⚠️ Expected data size: ~{len(trading_dates) * 23400:,} records")
        
        all_data = []
        stats = {
            'successful_days': 0,
            'empty_days': 0,
            'total_records': 0,
            'date_range': f"{start_date} to {end_date}"
        }
        
        # Progress bar for this symbol
        pbar = tqdm(trading_dates, desc=f"{symbol} (1s)", unit="day", position=worker_id, leave=False)
        
        for date in pbar:
            # Fetch 1-second data for the day
            day_data = self.fetch_day_data_with_retry(symbol, date)
            
            if not day_data.empty:
                # Align the data to 23,400 bars
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
                'empty': stats['empty_days'],
                'records': f"{stats['total_records']:,}"
            })
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        pbar.close()
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"{worker_prefix}✅ {symbol}: {len(combined):,} 1-second records ({stats['successful_days']} days)")
            return combined, stats
        
        logger.warning(f"{worker_prefix}⚠️ {symbol}: No valid 1-second data collected")
        return pd.DataFrame(), stats

    def validate_and_save_data(self, symbol: str, df: pd.DataFrame, stats: Dict) -> dict:
        """Validate and save 1-second symbol data"""
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
        
        # Check for consistent 23,400 bars per day
        bars_per_day = df.groupby(df['datetime'].dt.date).size()
        inconsistent_days = bars_per_day[bars_per_day != 23400]
        if len(inconsistent_days) > 0:
            validation['issues'].append(f"{len(inconsistent_days)} days with != 23,400 bars")
            validation['bar_count_check'] = False
        
        # Check for data quality
        if df['open'].isna().any() or df['close'].isna().any():
            validation['issues'].append("NaN values in price data")
            validation['data_quality_check'] = False
        
        # Save data (consider compression for large files)
        filename = f"{symbol}_201901_202507_1second.csv.gz"
        filepath = self.output_dir / filename
        
        # Sort by datetime before saving
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Save with compression to reduce file size
        df.to_csv(filepath, index=False, compression='gzip')
        logger.info(f"💾 Saved with gzip compression: {filepath}")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
        trading_days = df['datetime'].dt.date.nunique()
        
        file_info = {
            "records": len(df),
            "trading_days": trading_days,
            "size_mb": round(file_size_mb, 2),
            "date_range": f"{min_date.date()} to {max_date.date()}",
            "stats": stats,
            "validation": validation,
            "resolution": "1-second",
            "bars_per_day": 23400,
            "completed_at": datetime.now().isoformat()
        }
        
        if validation['issues']:
            logger.warning(f"⚠️ {symbol} validation issues: {', '.join(validation['issues'])}")
        
        logger.info(f"💾 {filename}: {len(df):,} 1-second records, {trading_days} days, ~{file_size_mb:.1f}MB")
        return file_info
    
    def process_single_symbol(self, args):
        """Process a single symbol with validation"""
        symbol, worker_id = args
        
        try:
            # Check if already downloaded
            exists, filepath, info = self.is_already_downloaded(symbol)
            if exists:
                logger.info(f"⏩ Skipping {symbol} - already downloaded: {info['size_mb']}MB, ~{info['estimated_records']:,} records")
                return {
                    'symbol': symbol,
                    'status': 'skipped',
                    'info': info,
                    'message': f"Already exists: {info['size_mb']}MB"
                }
            
            # Fetch 1-second data (use test_days if specified)
            df, stats = self.fetch_symbol_data(symbol, worker_id, limit_days=self.test_days)
            
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
                    'message': f"Saved: {len(df):,} 1-second records from {stats['date_range']}"
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
                    'message': f"No 1-second data available for {symbol}"
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
        """Run comprehensive validation for 1-second data"""
        logger.info("="*80)
        logger.info("🔍 RUNNING FINAL VALIDATION FOR 1-SECOND DATA")
        logger.info("="*80)
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'target_period': f"{self.global_start} to {self.global_end}",
            'resolution': '1-second',
            'expected_bars_per_day': 23400,
            'symbols': {},
            'summary': {
                'total_symbols': len(self.symbols),
                'successfully_downloaded': 0,
                'has_issues': 0,
                'missing': 0,
                'total_size_gb': 0
            }
        }
        
        total_size_bytes = 0
        
        for symbol in self.symbols:
            filepath = self.output_dir / f"{symbol}_201901_202507_1second.csv.gz"
            
            if filepath.exists():
                file_size = filepath.stat().st_size
                total_size_bytes += file_size
                validation_report['symbols'][symbol] = {
                    'exists': True,
                    'size_mb': round(file_size / (1024*1024), 1)
                }
                validation_report['summary']['successfully_downloaded'] += 1
            else:
                validation_report['symbols'][symbol] = {'exists': False}
                validation_report['summary']['missing'] += 1
        
        validation_report['summary']['total_size_gb'] = round(total_size_bytes / (1024**3), 2)
        
        # Save report
        with open(self.validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Print summary
        summary = validation_report['summary']
        logger.info(f"✅ Successfully downloaded: {summary['successfully_downloaded']}/{summary['total_symbols']}")
        logger.info(f"❌ Missing: {summary['missing']}/{summary['total_symbols']}")
        logger.info(f"💾 Total data size: {summary['total_size_gb']} GB")
        logger.info(f"📊 Full report: {self.validation_file}")
        
        return validation_report
    
    def run_pipeline(self, symbols: list = None, resume: bool = True):
        """Run the 1-second data fetching pipeline"""
        if symbols is None:
            symbols = self.symbols  # Use all DOW30 symbols
            logger.info(f"📊 Processing all {len(symbols)} DOW30 symbols")
        
        # Load progress
        progress = self.load_progress()
        if not progress["started_at"]:
            progress["started_at"] = datetime.now().isoformat()
            self.save_progress(progress)
        
        # Calculate expected data scope
        all_trading_dates = self.get_trading_dates(self.global_start, self.global_end)
        total_days = len(all_trading_dates)
        expected_records_per_symbol = total_days * 23400
        
        logger.info("="*80)
        logger.info("🚀 POLYGON.IO 1-SECOND DATA FETCHER FOR DOW30")
        logger.info("="*80)
        logger.info(f"📅 Target period: {self.global_start} to {self.global_end} ({total_days} trading days)")
        logger.info(f"📈 Symbols: {len(symbols)} stocks")
        logger.info(f"⏱️ Resolution: 1-SECOND (23,400 bars per day)")
        logger.info(f"📊 Expected records per symbol: ~{expected_records_per_symbol:,}")
        logger.info(f"💾 Expected size per symbol: ~{expected_records_per_symbol * 100 / (1024**2):.0f} MB")
        logger.info(f"⚡ Parallel workers: {self.max_workers}")
        logger.info(f"💾 Output: {self.output_dir}")
        logger.info(f"🔄 Resume mode: {'ON' if resume else 'OFF'}")
        logger.info("="*80)
        logger.info("⚠️ WARNING: 1-second data requires significant storage space!")
        logger.info(f"⚠️ Estimated total size for {len(symbols)} symbols: ~{len(symbols) * expected_records_per_symbol * 100 / (1024**3):.1f} GB")
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
        
        # Execute in parallel (with fewer workers due to data volume)
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
                    elif result['status'] == 'skipped':
                        skipped_count += 1
                        logger.info(f"⏩ {symbol}: {result['message']}")
                    else:
                        failed_count += 1
                        logger.warning(f"❌ {symbol}: {result['message']}")
                    
                    # Update main progress bar
                    main_pbar.update(1)
                    main_pbar.set_postfix({
                        'completed': completed_count,
                        'failed': failed_count,
                        'skipped': skipped_count
                    })
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ {symbol}: Execution error: {e}")
                    main_pbar.update(1)
            
            main_pbar.close()
        
        # Run final validation
        validation_report = self.run_final_validation()
        
        logger.info("="*80)
        logger.info("🎉 1-SECOND DATA FETCHING COMPLETED!")
        logger.info(f"✅ Successfully processed: {completed_count} symbols")
        logger.info(f"⏩ Skipped (already done): {skipped_count} symbols")
        logger.info(f"❌ Failed: {failed_count} symbols")
        logger.info(f"💾 Output directory: {self.output_dir}")
        logger.info(f"📊 Validation report: {self.validation_file}")
        logger.info(f"💾 Total data size: {validation_report['summary']['total_size_gb']} GB")
        logger.info("="*80)


def main():
    """Main entry point for 1-second data fetching"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch 1-second resolution data from Polygon.io')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to fetch (default: first 3 for testing)')
    parser.add_argument('--all', action='store_true', help='Fetch all DOW30 symbols (WARNING: Large data volume)')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore previous progress')
    parser.add_argument('--test-days', type=int, default=None, help='Limit to N most recent days for testing (e.g., 2)')
    
    args = parser.parse_args()
    
    # Initialize 1-second data fetcher
    fetcher = PolygonSecondDataFetcher(max_workers=args.workers, test_days=args.test_days)
    
    # Determine which symbols to fetch
    if args.all:
        symbols = None  # Will use all DOW30
        logger.warning("⚠️ Fetching ALL symbols - this will require significant storage space!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted by user")
            return
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DOW30_SYMBOLS[:3]  # Default to first 3 for testing
    
    # Run pipeline
    fetcher.run_pipeline(symbols=symbols, resume=not args.no_resume)

if __name__ == "__main__":
    main()