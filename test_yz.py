import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
from typing import Optional, Tuple, List
import warnings
import json
warnings.filterwarnings('ignore')

class YangZhangVolatilityEstimator:
    """
    Implementation of Yang-Zhang Volatility Estimator for intraday data
    This estimator is the most powerful for handling drift and opening jumps
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        
    def test_api_connection(self):
        """Test API connection and print account status"""
        url = f"https://api.polygon.io/v1/marketstatus/now"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            print(f"API Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"API Connection Error: {e}")
            return False
    
    def fetch_minute_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV data from Polygon.io
        """
        url = f"{self.base_url}/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, params=params)
            print(f"Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error Response: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            print(f"Response keys: {data.keys()}")
            
            if 'status' in data:
                print(f"Data status: {data['status']}")
            
            if 'resultsCount' in data:
                print(f"Results count: {data['resultsCount']}")
            
            if data.get('status') != 'OK' or 'results' not in data or data.get('resultsCount', 0) == 0:
                print(f"No data available for {ticker}")
                if 'message' in data:
                    print(f"Message: {data['message']}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Add timezone awareness (US/Eastern for US stocks)
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
            
            # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
            df = df.between_time('09:30', '16:00')
            
            print(f"Successfully fetched {len(df)} data points for {ticker}")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def calculate_yang_zhang_volatility(self, 
                                       df: pd.DataFrame,
                                       window: int = 30,
                                       trading_periods: int = 390,
                                       clean: bool = True) -> pd.Series:
        """
        Calculate Yang-Zhang volatility estimator
        
        Parameters:
        -----------
        df : DataFrame with OHLC data
        window : lookback window in minutes (default 30)
        trading_periods : number of trading periods per year for annualization
                         (390 minutes per day * 252 days = 98,280)
        clean : whether to clean data for outliers
        
        Returns:
        --------
        Series with Yang-Zhang volatility estimates (annualized)
        """
        
        if clean:
            df = self._clean_data(df)
        
        # Calculate log returns for different price points
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        
        # Calculate overnight returns (open to previous close)
        log_oc = np.log(df['open'] / df['close'].shift(1))
        log_oc_sq = log_oc ** 2
        
        # Calculate close-to-close returns
        log_cc = np.log(df['close'] / df['close'].shift(1))
        
        # Rogers-Satchell volatility component (handles drift)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        # Calculate components with rolling windows
        # Overnight variance
        n = window
        overnight_var = log_oc_sq.rolling(window=n).mean()
        
        # Open-to-close variance  
        oc_var = log_co.rolling(window=n).var()
        
        # Rogers-Satchell variance
        rs_var = rs.rolling(window=n).mean()
        
        # Yang-Zhang estimator combining all components
        # k is chosen to minimize variance (usually around 0.34)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        
        # Final Yang-Zhang variance
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
        
        # Convert to volatility and annualize
        # Annualization factor: sqrt(periods per year)
        annualization_factor = np.sqrt(trading_periods * 252 / window)
        yz_vol = np.sqrt(yz_var) * annualization_factor
        
        return yz_vol
    
    def _clean_data(self, df: pd.DataFrame, threshold: float = 5) -> pd.DataFrame:
        """
        Clean data for outliers using modified z-score method
        """
        df_clean = df.copy()
        
        # Calculate returns
        returns = np.log(df_clean['close'] / df_clean['close'].shift(1))
        
        # Use median absolute deviation for robust outlier detection
        median_return = returns.median()
        mad = np.abs(returns - median_return).median()
        
        if mad != 0:
            modified_z_scores = 0.6745 * (returns - median_return) / mad
            # Mark outliers
            outliers = np.abs(modified_z_scores) > threshold
            
            # Replace outlier values with interpolated values
            for col in ['open', 'high', 'low', 'close']:
                df_clean.loc[outliers, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear', limit=2)
        
        return df_clean.dropna()
    
    def calculate_alternative_estimators(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate multiple volatility estimators for comparison
        """
        results = pd.DataFrame(index=df.index)
        
        # 1. Close-to-Close (Classical)
        returns = np.log(df['close'] / df['close'].shift(1))
        results['close_to_close'] = returns.rolling(window=window).std() * np.sqrt(252 * 390 / window)
        
        # 2. Parkinson (High-Low)
        hl_ratio = np.log(df['high'] / df['low'])
        results['parkinson'] = np.sqrt((hl_ratio ** 2).rolling(window=window).mean() / (4 * np.log(2))) * np.sqrt(252 * 390 / window)
        
        # 3. Garman-Klass
        gk_hl = 0.5 * np.log(df['high'] / df['low']) ** 2
        gk_co = (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        results['garman_klass'] = np.sqrt((gk_hl - gk_co).rolling(window=window).mean()) * np.sqrt(252 * 390 / window)
        
        # 4. Rogers-Satchell
        rs = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) + 
              np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
        results['rogers_satchell'] = np.sqrt(rs.rolling(window=window).mean()) * np.sqrt(252 * 390 / window)
        
        # 5. Yang-Zhang
        results['yang_zhang'] = self.calculate_yang_zhang_volatility(df, window=window)
        
        return results
    
    def plot_intraday_volatility(self, ticker: str, df: pd.DataFrame, 
                                 volatilities: pd.DataFrame, window: int = 30):
        """
        Create comprehensive visualization of intraday volatility patterns
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2)
        
        # 1. Price and Volume
        ax1 = fig.add_subplot(gs[0, :])
        ax1_twin = ax1.twinx()
        
        ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
        ax1.fill_between(df.index, df['low'], df['high'], alpha=0.2, color='gray', label='High-Low Range')
        ax1_twin.bar(df.index, df['volume'], alpha=0.3, color='blue', label='Volume', width=0.0003)
        
        ax1.set_title(f'{ticker} - Price and Volume', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1_twin.set_ylabel('Volume', fontsize=11)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Yang-Zhang Volatility with Components
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(volatilities.index, volatilities['yang_zhang'], 
                label='Yang-Zhang', color='red', linewidth=2)
        ax2.fill_between(volatilities.index, 0, volatilities['yang_zhang'], 
                         alpha=0.3, color='red')
        
        # Add 30-period moving average
        yz_ma = volatilities['yang_zhang'].rolling(window=30).mean()
        ax2.plot(volatilities.index, yz_ma, label='30-period MA', 
                color='darkred', linestyle='--', linewidth=1)
        
        ax2.set_title(f'Yang-Zhang Volatility (Annualized, {window}-min window)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatility (%)', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # 3. Comparison of Estimators
        ax3 = fig.add_subplot(gs[2, :])
        for col in volatilities.columns:
            if col != 'yang_zhang':
                ax3.plot(volatilities.index, volatilities[col], label=col.replace('_', ' ').title(), 
                        alpha=0.7, linewidth=1)
        ax3.plot(volatilities.index, volatilities['yang_zhang'], 
                label='Yang-Zhang', color='red', linewidth=2)
        
        ax3.set_title('Comparison of Volatility Estimators', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volatility (%)', fontsize=11)
        ax3.legend(loc='upper right', ncol=3)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # 4. Intraday Volatility Pattern (Heatmap by hour)
        ax4 = fig.add_subplot(gs[3, 0])
        
        # Create hourly aggregation
        volatility_hourly = volatilities.copy()
        volatility_hourly['hour'] = volatility_hourly.index.hour
        volatility_hourly['date'] = volatility_hourly.index.date
        
        pivot_data = volatility_hourly.pivot_table(
            values='yang_zhang', 
            index='hour', 
            columns='date', 
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, ax=ax4, cmap='YlOrRd', cbar_kws={'label': 'Volatility'})
            ax4.set_title('Intraday Volatility Pattern by Hour', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Hour of Day')
        
        # 5. Distribution of Volatility
        ax5 = fig.add_subplot(gs[3, 1])
        
        # Plot histogram and KDE
        vol_data = volatilities['yang_zhang'].dropna()
        if len(vol_data) > 0:
            vol_data.hist(bins=50, ax=ax5, alpha=0.7, color='red', edgecolor='black')
            vol_data.plot(kind='kde', ax=ax5, secondary_y=True, color='darkred', linewidth=2)
            
            # Add statistics
            mean_vol = vol_data.mean()
            median_vol = vol_data.median()
            ax5.axvline(mean_vol, color='blue', linestyle='--', label=f'Mean: {mean_vol:.1%}')
            ax5.axvline(median_vol, color='green', linestyle='--', label=f'Median: {median_vol:.1%}')
            
            ax5.set_title('Distribution of Yang-Zhang Volatility', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Volatility (%)')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f'Intraday Volatility Analysis - {ticker}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def analyze_multiple_stocks(self, tickers: List[str], start_date: str, 
                               end_date: str, window: int = 30):
        """
        Analyze multiple stocks and create comparison plots
        """
        all_volatilities = {}
        
        for ticker in tickers:
            print(f"\n{'='*60}")
            print(f"Processing {ticker}...")
            df = self.fetch_minute_data(ticker, start_date, end_date)
            
            if df.empty:
                print(f"Skipping {ticker} - no data available")
                continue
            
            # Calculate volatilities
            volatilities = self.calculate_alternative_estimators(df, window=window)
            all_volatilities[ticker] = volatilities['yang_zhang']
            
            # Create individual plot
            fig = self.plot_intraday_volatility(ticker, df, volatilities, window=window)
            plt.show()
        
        # Create comparison plot
        if len(all_volatilities) > 1:
            self.plot_comparison(all_volatilities)
    
    def plot_comparison(self, all_volatilities: dict):
        """
        Create comparison plot for multiple stocks
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: All volatilities on same axis
        ax1 = axes[0]
        for ticker, vol in all_volatilities.items():
            ax1.plot(vol.index, vol, label=ticker, linewidth=1.5, alpha=0.8)
        
        ax1.set_title('Yang-Zhang Volatility Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Volatility (%)', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Plot 2: Correlation heatmap
        ax2 = axes[1]
        
        # Create DataFrame for correlation
        vol_df = pd.DataFrame(all_volatilities)
        correlation = vol_df.corr()
        
        if not correlation.empty:
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax2, vmin=-1, vmax=1)
            ax2.set_title('Volatility Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_volatility_metrics(self, volatility: pd.Series) -> dict:
        """
        Calculate summary metrics for volatility series
        """
        return {
            'mean': volatility.mean(),
            'median': volatility.median(),
            'std': volatility.std(),
            'skew': volatility.skew(),
            'kurtosis': volatility.kurtosis(),
            'min': volatility.min(),
            'max': volatility.max(),
            'q25': volatility.quantile(0.25),
            'q75': volatility.quantile(0.75),
            'iqr': volatility.quantile(0.75) - volatility.quantile(0.25)
        }

# Main execution
def main():
    # Initialize the estimator with your API key
    api_key = "W4WJfLUGGO5vdylrgw9RKKyD9LANdWRu"
    estimator = YangZhangVolatilityEstimator(api_key)
    
    # First test the API connection
    print("Testing API connection...")
    if not estimator.test_api_connection():
        print("API connection failed. Please check your API key.")
        return
    
    # Define classic stocks to analyze
    tickers = ['AAPL', 'MSFT', 'SPY', 'TSLA', 'NVDA']
    
    # Use proper dates - let's get the last 5 trading days
    # We'll use a date range that's sure to have data
    end_date = '2024-08-09'  # A recent Friday
    start_date = '2024-08-05'  # The Monday of that week
    
    print(f"\nAnalyzing stocks from {start_date} to {end_date}")
    print("=" * 60)
    
    # Analyze multiple stocks
    estimator.analyze_multiple_stocks(tickers, start_date, end_date, window=30)
    
    # Detailed analysis for a single stock (AAPL)
    print("\n" + "="*60)
    print("Detailed Analysis for AAPL:")
    print("=" * 60)
    
    df_aapl = estimator.fetch_minute_data('AAPL', start_date, end_date)
    if not df_aapl.empty:
        # Calculate with different window sizes
        windows = [15, 30, 60]
        results = {}
        
        for window in windows:
            vol = estimator.calculate_yang_zhang_volatility(df_aapl, window=window)
            results[f'{window}min'] = vol
            
            metrics = estimator.calculate_volatility_metrics(vol.dropna())
            print(f"\n{window}-minute window metrics:")
            for key, value in metrics.items():
                if key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'iqr']:
                    print(f"  {key:10s}: {value:6.2%}")
                else:
                    print(f"  {key:10s}: {value:6.2f}")
        
        # Plot comparison of different windows
        fig, ax = plt.subplots(figsize=(14, 6))
        for window_name, vol in results.items():
            ax.plot(vol.index, vol, label=window_name, alpha=0.8)
        
        ax.set_title('AAPL: Yang-Zhang Volatility with Different Windows', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.tight_layout()
        plt.show()
    else:
        print("No data available for detailed analysis")

if __name__ == "__main__":
    main()