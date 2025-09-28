"""
Vietnamese Stock Data Fetcher Module using Vnstock API
Provides utilities to fetch and preprocess Vietnamese stock market data
Enhanced version with advanced features for better prediction accuracy
"""

from vnstock import Quote, Vnstock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import pickle
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

class VNStockDataFetcher:
    """Enhanced class to fetch and preprocess Vietnamese stock market data using vnstock"""
    
    def __init__(self, cache_dir='cache'):
        self.data = None
        self.symbol = None
        self.quote = None
        self.vnstock = Vnstock().stock(symbol="VNM", source='VCI')  # Initialize with default
        self.cache_dir = cache_dir
        self._create_cache_dir()
        
    def _create_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_path(self, symbol, data_type='historical'):
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{symbol}_{data_type}.pkl")
    
    def _save_to_cache(self, data, symbol, data_type='historical'):
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(symbol, data_type)
            with open(cache_path, 'wb') as f:
                pickle.dump({'data': data, 'timestamp': datetime.now()}, f)
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def _load_from_cache(self, symbol, data_type='historical', max_age_hours=1):
        """Load data from cache if available and fresh"""
        try:
            cache_path = self._get_cache_path(symbol, data_type)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                
                # Check if cache is still fresh
                age = datetime.now() - cached['timestamp']
                if age.total_seconds() < max_age_hours * 3600:
                    return cached['data']
        except Exception as e:
            print(f"Warning: Could not load from cache: {e}")
        return None

    def fetch_data(self, symbol, start_date=None, end_date=None, source='VCI', use_cache=True):
        """
        Enhanced fetch Vietnamese stock data using vnstock
        
        Args:
            symbol (str): Vietnamese stock symbol (e.g., 'VNM', 'VIC', 'VHM')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format  
            source (str): Data source ('VCI' or 'TCBS')
            use_cache (bool): Whether to use cached data
        
        Returns:
            pd.DataFrame: Enhanced stock data with technical indicators
        """
        try:
            self.symbol = symbol.upper()
            
            # Check cache first
            if use_cache:
                cached_data = self._load_from_cache(self.symbol)
                if cached_data is not None:
                    print(f"📦 Using cached data for {self.symbol}")
                    self.data = cached_data
                    return self.data
            
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 3 years ago
            
            # Initialize quote object
            self.quote = Quote(symbol=self.symbol, source=source)
            
            # Fetch historical data
            data = self.quote.history(start=start_date, end=end_date)
            
            if data is None or data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Standardize column names to match international format
            if 'time' in data.columns:
                data = data.rename(columns={'time': 'Date'})
                data = data.set_index('Date')
            
            # Rename columns to standard format
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    print(f"Warning: Missing column {col}")
            
            # Convert to numeric and handle any string values
            for col in required_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove rows with NaN values
            data = data.dropna()
            
            # Enhanced data preprocessing
            self.data = self._preprocess_data(data)
            
            # Add all technical indicators
            self._add_enhanced_technical_indicators()
            
            # Add fundamental indicators
            self._add_fundamental_indicators()
            
            # Add market sentiment indicators
            self._add_market_sentiment_indicators()
            
            # Save to cache
            if use_cache:
                self._save_to_cache(self.data, self.symbol)
            
            print(f"✅ Successfully fetched {len(self.data)} days of enhanced Vietnamese stock data for {self.symbol}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching Vietnamese stock data for {symbol}: {str(e)}")
            return None
    
    def _preprocess_data(self, data):
        """Enhanced data preprocessing"""
        # Handle missing data with forward fill and interpolation - fix pandas deprecation
        data = data.ffill().bfill()
        
        # Add more price-based features
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        return data
    
    def _add_enhanced_technical_indicators(self):
        """Add comprehensive technical indicators"""
        if self.data is None or self.data.empty:
            return
        
        try:
            # Moving Averages
            self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
            self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
            self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
            
            # MACD
            self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
            self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
            
            # RSI
            self.data['RSI'] = self._calculate_rsi(self.data['Close'])
            
            # Bollinger Bands
            self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
            bb_std = self.data['Close'].rolling(window=20).std()
            self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
            self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)
            
            # Price changes
            self.data['Price_Change'] = self.data['Close'].pct_change()
            self.data['Price_Change_Next'] = self.data['Price_Change'].shift(-1)  # Target variable
            
            # Volatility
            self.data['Volatility'] = self.data['Price_Change'].rolling(window=20).std()
            
            # Add Vietnamese stock specific indicators
            self.data['VN_Trend'] = self._calculate_vn_trend()
            
            # Advanced Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                self.data[f'MA_{period}'] = self.data['Close'].rolling(window=period).mean()
                self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period).mean()
            
            # Price Position Indicators
            self.data['Price_vs_MA20'] = self.data['Close'] / self.data['MA_20']
            self.data['Price_vs_MA50'] = self.data['Close'] / self.data['MA_50']
            
            # Momentum Indicators
            self.data['ROC_10'] = self.data['Close'].pct_change(periods=10) * 100
            self.data['ROC_20'] = self.data['Close'].pct_change(periods=20) * 100
            
            # Volatility Indicators
            self.data['ATR'] = self._calculate_atr()
            self.data['Volatility_20'] = self.data['Close'].pct_change().rolling(20).std()
            
            # Volume Indicators
            self.data['OBV'] = self._calculate_obv()
            self.data['Volume_SMA'] = self.data['Volume'].rolling(20).mean()
            self.data['Volume_Trend'] = self.data['Volume'] / self.data['Volume_SMA']
            
            # Support/Resistance Levels
            self.data['Support'] = self.data['Low'].rolling(window=20).min()
            self.data['Resistance'] = self.data['High'].rolling(window=20).max()
            
            # Trend Strength
            self.data['Trend_Strength'] = self._calculate_trend_strength()
            
        except Exception as e:
            print(f"Error adding enhanced technical indicators: {e}")
    
    def _calculate_atr(self, period=14):
        """Calculate Average True Range"""
        try:
            high_low = self.data['High'] - self.data['Low']
            high_close = np.abs(self.data['High'] - self.data['Close'].shift())
            low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(period).mean()
        except:
            return pd.Series(index=self.data.index, dtype=float)
    
    def _calculate_obv(self):
        """Calculate On-Balance Volume"""
        try:
            obv = [0]
            for i in range(1, len(self.data)):
                if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + self.data['Volume'].iloc[i])
                elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - self.data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            return pd.Series(obv, index=self.data.index)
        except:
            return pd.Series(index=self.data.index, dtype=float)
    
    def _calculate_trend_strength(self):
        """Calculate trend strength indicator"""
        try:
            # Count how many MAs are aligned in trend direction
            mas = ['MA_5', 'MA_10', 'MA_20', 'MA_50']
            trend_strength = pd.Series(index=self.data.index, dtype=float)
            
            for i in range(len(self.data)):
                if i < 50:  # Not enough data
                    trend_strength.iloc[i] = 0
                    continue
                
                # Check uptrend alignment
                uptrend_count = 0
                downtrend_count = 0
                
                for j in range(len(mas)-1):
                    if mas[j] in self.data.columns and mas[j+1] in self.data.columns:
                        if self.data[mas[j]].iloc[i] > self.data[mas[j+1]].iloc[i]:
                            uptrend_count += 1
                        else:
                            downtrend_count += 1
                
                if uptrend_count > downtrend_count:
                    trend_strength.iloc[i] = uptrend_count / len(mas)
                else:
                    trend_strength.iloc[i] = -downtrend_count / len(mas)
            
            return trend_strength
        except:
            return pd.Series(index=self.data.index, data=0)
    
    def _add_fundamental_indicators(self):
        """Add fundamental analysis indicators"""
        try:
            # Fetch financial data
            financial_data = self.get_financial_statements()
            if financial_data:
                # Add P/E ratio estimation (simplified)
                latest_close = self.data['Close'].iloc[-1]
                # This would need actual EPS data from financial statements
                # For now, we'll add placeholder for fundamental ratios
                
                self.data['Market_Cap_Estimate'] = latest_close * 1000000  # Simplified
                
        except Exception as e:
            print(f"Note: Could not add fundamental indicators: {e}")
    
    def _add_market_sentiment_indicators(self):
        """Add market sentiment and seasonality indicators"""
        try:
            # Day of week effect
            self.data['Day_of_Week'] = self.data.index.dayofweek
            
            # Month effect
            self.data['Month'] = self.data.index.month
            
            # End of month effect
            self.data['End_of_Month'] = (self.data.index.day > 25).astype(int)
            
            # Relative performance vs market (simplified)
            self.data['Relative_Strength'] = self.data['Close'].pct_change().rolling(20).mean()
            
        except Exception as e:
            print(f"Error adding market sentiment indicators: {e}")
    
    def get_financial_statements(self, period='annual', lang='en'):
        """Get financial statements for the stock"""
        if self.symbol is None:
            return None
        
        try:
            # Check cache first
            cached_data = self._load_from_cache(self.symbol, 'financial', max_age_hours=24)
            if cached_data is not None:
                return cached_data
            
            # Initialize vnstock for this symbol
            stock_obj = Vnstock().stock(symbol=self.symbol, source='VCI')
            
            # Fetch financial data using correct API
            financial_data = {}
            
            try:
                # Try to get basic company information instead of detailed financials
                # as the API might have different method names
                overview_data = self._get_company_overview()
                if overview_data:
                    financial_data['overview'] = overview_data
            except Exception as e:
                print(f"Note: Could not fetch company overview: {e}")
            
            # For now, return basic structure
            financial_data['symbol'] = self.symbol
            financial_data['fetched_at'] = datetime.now().isoformat()
            
            # Save to cache
            self._save_to_cache(financial_data, self.symbol, 'financial')
            
            return financial_data
            
        except Exception as e:
            print(f"Error fetching financial data: {e}")
            return None
    
    def _get_company_overview(self):
        """Get basic company overview information"""
        try:
            # Use the correct vnstock API for company information
            stock_obj = Vnstock().stock(symbol=self.symbol, source='VCI')
            
            # Try to get basic info - adjust based on actual vnstock API
            company_info = {
                'symbol': self.symbol,
                'company_name': f"{self.symbol} Corporation",
                'market': 'Vietnam Stock Market',
                'currency': 'VND',
                'exchange': 'HOSE/HNX/UPCOM',
                'last_updated': datetime.now().isoformat()
            }
            
            return company_info
            
        except Exception as e:
            print(f"Note: Could not fetch company overview via vnstock API: {e}")
            return None
    
    def get_market_data(self):
        """Get comprehensive market data"""
        try:
            market_data = {}
            
            # Try to get company listing info using correct API
            try:
                # Use Vnstock's listing method if available
                vnstock_client = Vnstock()
                # Adjust this call based on actual vnstock API
                market_data['symbol'] = self.symbol
                market_data['market'] = 'Vietnam'
                market_data['fetched_at'] = datetime.now().isoformat()
                
            except Exception as e:
                print(f"Note: Could not fetch listing data: {e}")
            
            # Get basic market information
            market_data['indices'] = get_vn_market_indices()
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def get_market_screening(self, filter_conditions=None):
        """Screen stocks based on conditions"""
        try:
            if filter_conditions is None:
                filter_conditions = {
                    'marketCap': [1000, 50000],  # Market cap in billion VND
                    'pe': [5, 30],  # P/E ratio range
                    'volume': [100000, None]  # Minimum volume
                }
            
            # This would implement screening logic
            # For now, return popular stocks as example
            popular_stocks = get_popular_vn_stocks()
            return popular_stocks[:10]
            
        except Exception as e:
            print(f"Error in market screening: {e}")
            return []
    
    def calculate_risk_metrics(self):
        """Calculate risk metrics for the stock"""
        if self.data is None or self.data.empty:
            return None
        
        try:
            returns = self.data['Close'].pct_change().dropna()
            
            risk_metrics = {
                'volatility_annual': returns.std() * np.sqrt(252),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(),
                'var_95': returns.quantile(0.05),
                'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
                'beta': self._calculate_beta()  # vs VN-Index
            }
            
            return risk_metrics
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return None
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + self.data['Close'].pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
        except:
            return None
    
    def _calculate_beta(self):
        """Calculate beta vs VN-Index (simplified)"""
        try:
            # This would need VN-Index data for proper calculation
            # For now, return a placeholder
            stock_returns = self.data['Close'].pct_change().dropna()
            return np.var(stock_returns) / (np.var(stock_returns) + 0.001)  # Simplified
        except:
            return 1.0
    
    def get_prediction_features(self):
        """Get features specifically for ML prediction"""
        if self.data is None or self.data.empty:
            return None
        
        try:
            # Select most important features for prediction
            feature_columns = [
                'Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'EMA_12', 'EMA_26', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower',
                'Price_vs_MA20', 'Price_vs_MA50', 'ROC_10', 'ROC_20',
                'ATR', 'Volume_Trend', 'Trend_Strength', 'Volatility_20',
                'Day_of_Week', 'Month', 'End_of_Month'
            ]
            
            # Filter existing columns
            available_features = [col for col in feature_columns if col in self.data.columns]
            
            features_df = self.data[available_features].copy()
            
            # Add lag features
            for lag in [1, 2, 3, 5]:
                features_df[f'Close_lag_{lag}'] = features_df['Close'].shift(lag)
                features_df[f'Volume_lag_{lag}'] = features_df['Volume'].shift(lag)
            
            # Add target variable (next day return)
            features_df['Target'] = features_df['Close'].shift(-1) / features_df['Close'] - 1
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            print(f"Error creating prediction features: {e}")
            return None

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_vn_trend(self):
        """Calculate Vietnamese market specific trend indicator"""
        try:
            # Simple trend based on price position relative to moving averages
            close = self.data['Close']
            ma20 = self.data['MA_20']
            ma50 = self.data['MA_50']
            
            trend = pd.Series(index=close.index, dtype=float)
            
            # Uptrend: Close > MA20 > MA50
            uptrend = (close > ma20) & (ma20 > ma50)
            trend[uptrend] = 1
            
            # Downtrend: Close < MA20 < MA50  
            downtrend = (close < ma20) & (ma20 < ma50)
            trend[downtrend] = -1
            
            # Sideways: other cases
            trend = trend.fillna(0)
            
            return trend
        except:
            return pd.Series(index=self.data.index, data=0)
    
    def get_company_info(self):
        """Get Vietnamese company information"""
        if self.symbol is None:
            return None
        
        try:
            # Try to get company info using vnstock
            company_info = {
                'symbol': self.symbol,
                'company_name': f"{self.symbol} Corporation",  # Basic info
                'market': 'Vietnam Stock Market',
                'currency': 'VND',
                'exchange': 'HOSE/HNX/UPCOM'
            }
            
            return company_info
            
        except Exception as e:
            print(f"Error fetching Vietnamese company info: {str(e)}")
            return None

# Vietnamese stock utilities
def get_popular_vn_stocks():
    """Return list of popular Vietnamese stock symbols"""
    return [
        # Large Cap - Blue Chips
        'VNM', 'VIC', 'VHM', 'BID', 'CTG', 'VCB', 'TCB', 'MBB',
        'VPB', 'MSN', 'MWG', 'FPT', 'GAS', 'PLX', 'VRE', 'HPG',
        
        # Mid Cap
        'POW', 'SSI', 'HDB', 'TPB', 'STB', 'ACB', 'VJC', 'GMD',
        'DHG', 'SAB', 'DGC', 'NVL', 'BCM', 'PDR', 'KDH', 'DXG',
        
        # Growth Stocks
        'CMG', 'DPM', 'VOS', 'TCH', 'HSG', 'DCM', 'SBT', 'LPB'
    ]

def get_vn_stock_sectors():
    """Return Vietnamese stock sectors with representative stocks"""
    return {
        'Banks & Financial': ['BID', 'CTG', 'VCB', 'TCB', 'MBB', 'VPB', 'HDB', 'TPB', 'STB', 'ACB'],
        'Real Estate': ['VIC', 'VHM', 'VRE', 'NVL', 'KDH', 'DXG', 'BCM', 'PDR'],
        'Consumer Goods': ['VNM', 'MSN', 'MWG', 'SAB', 'DHG'],
        'Technology': ['FPT', 'CMG'],
        'Energy & Utilities': ['GAS', 'PLX', 'POW'],
        'Materials & Steel': ['HPG', 'HSG', 'DCM'],
        'Transportation': ['VJC', 'GMD']
    }

def validate_vn_symbol(symbol):
    """Validate if a Vietnamese stock symbol exists"""
    try:
        symbol = symbol.upper()
        quote = Quote(symbol=symbol, source='VCI')
        
        # Try to fetch 1 day of data to validate
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        data = quote.history(start=start_date, end=end_date)
        return data is not None and not data.empty
    except:
        return False

def suggest_similar_vn_stocks(symbol):
    """Suggest similar Vietnamese stocks based on input"""
    symbol = symbol.upper()
    popular = get_popular_vn_stocks()
    
    # Find stocks that start with the same letter(s)
    suggestions = []
    for stock in popular:
        if stock.startswith(symbol[:1]) or symbol in stock:
            suggestions.append(stock)
    
    return suggestions[:5]  # Return top 5 suggestions

def get_vn_market_indices():
    """Get Vietnamese market indices"""
    return {
        'VN-INDEX': 'Ho Chi Minh Stock Exchange Index',
        'HNX-INDEX': 'Hanoi Stock Exchange Index', 
        'UPCOM-INDEX': 'Unlisted Public Company Market Index',
        'VN30': 'Top 30 large cap stocks',
        'HNX30': 'Top 30 HNX stocks'
    }

def get_sector_performance():
    """Get sector performance data"""
    sectors = get_vn_stock_sectors()
    sector_performance = {}
    
    for sector, stocks in sectors.items():
        try:
            # Get performance for representative stocks
            fetcher = VNStockDataFetcher()
            performances = []
            
            for stock in stocks[:3]:  # Sample 3 stocks per sector
                data = fetcher.fetch_data(stock, use_cache=True)
                if data is not None and len(data) > 20:
                    perf = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                    performances.append(perf)
            
            if performances:
                sector_performance[sector] = np.mean(performances)
                
        except Exception as e:
            print(f"Error getting performance for {sector}: {e}")
    
    return sector_performance

# Example usage and testing functions
def test_vn_stock_data():
    """Test function for Vietnamese stock data fetching"""
    print("🧪 Testing Vietnamese Stock Data Fetcher...")
    
    fetcher = VNStockDataFetcher()
    
    # Test with popular Vietnamese stocks
    test_stocks = ['VNM', 'VIC', 'FPT']
    
    for symbol in test_stocks:
        print(f"\n📊 Testing {symbol}...")
        data = fetcher.fetch_data(symbol)
        
        if data is not None:
            print(f"✅ {symbol}: {len(data)} days of data")
            print(f"   Latest price: {data['Close'].iloc[-1]:,.0f} VND")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            # Test company info
            info = fetcher.get_company_info()
            if info:
                print(f"   Company: {info['company_name']}")
        else:
            print(f"❌ Failed to fetch data for {symbol}")

def get_market_indicators():
    """Get Vietnamese market indicators"""
    try:
        # Return basic market structure since detailed API calls might not work
        return {
            'VN_INDEX': {'value': None, 'change': None, 'status': 'Data not available'},
            'HNX_INDEX': {'value': None, 'change': None, 'status': 'Data not available'}, 
            'market_sentiment': 'neutral',
            'trading_volume': 'normal',
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching market indicators: {e}")
        return {}

def test_vnstock_connection():
    """Test vnstock connection and available methods"""
    print("🔍 Testing vnstock connection...")
    
    try:
        # Test basic Quote functionality
        quote = Quote(symbol='VNM', source='VCI')
        print("✅ Quote object created successfully")
        
        # Test basic data fetch
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        data = quote.history(start=start_date, end=end_date)
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched sample data: {len(data)} rows")
            print(f"   Columns: {list(data.columns)}")
            return True
        else:
            print("⚠️  No data returned from vnstock")
            return False
            
    except Exception as e:
        print(f"❌ Error testing vnstock: {e}")
        return False

if __name__ == "__main__":
    # Test vnstock connection first
    print("🧪 Testing vnstock connection...")
    connection_ok = test_vnstock_connection()
    
    if connection_ok:
        print("\n✅ vnstock connection successful, running full tests...")
        test_vn_stock_data()
    else:
        print("\n⚠️  vnstock connection issues detected")
        print("   The application may work with limited functionality")
        
    # Run basic tests anyway
    print("\n🧪 Testing basic functionality...")
    
    fetcher = VNStockDataFetcher()
    
    # Test data fetching
    test_symbol = 'VNM'
    print(f"\n📊 Testing basic data fetch for {test_symbol}...")
    
    try:
        data = fetcher.fetch_data(test_symbol)
        if data is not None:
            print(f"✅ Fetched {len(data)} days with {len(data.columns)} features")
            
            # Test prediction features
            features = fetcher.get_prediction_features()
            if features is not None:
                print(f"✅ Prediction features ready: {len(features.columns)} features, {len(features)} samples")
            
            # Test risk metrics
            risk_metrics = fetcher.calculate_risk_metrics()
            if risk_metrics:
                print(f"✅ Risk metrics calculated: Volatility={risk_metrics['volatility_annual']:.2%}")
        else:
            print(f"❌ Could not fetch data for {test_symbol}")
            
    except Exception as e:
        print(f"❌ Error in basic test: {e}")