"""
Stock Data Fetcher Module
Provides utilities to fetch and preprocess stock market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockDataFetcher:
    """Class to fetch and preprocess stock market data"""
    
    def __init__(self):
        self.data = None
        self.symbol = None
    
    def fetch_data(self, symbol, period="5y", interval="1d"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            self.symbol = symbol.upper()
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period, interval=interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Clean column names
            self.data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Add basic technical indicators
            self._add_technical_indicators()
            
            return self.data
        
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _add_technical_indicators(self):
        """Add basic technical indicators to the data"""
        if self.data is None or self.data.empty:
            return
        
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
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_features_for_ml(self, lookback_days=30):
        """
        Prepare features for machine learning models
        
        Args:
            lookback_days (int): Number of days to look back for features
        
        Returns:
            tuple: (X, y) features and target variables
        """
        if self.data is None or self.data.empty:
            return None, None
        
        # Select feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Price_Change', 'Volatility'
        ]
        
        # Create sequences for time series prediction
        X, y = [], []
        data_values = self.data[feature_columns].dropna().values
        target_values = self.data['Price_Change_Next'].dropna().values
        
        for i in range(lookback_days, len(data_values) - 1):
            X.append(data_values[i-lookback_days:i])
            y.append(target_values[i])
        
        return np.array(X), np.array(y)
    
    def get_company_info(self):
        """Get company information"""
        if self.symbol is None:
            return None
        
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            return {
                'symbol': self.symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
        except Exception as e:
            print(f"Error fetching company info: {str(e)}")
            return None

# Utility functions
def get_popular_stocks():
    """Return list of popular stock symbols"""
    return [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
        'ORCL', 'IBM', 'INTC', 'AMD', 'UBER'
    ]

def validate_symbol(symbol):
    """Validate if a stock symbol exists"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return not data.empty
    except:
        return False