# Stock Prediction Configuration
# Default settings for the stock prediction application

# Model Configuration
MODELS = {
    'lstm': {
        'lookback_days': 60,
        'lstm_units': 50,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32
    },
    'prophet': {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05
    },
    'ensemble': {
        'test_size': 0.2,
        'random_state': 42
    }
}

# Data Configuration
DATA_CONFIG = {
    'default_period': '2y',
    'default_interval': '1d',
    'min_data_points': 50,
    'feature_columns': [
        'Open', 'High', 'Low', 'Volume',
        'MA_5', 'MA_10', 'MA_20', 'MA_50',
        'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
        'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Price_Change', 'Volatility'
    ]
}

# Popular stocks for quick selection
POPULAR_STOCKS = [
    # Technology
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    # Financial
    'BRK-B', 'JPM', 'BAC', 'WFC', 'GS',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
    # Consumer
    'KO', 'PEP', 'WMT', 'HD', 'MCD',
    # Industrial
    'BA', 'CAT', 'GE', 'MMM', 'UPS'
]

# Vietnamese stocks (if using VN market)
VN_STOCKS = [
    'VNM', 'VIC', 'VHM', 'BID', 'CTG', 'VCB', 'GAS', 'MSN', 'PLX', 'POW'
]

# Prediction settings
PREDICTION_CONFIG = {
    'default_prediction_days': 7,
    'max_prediction_days': 30,
    'confidence_threshold': 0.7,
    'risk_levels': {
        'low': 0.02,    # 2% volatility
        'medium': 0.05, # 5% volatility
        'high': 0.10    # 10% volatility
    }
}

# Display settings
DISPLAY_CONFIG = {
    'currency_symbol': '$',
    'decimal_places': 2,
    'percentage_places': 2,
    'date_format': '%Y-%m-%d',
    'chart_style': 'seaborn',
    'figure_size': (12, 8)
}