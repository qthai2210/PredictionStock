"""
Example Usage of Stock Prediction System
Demonstrates how to use different prediction models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.stock_data import StockDataFetcher
from src.models.ensemble_model import EnsembleStockPredictor
import pandas as pd

def example_basic_usage():
    """Basic example of fetching data and making predictions"""
    print("🔥 BASIC STOCK PREDICTION EXAMPLE")
    print("="*50)
    
    # Initialize data fetcher
    fetcher = StockDataFetcher()
    
    # Fetch Apple stock data
    print("📊 Fetching AAPL data...")
    data = fetcher.fetch_data('AAPL', period='1y')
    
    if data is not None:
        print(f"✅ Fetched {len(data)} days of data")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"💰 Current price: ${data['Close'].iloc[-1]:.2f}")
        
        # Initialize and train ensemble model
        print("\n🎯 Training ensemble model...")
        model = EnsembleStockPredictor()
        
        # Train model
        results = model.train(data)
        
        if results:
            print("✅ Model trained successfully!")
            
            # Make prediction
            prediction = model.predict_next_price(data)
            
            if prediction:
                print(f"\n🔮 PREDICTION RESULTS:")
                print(f"   Current Price: ${prediction['current_price']}")
                print(f"   Predicted Price: ${prediction['predicted_price']}")
                print(f"   Expected Change: ${prediction['price_change']:.2f} ({prediction['price_change_percent']:+.2f}%)")
                print(f"   Direction: {prediction['direction']} {'📈' if prediction['direction'] == 'UP' else '📉'}")
                print(f"   Confidence: {prediction['confidence']:.3f}")
        
        return data, model
    
    return None, None

def example_multiple_stocks():
    """Example of analyzing multiple stocks"""
    print("\n\n🚀 MULTIPLE STOCKS ANALYSIS")
    print("="*50)
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    fetcher = StockDataFetcher()
    
    results = {}
    
    for symbol in stocks:
        print(f"\n📊 Analyzing {symbol}...")
        
        try:
            # Fetch data
            data = fetcher.fetch_data(symbol, period='6mo')
            
            if data is not None:
                # Train model
                model = EnsembleStockPredictor()
                train_results = model.train(data)
                
                if train_results:
                    # Make prediction
                    prediction = model.predict_next_price(data)
                    
                    if prediction:
                        results[symbol] = {
                            'current_price': prediction['current_price'],
                            'predicted_price': prediction['predicted_price'],
                            'change_percent': prediction['price_change_percent'],
                            'direction': prediction['direction'],
                            'confidence': prediction['confidence']
                        }
                        
                        print(f"   Current: ${prediction['current_price']:.2f}")
                        print(f"   Predicted: ${prediction['predicted_price']:.2f} ({prediction['price_change_percent']:+.2f}%)")
                        print(f"   Direction: {prediction['direction']} {'📈' if prediction['direction'] == 'UP' else '📉'}")
        
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {str(e)}")
    
    # Summary
    if results:
        print(f"\n📊 SUMMARY OF {len(results)} STOCKS:")
        print("-" * 60)
        print(f"{'Stock':<8} {'Current':<10} {'Predicted':<10} {'Change%':<8} {'Direction':<10}")
        print("-" * 60)
        
        for symbol, data in results.items():
            direction_emoji = "📈" if data['direction'] == 'UP' else "📉"
            print(f"{symbol:<8} ${data['current_price']:<9.2f} ${data['predicted_price']:<9.2f} "
                  f"{data['change_percent']:>+6.2f}%  {direction_emoji} {data['direction']:<6}")

def example_company_analysis():
    """Example of detailed company analysis"""
    print("\n\n🏢 DETAILED COMPANY ANALYSIS")
    print("="*50)
    
    symbol = 'AAPL'
    fetcher = StockDataFetcher()
    
    # Fetch data and company info
    data = fetcher.fetch_data(symbol, period='1y')
    company_info = fetcher.get_company_info()
    
    if data is not None and company_info:
        print(f"📊 Analysis for {company_info['company_name']} ({symbol})")
        print(f"🏭 Sector: {company_info['sector']}")
        print(f"🏢 Industry: {company_info['industry']}")
        print(f"💰 Market Cap: {company_info['market_cap']:,}" if company_info['market_cap'] != 'N/A' else "💰 Market Cap: N/A")
        print(f"📊 P/E Ratio: {company_info['pe_ratio']}")
        
        # Technical analysis
        print(f"\n📈 Technical Analysis:")
        print(f"   Current Price: ${data['Close'].iloc[-1]:.2f}")
        print(f"   52-week High: ${data['Close'].max():.2f}")
        print(f"   52-week Low: ${data['Close'].min():.2f}")
        print(f"   RSI: {data['RSI'].iloc[-1]:.2f}")
        print(f"   MACD: {data['MACD'].iloc[-1]:.4f}")
        print(f"   Volatility: {data['Volatility'].iloc[-1]:.4f}")
        
        # Price changes
        daily_change = data['Price_Change'].iloc[-1] * 100
        weekly_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-7]) - 1) * 100
        monthly_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-30]) - 1) * 100
        
        print(f"\n📊 Recent Performance:")
        print(f"   Daily: {daily_change:+.2f}%")
        print(f"   Weekly: {weekly_change:+.2f}%")
        print(f"   Monthly: {monthly_change:+.2f}%")

def main():
    """Run all examples"""
    try:
        # Run basic example
        data, model = example_basic_usage()
        
        # Run multiple stocks example
        example_multiple_stocks()
        
        # Run company analysis example
        example_company_analysis()
        
        print(f"\n\n✅ All examples completed successfully!")
        print("💡 To run the full interactive application, execute: python src/main.py")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("💡 Make sure to install dependencies first: pip install -r requirements.txt")

if __name__ == "__main__":
    main()