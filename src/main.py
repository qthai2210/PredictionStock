"""
Advanced Stock Price Prediction Application
Combines multiple ML models for accurate stock forecasting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.stock_data import StockDataFetcher, get_popular_stocks, validate_symbol, get_stock_market_info
from utils.vn_stock_data import get_popular_vn_stocks, get_vn_stock_sectors
from models.lstm_model import LSTMStockPredictor
from models.prophet_model import ProphetStockPredictor
from models.ensemble_model import EnsembleStockPredictor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictionApp:
    """Main application class for stock prediction"""
    
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.current_data = None
        self.current_symbol = None
        
    def display_menu(self):
        """Display main menu options"""
        print("\n" + "="*60)
        print("🚀 ADVANCED STOCK PRICE PREDICTION SYSTEM 🚀")
        print("🌍 International & 🇻🇳 Vietnamese Stocks")
        print("="*60)
        print("1. 📊 Fetch Stock Data")
        print("2. 🧠 Train LSTM Model")
        print("3. 📈 Train Prophet Model")
        print("4. 🎯 Train Ensemble Model")
        print("5. 🔮 Make Predictions")
        print("6. 📋 View Popular Stocks")
        print("7. 🇻🇳 Vietnamese Stock Sectors")
        print("8. 📊 Show Data Statistics")
        print("9. 💾 Export Results")
        print("10. ❌ Exit")
        print("="*60)
    
    def fetch_stock_data(self):
        """Fetch stock data from multiple sources (International + Vietnamese)"""
        print("\n📊 UNIVERSAL STOCK DATA FETCHER")
        print("-" * 40)
        
        # Show popular stocks from both markets
        print("🌍 Popular International: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA")
        print("🇻🇳 Popular Vietnamese: VNM, VIC, VHM, BID, CTG, VCB, FPT, MWG")
        
        symbol = input("\nEnter stock symbol (e.g., AAPL, VNM, FPT): ").strip().upper()
        
        if not symbol:
            print("❌ Invalid symbol!")
            return
        
        # Validate symbol
        if not validate_symbol(symbol):
            print(f"❌ Symbol '{symbol}' not found!")
            return
        
        period = input("Enter period (1y, 2y, 5y, max) [default: 2y]: ").strip() or "2y"
        
        print(f"\n🔄 Fetching data for {symbol}...")
        
        data = self.data_fetcher.fetch_data(symbol, period=period)
        
        if data is not None:
            self.current_data = data
            self.current_symbol = symbol
            
            print(f"✅ Successfully fetched {len(data)} days of data!")
            print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            # Show price with appropriate currency
            if self.data_fetcher.is_vietnamese_stock:
                print(f"💰 Current price: {data['Close'].iloc[-1]:,.0f} VND")
            else:
                print(f"💰 Current price: ${data['Close'].iloc[-1]:.2f}")
            
            # Show company info
            company_info = self.data_fetcher.get_company_info()
            if company_info:
                print(f"🏢 Company: {company_info['company_name']}")
                if 'sector' in company_info:
                    print(f"🏭 Sector: {company_info['sector']}")
                if 'market' in company_info:
                    print(f"🌍 Market: {company_info['market']}")
        else:
            print("❌ Failed to fetch data!")
    
    def train_lstm_model(self):
        """Train LSTM neural network model"""
        if self.current_data is None:
            print("❌ Please fetch stock data first!")
            return
        
        print(f"\n🧠 TRAINING LSTM MODEL FOR {self.current_symbol}")
        print("-" * 40)
        
        try:
            # Initialize LSTM model
            lstm_model = LSTMStockPredictor(lookback_days=60, lstm_units=50)
            
            print("🔄 Training LSTM model... (this may take a few minutes)")
            
            # Train model
            history = lstm_model.train(self.current_data, epochs=50, batch_size=32)
            
            if history:
                print("✅ LSTM model trained successfully!")
                
                # Evaluate model
                evaluation = lstm_model.evaluate(self.current_data)
                if evaluation:
                    print(f"📊 Model Performance:")
                    print(f"   R² Score: {evaluation['r2_score']:.4f}")
                    print(f"   RMSE: ${evaluation['rmse']:.2f}")
                    print(f"   Accuracy: {evaluation['accuracy']:.2f}%")
                
                # Make predictions
                predictions = lstm_model.predict(self.current_data, days_ahead=7)
                if predictions is not None:
                    print(f"\n🔮 7-Day Predictions:")
                    current_price = self.current_data['Close'].iloc[-1]
                    for i, pred in enumerate(predictions, 1):
                        change = pred - current_price
                        change_pct = (change / current_price) * 100
                        direction = "📈" if change > 0 else "📉"
                        print(f"   Day {i}: ${pred:.2f} ({direction} {change_pct:+.2f}%)")
            
        except Exception as e:
            print(f"❌ Error training LSTM model: {str(e)}")
    
    def train_prophet_model(self):
        """Train Prophet time series model"""
        if self.current_data is None:
            print("❌ Please fetch stock data first!")
            return
        
        print(f"\n📈 TRAINING PROPHET MODEL FOR {self.current_symbol}")
        print("-" * 40)
        
        try:
            # Initialize Prophet model
            prophet_model = ProphetStockPredictor()
            
            print("🔄 Training Prophet model...")
            
            # Train model
            success = prophet_model.train(self.current_data)
            
            if success:
                print("✅ Prophet model trained successfully!")
                
                # Make predictions
                predictions = prophet_model.predict_next_days(days=7)
                
                if predictions:
                    print(f"\n🔮 7-Day Predictions:")
                    for date, pred in predictions.items():
                        print(f"   {date}: ${pred['predicted_price']:.2f} "
                              f"(${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f})")
                
                # Evaluate model
                evaluation = prophet_model.evaluate(self.current_data)
                if evaluation:
                    print(f"\n📊 Model Performance:")
                    print(f"   MAPE: {evaluation['mape']:.4f}")
                    print(f"   RMSE: ${evaluation['rmse']:.2f}")
                    print(f"   Accuracy: {evaluation['accuracy']:.2f}%")
            
        except Exception as e:
            print(f"❌ Error training Prophet model: {str(e)}")
    
    def train_ensemble_model(self):
        """Train ensemble model combining multiple algorithms"""
        if self.current_data is None:
            print("❌ Please fetch stock data first!")
            return
        
        print(f"\n🎯 TRAINING ENSEMBLE MODEL FOR {self.current_symbol}")
        print("-" * 40)
        
        try:
            # Initialize ensemble model
            ensemble_model = EnsembleStockPredictor()
            
            print("🔄 Training ensemble model...")
            
            # Train model
            results = ensemble_model.train(self.current_data)
            
            if results:
                print("✅ Ensemble model trained successfully!")
                
                print(f"\n📊 Individual Model Performance:")
                for model_name, metrics in results.items():
                    if metrics:
                        print(f"   {model_name}:")
                        print(f"     R² Score: {metrics['r2_score']:.4f}")
                        print(f"     RMSE: ${metrics['rmse']:.2f}")
                        print(f"     Accuracy: {metrics['accuracy']:.2f}%")
                
                # Make prediction
                prediction = ensemble_model.predict_next_price(self.current_data)
                
                if prediction:
                    print(f"\n🔮 Next Day Prediction:")
                    print(f"   Current Price: ${prediction['current_price']:.2f}")
                    print(f"   Predicted Price: ${prediction['predicted_price']:.2f}")
                    print(f"   Change: ${prediction['price_change']:.2f} ({prediction['price_change_percent']:+.2f}%)")
                    print(f"   Direction: {prediction['direction']} {('📈' if prediction['direction'] == 'UP' else '📉')}")
                    print(f"   Confidence: {prediction['confidence']:.3f}")
            
        except Exception as e:
            print(f"❌ Error training ensemble model: {str(e)}")
    
    def make_predictions(self):
        """Make predictions using available models"""
        if self.current_data is None:
            print("❌ Please fetch stock data first!")
            return
        
        print(f"\n🔮 MAKING PREDICTIONS FOR {self.current_symbol}")
        print("-" * 40)
        
        print("Select prediction model:")
        print("1. LSTM Neural Network")
        print("2. Prophet Time Series")
        print("3. Ensemble Model")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            self.train_lstm_model()
        elif choice == "2":
            self.train_prophet_model()
        elif choice == "3":
            self.train_ensemble_model()
        else:
            print("❌ Invalid choice!")
    
    def view_popular_stocks(self):
        """Display popular stock symbols from both markets"""
        print("\n📋 POPULAR STOCK SYMBOLS")
        print("-" * 40)
        
        # International stocks
        print("🌍 INTERNATIONAL STOCKS:")
        international = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM']
        for i in range(0, len(international), 5):
            row = international[i:i+5]
            print("   " + "  ".join(f"{stock:>6}" for stock in row))
        
        print("\n🇻🇳 VIETNAMESE STOCKS:")
        vn_stocks = get_popular_vn_stocks()[:15]  # Top 15
        for i in range(0, len(vn_stocks), 5):
            row = vn_stocks[i:i+5]
            print("   " + "  ".join(f"{stock:>6}" for stock in row))
    
    def view_vn_stock_sectors(self):
        """Display Vietnamese stock sectors"""
        print("\n🇻🇳 VIETNAMESE STOCK SECTORS")
        print("-" * 40)
        
        sectors = get_vn_stock_sectors()
        
        for sector, stocks in sectors.items():
            print(f"\n📊 {sector}:")
            # Display stocks in rows of 5
            for i in range(0, len(stocks), 5):
                row = stocks[i:i+5]
                print("   " + "  ".join(f"{stock:>6}" for stock in row))
    
    def show_data_statistics(self):
        """Show statistics of current data with appropriate currency formatting"""
        if self.current_data is None:
            print("❌ Please fetch stock data first!")
            return
        
        print(f"\n📊 DATA STATISTICS FOR {self.current_symbol}")
        print("-" * 40)
        
        data = self.current_data
        is_vn = self.data_fetcher.is_vietnamese_stock
        
        print(f"📅 Data Period: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"📊 Total Days: {len(data)}")
        
        # Format prices based on market
        if is_vn:
            print(f"💰 Price Range: {data['Close'].min():,.0f} - {data['Close'].max():,.0f} VND")
            print(f"📈 Current Price: {data['Close'].iloc[-1]:,.0f} VND")
        else:
            print(f"💰 Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"📈 Current Price: ${data['Close'].iloc[-1]:.2f}")
            
        print(f"📊 Average Volume: {data['Volume'].mean():,.0f}")
        print(f"📉 Volatility (20d): {data['Volatility'].iloc[-1]:.4f}")
        
        # Price changes
        daily_change = data['Price_Change'].iloc[-1] * 100
        weekly_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-7]) - 1) * 100
        monthly_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-30]) - 1) * 100
        
        print(f"\n📈 Recent Performance:")
        print(f"   Daily Change: {daily_change:+.2f}%")
        print(f"   Weekly Change: {weekly_change:+.2f}%")
        print(f"   Monthly Change: {monthly_change:+.2f}%")
        
        # Technical indicators
        print(f"\n📊 Technical Indicators:")
        print(f"   RSI: {data['RSI'].iloc[-1]:.2f}")
        print(f"   MACD: {data['MACD'].iloc[-1]:.4f}")
        
        if is_vn:
            print(f"   MA(20): {data['MA_20'].iloc[-1]:,.0f} VND")
            print(f"   MA(50): {data['MA_50'].iloc[-1]:,.0f} VND")
        else:
            print(f"   MA(20): ${data['MA_20'].iloc[-1]:.2f}")
            print(f"   MA(50): ${data['MA_50'].iloc[-1]:.2f}")
        
        # Vietnamese-specific info
        if is_vn:
            print(f"\n🇻🇳 Vietnamese Market Info:")
            print(f"   Market: Vietnam Stock Exchange")
            print(f"   Currency: Vietnamese Dong (VND)")
            print(f"   Trading Hours: 9:00 - 15:00 (GMT+7)")
    
    def export_results(self):
        """Export data and predictions to CSV"""
        if self.current_data is None:
            print("❌ Please fetch stock data first!")
            return
        
        try:
            filename = f"{self.current_symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv"
            self.current_data.to_csv(filename)
            print(f"✅ Data exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting data: {str(e)}")
    
    def run(self):
        """Main application loop"""
        print("🚀 Welcome to Advanced Stock Prediction System!")
        
        while True:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (1-9): ").strip()
                
                if choice == "1":
                    self.fetch_stock_data()
                elif choice == "2":
                    self.train_lstm_model()
                elif choice == "3":
                    self.train_prophet_model()
                elif choice == "4":
                    self.train_ensemble_model()
                elif choice == "5":
                    self.make_predictions()
                elif choice == "6":
                    self.view_popular_stocks()
                elif choice == "7":
                    self.view_vn_stock_sectors()
                elif choice == "8":
                    self.show_data_statistics()
                elif choice == "9":
                    self.export_results()
                elif choice == "10":
                    print("\n👋 Thank you for using Stock Prediction System!")
                    break
                else:
                    print("❌ Invalid choice! Please enter 1-10.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        app = StockPredictionApp()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main()