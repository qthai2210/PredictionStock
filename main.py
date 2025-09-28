"""
Vietnamese Stock Market Analysis and Prediction Tool
Main application file to run stock data analysis and predictions
"""

import sys
import os

# Add src directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from utils.vn_stock_data import VNStockDataFetcher, get_popular_vn_stocks, get_vn_stock_sectors
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Try to import test function, create fallback if not available
    try:
        from utils.vn_stock_data import test_vnstock_connection
    except ImportError:
        def test_vnstock_connection():
            """Fallback test function"""
            try:
                from vnstock import Quote
                quote = Quote(symbol='VNM', source='VCI')
                return True
            except Exception as e:
                print(f"vnstock test failed: {e}")
                return False
    
    # Test vnstock connection at startup
    print("🔍 Testing vnstock connection...")
    vnstock_available = test_vnstock_connection()
    
    if not vnstock_available:
        print("⚠️  Warning: vnstock connection issues detected")
        print("   Some features may not work properly")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main function to run Vietnamese stock analysis"""
    print("\n🇻🇳 Vietnamese Stock Market Analysis Tool")
    print("=" * 50)
    
    # Initialize data fetcher
    try:
        fetcher = VNStockDataFetcher()
        print("✅ Data fetcher initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing data fetcher: {e}")
        return
    
    while True:
        print("\n📊 Choose an option:")
        print("1. Analyze a specific stock")
        print("2. Show popular Vietnamese stocks")
        print("3. Sector performance analysis")
        print("4. Get prediction features for a stock")
        print("5. Risk analysis for a stock")
        print("6. Test system functionality")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        try:
            if choice == '1':
                analyze_stock(fetcher)
            elif choice == '2':
                show_popular_stocks()
            elif choice == '3':
                sector_analysis()
            elif choice == '4':
                prediction_features_analysis(fetcher)
            elif choice == '5':
                risk_analysis(fetcher)
            elif choice == '6':
                test_system_functionality(fetcher)
            elif choice == '7':
                print("👋 Thank you for using Vietnamese Stock Analysis Tool!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or choose a different option.")

def test_system_functionality(fetcher):
    """Test system functionality"""
    print("\n🧪 System Functionality Test")
    print("-" * 30)
    
    # Test 1: Basic data fetching
    print("\n1️⃣ Testing basic data fetching...")
    try:
        test_data = fetcher.fetch_data('VNM', use_cache=False)
        if test_data is not None:
            print(f"   ✅ Data fetch successful: {len(test_data)} rows")
        else:
            print("   ❌ Data fetch failed")
    except Exception as e:
        print(f"   ❌ Error in data fetch: {e}")
    
    # Test 2: Technical indicators
    print("\n2️⃣ Testing technical indicators...")
    try:
        if hasattr(fetcher, 'data') and fetcher.data is not None:
            indicators = ['RSI', 'MACD', 'MA_20', 'BB_Upper']
            available = [ind for ind in indicators if ind in fetcher.data.columns]
            print(f"   ✅ Available indicators: {available}")
        else:
            print("   ⚠️  No data available for indicator test")
    except Exception as e:
        print(f"   ❌ Error checking indicators: {e}")
    
    # Test 3: Cache system
    print("\n3️⃣ Testing cache system...")
    try:
        cache_dir = fetcher.cache_dir
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            print(f"   ✅ Cache directory exists with {len(cache_files)} files")
        else:
            print("   ⚠️  Cache directory not found")
    except Exception as e:
        print(f"   ❌ Error checking cache: {e}")
    
    print("\n✅ System test completed!")

def analyze_stock(fetcher):
    """Analyze a specific Vietnamese stock with enhanced error handling"""
    print("\n📈 Stock Analysis")
    print("-" * 20)
    
    # Get stock symbol from user
    symbol = input("Enter Vietnamese stock symbol (e.g., VNM, VIC, FPT): ").strip().upper()
    
    if not symbol:
        print("❌ Please enter a valid symbol.")
        return
    
    print(f"\n🔍 Analyzing {symbol}...")
    
    try:
        # Fetch data with error handling
        data = fetcher.fetch_data(symbol)
        
        if data is None or data.empty:
            print(f"❌ Could not fetch data for {symbol}")
            print("   Possible reasons:")
            print("   - Invalid stock symbol")
            print("   - Network connectivity issues")
            print("   - vnstock API limitations")
            return
        
        # Display basic information
        print(f"\n✅ Successfully loaded data for {symbol}")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"📊 Total trading days: {len(data)}")
        print(f"🔢 Available features: {len(data.columns)}")
        
        # Current price info
        latest_data = data.iloc[-1]
        print(f"\n💰 Latest Price Information:")
        print(f"   Close Price: {latest_data['Close']:,.0f} VND")
        print(f"   Volume: {latest_data['Volume']:,.0f}")
        
        # Technical indicators (if available)
        if 'MA_20' in data.columns:
            print(f"   20-day MA: {latest_data['MA_20']:,.0f} VND")
        if 'RSI' in data.columns and not pd.isna(latest_data['RSI']):
            print(f"   RSI: {latest_data['RSI']:.1f}")
        
        # Performance metrics
        if len(data) >= 20:
            perf_1m = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
            print(f"   1-Month Performance: {perf_1m:+.2f}%")
        
        if len(data) >= 60:
            perf_3m = (data['Close'].iloc[-1] / data['Close'].iloc[-60] - 1) * 100
            print(f"   3-Month Performance: {perf_3m:+.2f}%")
        
        # Technical signals
        print(f"\n🔍 Technical Analysis:")
        try:
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                if not pd.isna(latest_data['MACD']) and not pd.isna(latest_data['MACD_Signal']):
                    macd_signal = "BUY" if latest_data['MACD'] > latest_data['MACD_Signal'] else "SELL"
                    print(f"   MACD Signal: {macd_signal}")
            
            if 'RSI' in data.columns and not pd.isna(latest_data['RSI']):
                if latest_data['RSI'] > 70:
                    rsi_signal = "OVERBOUGHT"
                elif latest_data['RSI'] < 30:
                    rsi_signal = "OVERSOLD"
                else:
                    rsi_signal = "NEUTRAL"
                print(f"   RSI Signal: {rsi_signal}")
        except Exception as e:
            print(f"   ⚠️  Could not calculate technical signals: {e}")
            
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")

def show_popular_stocks():
    """Display popular Vietnamese stocks by sector"""
    print("\n🌟 Popular Vietnamese Stocks")
    print("-" * 30)
    
    try:
        sectors = get_vn_stock_sectors()
        
        for sector, stocks in sectors.items():
            print(f"\n📊 {sector}:")
            for i, stock in enumerate(stocks[:5], 1):  # Show top 5 per sector
                print(f"   {i}. {stock}")
    except Exception as e:
        print(f"Error displaying popular stocks: {e}")

def sector_analysis():
    """Analyze sector performance"""
    print("\n📊 Sector Performance Analysis")
    print("-" * 35)
    print("⏳ Analyzing sectors... (this may take a moment)")
    
    try:
        from utils.vn_stock_data import get_sector_performance
        sector_perf = get_sector_performance()
        
        if sector_perf:
            print("\n📈 Sector Performance (20-day):")
            sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
            
            for i, (sector, performance) in enumerate(sorted_sectors, 1):
                emoji = "🟢" if performance > 0 else "🔴" if performance < -5 else "🟡"
                print(f"   {i}. {emoji} {sector}: {performance:+.2f}%")
        else:
            print("❌ Could not fetch sector performance data")
            
    except Exception as e:
        print(f"❌ Error in sector analysis: {e}")

def prediction_features_analysis(fetcher):
    """Analyze prediction features for a stock"""
    print("\n🤖 Prediction Features Analysis")
    print("-" * 35)
    
    symbol = input("Enter stock symbol for feature analysis: ").strip().upper()
    
    if not symbol:
        print("❌ Please enter a valid symbol.")
        return
    
    print(f"\n🔍 Preparing prediction features for {symbol}...")
    
    try:
        # Fetch data
        data = fetcher.fetch_data(symbol)
        
        if data is None:
            print(f"❌ Could not fetch data for {symbol}")
            return
        
        # Get prediction features
        features = fetcher.get_prediction_features()
        
        if features is None:
            print("❌ Could not generate prediction features")
            return
        
        print(f"\n✅ Generated prediction dataset:")
        print(f"   📊 Features: {len(features.columns)} columns")
        print(f"   📅 Samples: {len(features)} rows")
        print(f"   🎯 Target variable: Next day return")
        
        # Show feature importance (correlation with target)
        if 'Target' in features.columns:
            print(f"\n🔍 Top correlated features with next day return:")
            correlations = features.corr()['Target'].abs().sort_values(ascending=False)
            
            for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
                if feature != 'Target':
                    print(f"   {i}. {feature}: {corr:.3f}")
                    
    except Exception as e:
        print(f"❌ Error in prediction features analysis: {e}")

def risk_analysis(fetcher):
    """Perform risk analysis on a stock"""
    print("\n⚠️  Risk Analysis")
    print("-" * 20)
    
    symbol = input("Enter stock symbol for risk analysis: ").strip().upper()
    
    if not symbol:
        print("❌ Please enter a valid symbol.")
        return
    
    print(f"\n📊 Calculating risk metrics for {symbol}...")
    
    try:
        # Fetch data
        data = fetcher.fetch_data(symbol)
        
        if data is None:
            print(f"❌ Could not fetch data for {symbol}")
            return
        
        # Calculate risk metrics
        risk_metrics = fetcher.calculate_risk_metrics()
        
        if risk_metrics is None:
            print("❌ Could not calculate risk metrics")
            return
        
        print(f"\n📈 Risk Metrics for {symbol}:")
        print(f"   Annual Volatility: {risk_metrics['volatility_annual']:.2%}")
        print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"   Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"   VaR (95%): {risk_metrics['var_95']:.2%}")
        print(f"   CVaR (95%): {risk_metrics['cvar_95']:.2%}")
        print(f"   Beta (estimated): {risk_metrics['beta']:.2f}")
        
        # Risk interpretation
        print(f"\n🔍 Risk Assessment:")
        if risk_metrics['volatility_annual'] > 0.3:
            print("   ⚠️  HIGH RISK: High volatility stock")
        elif risk_metrics['volatility_annual'] > 0.2:
            print("   🟡 MEDIUM RISK: Moderate volatility")
        else:
            print("   🟢 LOW RISK: Low volatility stock")
            
    except Exception as e:
        print(f"❌ Error in risk analysis: {e}")

def quick_test():
    """Quick test function to verify the system works"""
    print("\n🧪 Quick System Test")
    print("-" * 20)
    
    fetcher = VNStockDataFetcher()
    test_symbols = ['VNM', 'FPT']
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        data = fetcher.fetch_data(symbol)
        
        if data is not None and len(data) > 0:
            print(f"   ✅ {symbol}: {len(data)} days, latest price: {data['Close'].iloc[-1]:,.0f} VND")
        else:
            print(f"   ❌ {symbol}: Failed to fetch data")

if __name__ == "__main__":
    main()
