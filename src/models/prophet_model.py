"""
Prophet Time Series Model for Stock Price Prediction
Using Facebook Prophet for time series forecasting
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import warnings
warnings.filterwarnings('ignore')

class ProphetStockPredictor:
    """Prophet Time Series Model for Stock Price Prediction"""
    
    def __init__(self, 
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=False,
                 changepoint_prior_scale=0.05):
        """
        Initialize Prophet model
        
        Args:
            yearly_seasonality (bool): Include yearly seasonality
            weekly_seasonality (bool): Include weekly seasonality  
            daily_seasonality (bool): Include daily seasonality
            changepoint_prior_scale (float): Flexibility of trend changes
        """
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )
        self.is_trained = False
        self.last_date = None
        
    def prepare_data(self, data, target_column='Close'):
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            data (pd.DataFrame): Input data with DatetimeIndex
            target_column (str): Column to predict
            
        Returns:
            pd.DataFrame: Data formatted for Prophet
        """
        df = pd.DataFrame()
        df['ds'] = data.index
        df['y'] = data[target_column].values
        df = df.dropna()
        
        return df
    
    def train(self, data, target_column='Close'):
        """
        Train the Prophet model
        
        Args:
            data (pd.DataFrame): Training data with DatetimeIndex
            target_column (str): Column to predict
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare data
            prophet_data = self.prepare_data(data, target_column)
            self.last_date = prophet_data['ds'].max()
            
            # Add custom regressors for better stock prediction
            self._add_regressors(data, prophet_data)
            
            # Fit model
            self.model.fit(prophet_data)
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"Error training Prophet model: {str(e)}")
            return False
    
    def _add_regressors(self, original_data, prophet_data):
        """Add additional regressors to improve prediction"""
        try:
            # Add volume as regressor if available
            if 'Volume' in original_data.columns:
                self.model.add_regressor('volume')
                prophet_data['volume'] = original_data['Volume'].values[:len(prophet_data)]
            
            # Add volatility as regressor
            if 'Close' in original_data.columns:
                volatility = original_data['Close'].pct_change().rolling(window=20).std()
                self.model.add_regressor('volatility')
                prophet_data['volatility'] = volatility.values[:len(prophet_data)]
                
        except Exception as e:
            print(f"Warning: Could not add regressors: {str(e)}")
    
    def predict(self, periods=30, freq='D', include_history=False):
        """
        Make future predictions
        
        Args:
            periods (int): Number of periods to predict
            freq (str): Frequency of predictions ('D' for daily)
            include_history (bool): Include historical predictions
            
        Returns:
            pd.DataFrame: Predictions with confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
            
            # Make predictions
            forecast = self.model.predict(future)
            
            if not include_history:
                # Return only future predictions
                forecast = forecast.tail(periods)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def predict_next_days(self, days=7):
        """
        Predict next N days
        
        Args:
            days (int): Number of days to predict
            
        Returns:
            dict: Predictions with dates and values
        """
        forecast = self.predict(periods=days, include_history=False)
        
        if forecast is not None:
            predictions = {}
            for _, row in forecast.iterrows():
                date = row['ds'].strftime('%Y-%m-%d')
                predictions[date] = {
                    'predicted_price': round(row['yhat'], 2),
                    'lower_bound': round(row['yhat_lower'], 2),
                    'upper_bound': round(row['yhat_upper'], 2)
                }
            return predictions
        
        return None
    
    def evaluate(self, data, target_column='Close', test_periods=30):
        """
        Evaluate model performance using cross-validation
        
        Args:
            data (pd.DataFrame): Test data
            target_column (str): Column to predict
            test_periods (int): Number of periods to test
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Perform cross-validation
            df_cv = cross_validation(
                self.model, 
                initial='365 days', 
                period='180 days', 
                horizon=f'{test_periods} days'
            )
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Calculate accuracy metrics
            mae = df_p['mae'].mean()
            mape = df_p['mape'].mean()
            rmse = df_p['rmse'].mean()
            
            return {
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'accuracy': max(0, (1 - mape) * 100)  # Convert MAPE to accuracy percentage
            }
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None
    
    def plot_forecast(self, forecast=None, periods=30):
        """
        Plot forecast results
        
        Args:
            forecast (pd.DataFrame): Forecast data (if None, will generate new forecast)
            periods (int): Number of periods to forecast if generating new
            
        Returns:
            plotly figure object
        """
        try:
            if forecast is None:
                forecast = self.predict(periods=periods, include_history=True)
            
            # Create plotly figure
            fig = plot_plotly(self.model, forecast)
            fig.update_layout(
                title='Stock Price Prediction using Prophet',
                xaxis_title='Date',
                yaxis_title='Price ($)'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error plotting forecast: {str(e)}")
            return None
    
    def get_trend_analysis(self):
        """
        Analyze trend components
        
        Returns:
            dict: Trend analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before trend analysis")
        
        try:
            # Get trend changepoints
            changepoints = self.model.changepoints
            changepoint_effects = self.model.params['delta'].mean()
            
            return {
                'changepoints': changepoints.tolist(),
                'trend_flexibility': changepoint_effects,
                'seasonality_strength': self.model.params.get('beta', [0])[0] if 'beta' in self.model.params else 0
            }
            
        except Exception as e:
            print(f"Error analyzing trend: {str(e)}")
            return None