"""
LSTM Neural Network Model for Stock Price Prediction
Using TensorFlow/Keras for deep learning time series forecasting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class LSTMStockPredictor:
    """LSTM Neural Network for Stock Price Prediction"""
    
    def __init__(self, lookback_days=60, lstm_units=50, dropout_rate=0.2):
        """
        Initialize LSTM model
        
        Args:
            lookback_days (int): Number of previous days to use for prediction
            lstm_units (int): Number of LSTM units in each layer
            dropout_rate (float): Dropout rate for regularization
        """
        self.lookback_days = lookback_days
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def _prepare_data(self, data, target_column='Close'):
        """Prepare data for LSTM training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_column]])
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data, target_column='Close', validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            data (pd.DataFrame): Training data
            target_column (str): Column to predict
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        
        Returns:
            dict: Training history
        """
        try:
            # Prepare data
            X, y = self._prepare_data(data, target_column)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build model
            self.model = self.build_model((X.shape[1], 1))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            self.is_trained = True
            
            return {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            }
            
        except Exception as e:
            print(f"Error training LSTM model: {str(e)}")
            return None
    
    def predict(self, data, target_column='Close', days_ahead=1):
        """
        Make predictions using the trained model
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Column to predict
            days_ahead (int): Number of days to predict ahead
        
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Use last lookback_days for prediction
            last_data = data[target_column].tail(self.lookback_days).values
            last_data_scaled = self.scaler.transform(last_data.reshape(-1, 1))
            
            predictions = []
            current_input = last_data_scaled.reshape(1, self.lookback_days, 1)
            
            for _ in range(days_ahead):
                pred = self.model.predict(current_input, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update input for next prediction
                current_input = np.append(current_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def evaluate(self, data, target_column='Close'):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            X, y_true = self._prepare_data(data, target_column)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Make predictions
            y_pred = self.model.predict(X, verbose=0)
            
            # Inverse transform
            y_true_inv = self.scaler.inverse_transform(y_true.reshape(-1, 1))
            y_pred_inv = self.scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            mse = mean_squared_error(y_true_inv, y_pred_inv)
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_inv, y_pred_inv)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'accuracy': max(0, r2 * 100)  # Convert R² to percentage
            }
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True