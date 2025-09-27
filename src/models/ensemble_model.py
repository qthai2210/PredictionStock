"""
Ensemble Model combining multiple prediction algorithms
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnsembleStockPredictor:
    """Ensemble model combining multiple ML algorithms"""
    
    def __init__(self, models=None):
        """
        Initialize ensemble model
        
        Args:
            models (list): List of models to use in ensemble
        """
        if models is None:
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
        else:
            self.models = models
            
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_weights = {}
        
    def prepare_features(self, data, target_column='Close', feature_columns=None):
        """
        Prepare features for training
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            feature_columns (list): List of feature columns to use
            
        Returns:
            tuple: (X, y) features and target
        """
        if feature_columns is None:
            # Default feature columns
            feature_columns = [
                'Open', 'High', 'Low', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'RSI', 'MACD', 'Price_Change', 'Volatility'
            ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if len(available_columns) == 0:
            raise ValueError("No feature columns available in data")
        
        X = data[available_columns].copy()
        y = data[target_column].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train(self, data, target_column='Close', test_size=0.2, feature_columns=None):
        """
        Train all models in the ensemble
        
        Args:
            data (pd.DataFrame): Training data
            target_column (str): Target column name
            test_size (float): Fraction of data to use for testing
            feature_columns (list): Feature columns to use
            
        Returns:
            dict: Training results for each model
        """
        try:
            # Prepare features
            X, y = self.prepare_features(data, target_column, feature_columns)
            
            if len(X) < 50:
                raise ValueError("Insufficient data for training (need at least 50 samples)")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            model_scores = {}
            
            # Train each model
            for name, model in self.models.items():
                try:
                    print(f"Training {name}...")
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate on test set
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    model_scores[name] = r2  # Use R² for weighting
                    
                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': np.sqrt(mse),
                        'r2_score': r2,
                        'accuracy': max(0, r2 * 100)
                    }
                    
                    print(f"{name}: R² = {r2:.4f}, RMSE = {np.sqrt(mse):.4f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    results[name] = None
            
            # Calculate model weights based on performance
            self._calculate_weights(model_scores)
            
            self.is_trained = True
            return results
            
        except Exception as e:
            print(f"Error training ensemble: {str(e)}")
            return None
    
    def _calculate_weights(self, model_scores):
        """Calculate weights for ensemble based on model performance"""
        # Convert negative R² scores to small positive values
        adjusted_scores = {}
        for name, score in model_scores.items():
            adjusted_scores[name] = max(0.01, score)  # Minimum weight of 0.01
        
        # Normalize weights to sum to 1
        total_score = sum(adjusted_scores.values())
        self.model_weights = {name: score/total_score for name, score in adjusted_scores.items()}
        
        print("Model weights:")
        for name, weight in self.model_weights.items():
            print(f"  {name}: {weight:.3f}")
    
    def predict(self, data, target_column='Close', feature_columns=None):
        """
        Make predictions using ensemble of models
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name (for feature preparation)
            feature_columns (list): Feature columns to use
            
        Returns:
            np.array: Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features (use last available target value if needed)
            X, _ = self.prepare_features(data, target_column, feature_columns)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from each model
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                    weights.append(self.model_weights.get(name, 0))
                except Exception as e:
                    print(f"Error predicting with {name}: {str(e)}")
                    continue
            
            if len(predictions) == 0:
                raise ValueError("No models could make predictions")
            
            # Weighted ensemble prediction
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            return ensemble_pred
            
        except Exception as e:
            print(f"Error making ensemble predictions: {str(e)}")
            return None
    
    def predict_next_price(self, data, target_column='Close', feature_columns=None):
        """
        Predict next day's price
        
        Args:
            data (pd.DataFrame): Historical data
            target_column (str): Target column name
            feature_columns (list): Feature columns to use
            
        Returns:
            dict: Prediction results
        """
        try:
            # Use last row for prediction
            last_data = data.tail(1)
            
            prediction = self.predict(last_data, target_column, feature_columns)
            
            if prediction is not None and len(prediction) > 0:
                current_price = data[target_column].iloc[-1]
                predicted_price = prediction[0]
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                return {
                    'current_price': round(current_price, 2),
                    'predicted_price': round(predicted_price, 2),
                    'price_change': round(price_change, 2),
                    'price_change_percent': round(price_change_pct, 2),
                    'direction': 'UP' if price_change > 0 else 'DOWN',
                    'confidence': self._calculate_confidence()
                }
            
            return None
            
        except Exception as e:
            print(f"Error predicting next price: {str(e)}")
            return None
    
    def _calculate_confidence(self):
        """Calculate prediction confidence based on model weights distribution"""
        if not self.model_weights:
            return 0.5
        
        # Higher confidence when weights are more evenly distributed
        weights = list(self.model_weights.values())
        entropy = -sum(w * np.log(w + 1e-10) for w in weights)
        max_entropy = np.log(len(weights))
        confidence = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return round(confidence, 3)
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        if 'random_forest' in self.models and self.is_trained:
            try:
                rf_model = self.models['random_forest']
                if hasattr(rf_model, 'feature_importances_'):
                    return rf_model.feature_importances_
            except:
                pass
        return None
    
    def evaluate(self, data, target_column='Close', feature_columns=None):
        """
        Evaluate ensemble model performance
        
        Args:
            data (pd.DataFrame): Test data
            target_column (str): Target column name
            feature_columns (list): Feature columns to use
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            X, y_true = self.prepare_features(data, target_column, feature_columns)
            y_pred = self.predict(data, target_column, feature_columns)
            
            if y_pred is None or len(y_pred) != len(y_true):
                return None
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'accuracy': max(0, r2 * 100),
                'model_weights': self.model_weights
            }
            
        except Exception as e:
            print(f"Error evaluating ensemble: {str(e)}")
            return None