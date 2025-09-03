"""
Prediction Interval Estimation for House Price Predictions
Provides uncertainty quantification for business decision making
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PredictionIntervalEstimator:
    """
    Provides prediction intervals using quantile regression and ensemble methods
    """
    
    def __init__(self, base_model, confidence_level=0.9):
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.lower_quantile = self.alpha / 2
        self.upper_quantile = 1 - self.alpha / 2
        
        # Quantile models
        self.lower_model = None
        self.upper_model = None
        self.residual_std = None
        
    def fit_quantile_models(self, X_train, y_train, X_val, y_val):
        """
        Fit quantile regression models for prediction intervals
        """
        print(f'Training quantile models for {self.confidence_level*100}% prediction intervals...')
        
        # Train quantile regression models using Gradient Boosting
        self.lower_model = GradientBoostingRegressor(
            loss='quantile', alpha=self.lower_quantile,
            n_estimators=100, max_depth=6, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        
        self.upper_model = GradientBoostingRegressor(
            loss='quantile', alpha=self.upper_quantile,
            n_estimators=100, max_depth=6, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        
        # Fit models
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)
        
        # Calculate residual standard deviation for alternative method
        base_pred = self.base_model.predict(X_val)
        residuals = y_val - base_pred
        self.residual_std = np.std(residuals)
        
        print('Quantile models trained successfully')
        
    def predict_with_intervals(self, X):
        """
        Generate predictions with confidence intervals
        """
        # Base prediction
        y_pred = self.base_model.predict(X)
        
        # Quantile predictions
        y_lower = self.lower_model.predict(X)
        y_upper = self.upper_model.predict(X)
        
        # Alternative method using residual distribution (for comparison)
        z_score = stats.norm.ppf(1 - self.alpha/2)
        y_lower_alt = y_pred - z_score * self.residual_std
        y_upper_alt = y_pred + z_score * self.residual_std
        
        return {
            'prediction': y_pred,
            'lower_bound': y_lower,
            'upper_bound': y_upper,
            'lower_bound_alt': y_lower_alt,
            'upper_bound_alt': y_upper_alt,
            'interval_width': y_upper - y_lower,
            'confidence_level': self.confidence_level
        }
    
    def evaluate_interval_coverage(self, X_test, y_test):
        """
        Evaluate prediction interval coverage and width
        """
        intervals = self.predict_with_intervals(X_test)
        
        # Coverage analysis
        within_bounds = ((y_test >= intervals['lower_bound']) & 
                        (y_test <= intervals['upper_bound']))
        coverage_rate = within_bounds.mean()
        
        # Alternative method coverage
        within_bounds_alt = ((y_test >= intervals['lower_bound_alt']) & 
                           (y_test <= intervals['upper_bound_alt']))
        coverage_rate_alt = within_bounds_alt.mean()
        
        # Width analysis
        avg_width = np.mean(intervals['interval_width'])
        median_width = np.median(intervals['interval_width'])
        
        results = {
            'coverage_rate': coverage_rate,
            'coverage_rate_alt': coverage_rate_alt,
            'target_coverage': self.confidence_level,
            'avg_interval_width': avg_width,
            'median_interval_width': median_width,
            'width_std': np.std(intervals['interval_width'])
        }
        
        print(f'\\n[PREDICTION INTERVALS] Coverage Analysis:')
        print(f'  Target coverage: {self.confidence_level*100:.0f}%')
        print(f'  Actual coverage (quantile): {coverage_rate*100:.1f}%')
        print(f'  Actual coverage (residual): {coverage_rate_alt*100:.1f}%')
        print(f'  Average interval width: {avg_width:.3f}')
        print(f'  Median interval width: {median_width:.3f}')
        
        return results

def add_prediction_intervals(predictor, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Add prediction intervals to an existing predictor
    """
    print('\\n=== PREDICTION INTERVAL ESTIMATION ===')
    
    # Create interval estimator
    interval_estimator = PredictionIntervalEstimator(predictor.best_model, confidence_level=0.9)
    
    # Fit quantile models
    interval_estimator.fit_quantile_models(X_train, y_train, X_val, y_val)
    
    # Evaluate coverage
    coverage_results = interval_estimator.evaluate_interval_coverage(X_test, y_test)
    
    # Generate sample predictions with intervals
    sample_indices = np.random.choice(len(X_test), size=10, replace=False)
    sample_predictions = interval_estimator.predict_with_intervals(X_test[sample_indices])
    
    print(f'\\n[SAMPLE PREDICTIONS] With {interval_estimator.confidence_level*100}% Intervals:')
    for i, idx in enumerate(sample_indices[:5]):  # Show first 5
        actual = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        pred = sample_predictions['prediction'][i]
        lower = sample_predictions['lower_bound'][i]
        upper = sample_predictions['upper_bound'][i]
        
        # Convert to price scale for interpretation
        actual_price = np.exp(actual)
        pred_price = np.exp(pred)
        lower_price = np.exp(lower)
        upper_price = np.exp(upper)
        
        within_interval = lower <= actual <= upper
        status = "✓" if within_interval else "✗"
        
        print(f'  {i+1}. Actual: ${actual_price:,.0f} | Predicted: ${pred_price:,.0f}')
        print(f'     Interval: [${lower_price:,.0f}, ${upper_price:,.0f}] {status}')
    
    return interval_estimator, coverage_results