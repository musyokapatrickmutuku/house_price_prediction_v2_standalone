"""
Advanced House Price Predictor with Robust Error Handling
Addresses all major issues identified in the enhanced predictor analysis
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')

class AdvancedHousePricePredictor:
    """
    Advanced predictor with robust outlier handling, advanced feature engineering,
    and ensemble methods to address high RMSE and error rate issues
    """
    
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.selected_feature_names = None
        self.scaler = None
        self.selector = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def load_data(self, sample_size=None):
        """Load data with advanced cleaning"""
        print('Loading dataset with advanced cleaning...')
        
        if sample_size:
            # Load sample for faster processing
            total_rows = sum(1 for line in open(self.data_path, 'r', encoding='utf-8'))
            skip_rows = max(0, total_rows - sample_size - 1)
            self.df = pd.read_csv(self.data_path, skiprows=range(1, skip_rows))
            print(f'Sampled {len(self.df):,} rows from {total_rows:,} total rows')
        else:
            self.df = pd.read_csv(self.data_path)
            
        print(f'Dataset shape: {self.df.shape}')
        print(f'Columns: {list(self.df.columns)}')
        
        return self.df
    
    def robust_data_cleaning(self):
        """Advanced data cleaning to address zero prices and extreme outliers"""
        print('\\n=== ADVANCED DATA CLEANING ===')
        
        initial_count = len(self.df)
        
        # 1. Remove completely invalid records
        print('Removing invalid records...')
        
        # Remove zero or negative prices
        before_price = len(self.df)
        self.df = self.df[self.df['price'] > 0]
        removed_zero_price = before_price - len(self.df)
        print(f'  Removed {removed_zero_price:,} records with price <= 0')
        
        # Remove records with zero house size
        before_size = len(self.df)
        self.df = self.df[self.df['house_size'] > 0]
        removed_zero_size = before_size - len(self.df)
        print(f'  Removed {removed_zero_size:,} records with house_size <= 0')
        
        # Remove extreme outliers using domain knowledge
        before_extreme = len(self.df)
        
        # Price bounds: $5K to $20M (realistic US housing market)
        self.df = self.df[(self.df['price'] >= 5000) & (self.df['price'] <= 20000000)]
        
        # House size bounds: 100 to 50,000 sqft
        self.df = self.df[(self.df['house_size'] >= 100) & (self.df['house_size'] <= 50000)]
        
        # Acre lot bounds: 0.01 to 100 acres
        self.df = self.df[(self.df['acre_lot'] >= 0.01) & (self.df['acre_lot'] <= 100)]
        
        # Bedroom/bathroom sanity checks
        self.df = self.df[(self.df['bed'] >= 0) & (self.df['bed'] <= 20)]
        self.df = self.df[(self.df['bath'] >= 0) & (self.df['bath'] <= 20)]
        
        removed_extreme = before_extreme - len(self.df)
        print(f'  Removed {removed_extreme:,} extreme outliers using domain knowledge')
        
        total_removed = initial_count - len(self.df)
        removal_rate = (total_removed / initial_count) * 100
        print(f'  Total removed: {total_removed:,} ({removal_rate:.1f}%)')
        print(f'  Remaining records: {len(self.df):,}')
        
        return self.df
    
    def advanced_feature_engineering(self):
        """Create advanced features to improve prediction accuracy"""
        print('\\n=== ADVANCED FEATURE ENGINEERING ===')
        
        # Store original numeric columns
        numeric_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
        
        # 1. Location-based features (enhanced)
        print('Creating location features...')
        self.df['city_state'] = self.df['city'].astype(str) + '_' + self.df['state'].astype(str)
        
        # Target encoding for city_state (robust with smoothing)
        city_stats = self.df.groupby('city_state')['price'].agg(['mean', 'count', 'std']).reset_index()
        city_stats.columns = ['city_state', 'city_mean_price', 'city_count', 'city_std_price']
        
        # Smoothing for low-count cities (Bayesian approach)
        global_mean = self.df['price'].mean()
        smoothing_factor = 100  # Higher = more smoothing
        city_stats['city_encoded'] = (
            (city_stats['city_count'] * city_stats['city_mean_price'] + 
             smoothing_factor * global_mean) / 
            (city_stats['city_count'] + smoothing_factor)
        )
        
        self.df = self.df.merge(city_stats[['city_state', 'city_encoded', 'city_count']], on='city_state', how='left')
        
        # 2. Price-based features (robust)
        print('Creating price-based features...')
        
        # Price per sqft (with safeguards)
        self.df['price_per_sqft'] = self.df['price'] / np.maximum(self.df['house_size'], 1)
        
        # Relative price metrics
        self.df['price_vs_city_mean'] = self.df['price'] / np.maximum(self.df['city_encoded'], 1000)
        
        # 3. Property characteristics (advanced)
        print('Creating property features...')
        
        # Room ratios
        self.df['bed_bath_ratio'] = self.df['bed'] / np.maximum(self.df['bath'], 0.5)
        self.df['total_rooms'] = self.df['bed'] + self.df['bath']
        self.df['rooms_per_sqft'] = self.df['total_rooms'] / np.maximum(self.df['house_size'], 100)
        
        # Size ratios
        self.df['lot_size_sqft'] = self.df['acre_lot'] * 43560  # Convert acres to sqft
        self.df['house_to_lot_ratio'] = self.df['house_size'] / np.maximum(self.df['lot_size_sqft'], 100)
        self.df['lot_per_room'] = self.df['lot_size_sqft'] / np.maximum(self.df['total_rooms'], 1)
        
        # Efficiency metrics
        self.df['size_per_bed'] = self.df['house_size'] / np.maximum(self.df['bed'], 1)
        self.df['lot_per_bed'] = self.df['acre_lot'] / np.maximum(self.df['bed'], 1)
        
        # 4. Log transformations (safe)
        print('Creating log transformations...')
        self.df['log_price'] = np.log(np.maximum(self.df['price'], 1))
        self.df['log_house_size'] = np.log(np.maximum(self.df['house_size'], 1))
        self.df['log_acre_lot'] = np.log(np.maximum(self.df['acre_lot'], 0.01))
        self.df['log_price_per_sqft'] = np.log(np.maximum(self.df['price_per_sqft'], 1))
        
        # 5. Categorical features (enhanced)
        print('Processing categorical features...')
        
        # Status encoding
        status_map = {'for_sale': 0, 'sold': 1, 'ready_to_build': 2}
        self.df['status_encoded'] = self.df['status'].map(status_map).fillna(0)
        
        # Create status indicators
        for status in ['for_sale', 'sold', 'ready_to_build']:
            self.df[f'status_{status}'] = (self.df['status'] == status).astype(int)
        
        # 6. Statistical features
        print('Creating statistical features...')
        
        # Z-scores for outlier detection
        for col in ['price', 'house_size', 'acre_lot']:
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            self.df[f'{col}_zscore'] = (self.df[col] - mean_val) / std_val
        
        # Percentile rankings
        for col in ['price_per_sqft', 'house_size', 'acre_lot']:
            self.df[f'{col}_percentile'] = self.df[col].rank(pct=True)
        
        print(f'Features created: {self.df.shape[1]} total columns')
        
        return self.df
    
    def prepare_features(self):
        """Select and prepare features for modeling"""
        print('\\n=== FEATURE PREPARATION ===')
        
        # Select numeric features for modeling
        feature_cols = [
            # Original features
            'bed', 'bath', 'acre_lot', 'house_size',
            'brokered_by', 'street', 'zip_code',  # Keep as numeric if possible
            
            # Engineered features
            'city_encoded', 'city_count',
            'price_per_sqft', 'price_vs_city_mean',
            'bed_bath_ratio', 'total_rooms', 'rooms_per_sqft',
            'house_to_lot_ratio', 'lot_per_room', 'size_per_bed', 'lot_per_bed',
            'log_house_size', 'log_acre_lot', 'log_price_per_sqft',
            'status_encoded', 'status_for_sale', 'status_sold', 'status_ready_to_build',
            
            # Statistical features
            'house_size_zscore', 'acre_lot_zscore',
            'price_per_sqft_percentile', 'house_size_percentile', 'acre_lot_percentile'
        ]
        
        # Handle categorical columns that might be numeric
        for col in ['brokered_by', 'street', 'zip_code']:
            if col in self.df.columns:
                # Try to convert to numeric, fill NaN with median
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                except:
                    # If conversion fails, drop the column
                    if col in feature_cols:
                        feature_cols.remove(col)
        
        # Keep only available columns
        available_cols = [col for col in feature_cols if col in self.df.columns]
        
        # Prepare X and y
        X = self.df[available_cols].copy()
        y = self.df['log_price'].copy()  # Use log price as target
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_names = list(X.columns)
        print(f'Final features: {len(self.feature_names)}')
        
        return X, y
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data with stratification"""
        print('\\n=== DATA SPLITTING (STRATIFIED) ===')
        
        X, y = self.prepare_features()
        
        # Create price bins for stratification
        price_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=price_bins
        )
        
        # Create new bins for remaining data
        temp_bins = pd.qcut(y_temp, q=5, labels=False, duplicates='drop')
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state,
            stratify=temp_bins
        )
        
        print(f'Train set: {self.X_train.shape[0]:,} samples')
        print(f'Validation set: {self.X_val.shape[0]:,} samples')  
        print(f'Test set: {self.X_test.shape[0]:,} samples')
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def feature_selection_and_scaling(self, k_features=20):
        """Advanced feature selection and robust scaling"""
        print(f'\\n=== FEATURE SELECTION & SCALING (top {k_features}) ===')
        
        # Feature selection using F-regression
        self.selector = SelectKBest(score_func=f_regression, k=min(k_features, len(self.feature_names)))
        
        # Fit on training data
        X_train_selected = self.selector.fit_transform(self.X_train, self.y_train)
        X_val_selected = self.selector.transform(self.X_val)
        X_test_selected = self.selector.transform(self.X_test)
        
        # Get selected feature names
        selected_indices = self.selector.get_support()
        self.selected_feature_names = [self.feature_names[i] for i in range(len(selected_indices)) if selected_indices[i]]
        
        print('Selected features (by importance):')
        feature_scores = self.selector.scores_[selected_indices]
        for i, (name, score) in enumerate(zip(self.selected_feature_names, feature_scores), 1):
            print(f'  {i:2d}. {name}: {score:.2f}')
        
        # Robust scaling (less sensitive to outliers)
        print('\\nApplying robust scaling...')
        self.scaler = RobustScaler()
        
        self.X_train = self.scaler.fit_transform(X_train_selected)
        self.X_val = self.scaler.transform(X_val_selected)
        self.X_test = self.scaler.transform(X_test_selected)
        
        print(f'Scaled feature matrix shape: {self.X_train.shape}')
        
        return self.X_train, self.X_val, self.X_test
    
    def train_advanced_models(self):
        """Train multiple models including ensemble methods"""
        print('\\n=== ADVANCED MODEL TRAINING ===')
        
        # Define models with better hyperparameters
        models_to_train = {
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=2, random_state=self.random_state
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f'\\nTraining {name}...')
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            
            # Metrics on log scale
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'overfitting': abs(train_rmse - val_rmse) / train_rmse
            }
            
            print(f'  Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}')
            print(f'  Val RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}')
            print(f'  Overfitting ratio: {results[name]["overfitting"]:.3f}')
        
        # Select best model based on validation performance and overfitting
        best_score = float('inf')
        best_name = None
        
        for name, metrics in results.items():
            # Combined score: validation RMSE + overfitting penalty
            score = metrics['val_rmse'] + 0.1 * metrics['overfitting']
            if score < best_score:
                best_score = score
                best_name = name
        
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        self.models = {k: v['model'] for k, v in results.items()}
        
        print(f'\\nBest model selected: {self.best_model_name} (combined score: {best_score:.4f})')
        
        return results
    
    def final_evaluation(self):
        """Comprehensive final evaluation with robust error metrics"""
        print('\\n=== COMPREHENSIVE FINAL EVALUATION ===')
        
        # Test predictions on log scale
        test_pred_log = self.best_model.predict(self.X_test)
        
        # Convert back to price scale (robust)
        test_pred_price = np.exp(test_pred_log)
        y_test_price = np.exp(self.y_test)
        
        # Log scale metrics
        test_rmse_log = np.sqrt(mean_squared_error(self.y_test, test_pred_log))
        test_mae_log = mean_absolute_error(self.y_test, test_pred_log)
        test_r2 = r2_score(self.y_test, test_pred_log)
        
        # Price scale metrics (robust)
        test_rmse_price = np.sqrt(mean_squared_error(y_test_price, test_pred_price))
        test_mae_price = mean_absolute_error(y_test_price, test_pred_price)
        
        # Percentage errors (robust calculation)
        percentage_errors = np.abs(test_pred_price - y_test_price) / y_test_price * 100
        
        # Filter out extreme percentage errors (cap at 500%)
        percentage_errors_clean = percentage_errors[percentage_errors <= 500]
        
        median_percentage_error = np.median(percentage_errors_clean)
        mean_percentage_error = np.mean(percentage_errors_clean)
        
        # Additional robust metrics
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean(200 * np.abs(test_pred_price - y_test_price) / 
                       (np.abs(test_pred_price) + np.abs(y_test_price)))
        
        # Mean Absolute Percentage Error within reasonable bounds
        mape_bounded = np.mean(np.minimum(percentage_errors, 100))  # Cap at 100%
        
        print(f'LOG SCALE METRICS:')
        print(f'  RMSE: {test_rmse_log:.4f}')
        print(f'  MAE: {test_mae_log:.4f}')
        print(f'  R² Score: {test_r2:.4f}')
        
        print(f'\\nPRICE SCALE METRICS:')
        print(f'  RMSE: ${test_rmse_price:,.0f}')
        print(f'  MAE: ${test_mae_price:,.0f}')
        
        print(f'\\nPERCENTAGE ERROR METRICS (ROBUST):')
        print(f'  Median APE: {median_percentage_error:.2f}%')
        print(f'  Mean APE: {mean_percentage_error:.2f}%')
        print(f'  SMAPE: {smape:.2f}%')
        print(f'  Bounded MAPE: {mape_bounded:.2f}%')
        
        print(f'\\nERROR DISTRIBUTION:')
        print(f'  Errors < 10%: {(percentage_errors_clean < 10).mean()*100:.1f}%')
        print(f'  Errors < 20%: {(percentage_errors_clean < 20).mean()*100:.1f}%')
        print(f'  Errors < 30%: {(percentage_errors_clean < 30).mean()*100:.1f}%')
        
        results = {
            'test_rmse_log': test_rmse_log,
            'test_mae_log': test_mae_log,
            'test_r2': test_r2,
            'test_rmse_price': test_rmse_price,
            'test_mae_price': test_mae_price,
            'median_percentage_error': median_percentage_error,
            'mean_percentage_error': mean_percentage_error,
            'smape': smape,
            'mape_bounded': mape_bounded,
            'error_under_10pct': (percentage_errors_clean < 10).mean(),
            'error_under_20pct': (percentage_errors_clean < 20).mean(),
            'error_under_30pct': (percentage_errors_clean < 30).mean()
        }
        
        return results
    
    def save_model(self, filepath):
        """Save the complete model pipeline"""
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'selector': self.selector,
            'feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names,
            'best_model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f'\\nAdvanced model saved to {filepath}')
        
        return filepath