#!/usr/bin/env python3
"""
Final improved house price prediction model with enhanced consistency
Addresses remaining challenges and applies minor improvements
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
sys.path.insert(0, 'src')

class FinalImprovedHousePricePredictor:
    def __init__(self, data_path='data/raw/df_imputed.csv', random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.selected_feature_names = None
        # Enhanced mandatory features for better consistency
        self.mandatory_features = ['house_size', 'bed', 'bath', 'acre_lot', 'log_house_size']
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, sample_size=50000):
        """Load data with robust cleaning"""
        print('=' * 60)
        print('LOADING DATA FOR FINAL IMPROVED MODEL')
        print('=' * 60)
        
        data_paths = [
            self.data_path,
            r'C:\Users\HP\Desktop\DataSets\df_imputed.csv',
            'data/raw/df_imputed.csv'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                print(f'Loading data from: {path}')
                if sample_size:
                    total_rows = sum(1 for line in open(path, 'r', encoding='utf-8'))
                    if total_rows > sample_size:
                        skip_rows = max(0, total_rows - sample_size - 1)
                        self.df = pd.read_csv(path, skiprows=range(1, skip_rows))
                        print(f'Sampled {len(self.df):,} rows from {total_rows:,} total rows')
                    else:
                        self.df = pd.read_csv(path)
                        print(f'Loaded full dataset: {self.df.shape}')
                else:
                    self.df = pd.read_csv(path)
                    print(f'Loaded full dataset: {self.df.shape}')
                break
        else:
            raise FileNotFoundError("Dataset not found in any expected location")
            
        print(f'Dataset columns: {list(self.df.columns)}')
        return self.df
    
    def enhanced_data_cleaning(self):
        """Enhanced data cleaning for better consistency"""
        print('\\n=== ENHANCED DATA CLEANING ===')
        
        print(f'Original dataset: {self.df.shape}')
        
        # Remove zero or negative prices
        original_count = len(self.df)
        self.df = self.df[self.df['price'] > 0]
        print(f'Removed {original_count - len(self.df):,} zero/negative price records')
        
        # Tighter domain bounds for more consistent predictions
        self.df = self.df[(self.df['price'] >= 10000) & (self.df['price'] <= 15000000)]
        print(f'Applied tighter domain bounds: {len(self.df):,} records remain')
        
        # Enhanced property feature cleaning
        self.df = self.df[self.df['bed'] >= 1]  # At least 1 bedroom
        self.df = self.df[self.df['bath'] >= 0.5]  # At least half bath
        self.df = self.df[self.df['house_size'] >= 200]  # Minimum 200 sqft
        self.df = self.df[self.df['house_size'] <= 15000]  # Maximum 15000 sqft
        self.df = self.df[self.df['acre_lot'] > 0]
        self.df = self.df[self.df['acre_lot'] <= 10]  # Maximum 10 acres
        
        # Remove extreme outliers more aggressively for consistency
        for feature in ['house_size', 'bed', 'bath', 'acre_lot']:
            q1 = self.df[feature].quantile(0.05)
            q3 = self.df[feature].quantile(0.95)
            iqr = q3 - q1
            lower = q1 - 1.0 * iqr  # Tighter bounds
            upper = q3 + 1.0 * iqr
            
            before = len(self.df)
            self.df = self.df[(self.df[feature] >= lower) & (self.df[feature] <= upper)]
            print(f'Removed {before - len(self.df):,} outliers for {feature}')
        
        print(f'Final cleaned dataset: {self.df.shape}')
        
        # Create log target for training stability
        self.df['log_price'] = np.log(self.df['price'])
        
        return self.df
    
    def enhanced_feature_engineering(self):
        """Enhanced feature engineering for better price consistency"""
        print('\\n=== ENHANCED FEATURE ENGINEERING ===')
        
        # 1. Enhanced location features
        print('Creating enhanced location features...')
        self.df['city_state'] = self.df['city'].astype(str) + '_' + self.df['state'].astype(str)
        
        # More comprehensive city frequency mapping
        city_counts = self.df['city_state'].value_counts()
        self.df['city_frequency'] = self.df['city_state'].map(city_counts)
        self.df['city_encoded'] = pd.Categorical(self.df['city_state']).codes
        self.df['log_city_frequency'] = np.log1p(self.df['city_frequency'])
        
        # Enhanced city tier system for better location pricing
        high_tier_cities = ['Los Angeles_CA', 'New York_NY', 'San Francisco_CA', 'Seattle_WA', 'Boston_MA']
        mid_tier_cities = ['Chicago_IL', 'Houston_TX', 'Phoenix_AZ', 'Philadelphia_PA', 'San Diego_CA']
        
        self.df['city_tier'] = 0  # Low tier default
        self.df.loc[self.df['city_state'].isin(mid_tier_cities), 'city_tier'] = 1
        self.df.loc[self.df['city_state'].isin(high_tier_cities), 'city_tier'] = 2
        
        # 2. Enhanced core property features
        print('Creating enhanced property features...')
        
        # Primary size features (mandatory)
        self.df['log_house_size'] = np.log1p(self.df['house_size'])
        self.df['sqrt_house_size'] = np.sqrt(self.df['house_size'])
        
        # Enhanced room features
        self.df['bed_bath_ratio'] = self.df['bed'] / np.maximum(self.df['bath'], 0.5)
        self.df['total_rooms'] = self.df['bed'] + self.df['bath']
        self.df['rooms_per_sqft'] = self.df['total_rooms'] / np.maximum(self.df['house_size'], 100)
        
        # More consistent size efficiency metrics
        self.df['sqft_per_room'] = self.df['house_size'] / np.maximum(self.df['total_rooms'], 1)
        self.df['sqft_per_bed'] = self.df['house_size'] / np.maximum(self.df['bed'], 1)
        self.df['bath_per_bed'] = self.df['bath'] / np.maximum(self.df['bed'], 1)
        
        # 3. Enhanced lot features
        print('Creating enhanced lot features...')
        self.df['lot_sqft'] = self.df['acre_lot'] * 43560
        self.df['log_lot_sqft'] = np.log1p(self.df['lot_sqft'])
        
        # Better land utilization ratios
        self.df['house_to_lot_ratio'] = self.df['house_size'] / np.maximum(self.df['lot_sqft'], 100)
        self.df['lot_per_room'] = self.df['lot_sqft'] / np.maximum(self.df['total_rooms'], 1)
        self.df['land_efficiency'] = np.log1p(self.df['house_size']) / np.log1p(self.df['lot_sqft'])
        
        # 4. Enhanced property scale features
        print('Creating enhanced scale features...')
        
        # More granular size categories
        self.df['house_size_category'] = pd.qcut(self.df['house_size'], q=20, labels=False, duplicates='drop')
        self.df['lot_size_category'] = pd.qcut(self.df['acre_lot'], q=15, labels=False, duplicates='drop')
        
        # Enhanced luxury indicators
        self.df['is_large_house'] = (self.df['house_size'] > self.df['house_size'].quantile(0.75)).astype(int)
        self.df['is_large_lot'] = (self.df['acre_lot'] > self.df['acre_lot'].quantile(0.75)).astype(int)
        self.df['many_bedrooms'] = (self.df['bed'] >= 4).astype(int)
        self.df['many_bathrooms'] = (self.df['bath'] >= 3).astype(int)
        
        # Size consistency features
        self.df['size_consistency'] = np.abs(self.df['sqft_per_room'] - self.df['sqft_per_room'].median())
        self.df['size_efficiency_score'] = self.df['house_size'] * np.log1p(self.df['total_rooms'])
        
        # 5. Enhanced status features
        print('Creating enhanced status features...')
        status_map = {'for_sale': 0, 'sold': 1, 'ready_to_build': 2}
        self.df['status_encoded'] = self.df['status'].map(status_map).fillna(0)
        self.df['status_for_sale'] = (self.df['status'] == 'for_sale').astype(int)
        self.df['status_sold'] = (self.df['status'] == 'sold').astype(int)
        self.df['status_ready_to_build'] = (self.df['status'] == 'ready_to_build').astype(int)
        
        # 6. Enhanced statistical features with better normalization
        print('Creating enhanced statistical features...')
        
        # More stable z-scores using robust statistics
        for feature in ['house_size', 'bed', 'bath', 'acre_lot']:
            feature_median = self.df[feature].median()
            # Calculate median absolute deviation manually
            feature_mad = (self.df[feature] - feature_median).abs().median()
            self.df[f'{feature}_zscore'] = (self.df[feature] - feature_median) / np.maximum(feature_mad, 1)
        
        # Price consistency features
        self.df['rooms_size_balance'] = self.df['total_rooms'] * np.log1p(self.df['house_size'])
        self.df['property_desirability'] = (
            self.df['house_size'] * 0.4 + 
            self.df['total_rooms'] * 500 * 0.3 + 
            self.df['lot_sqft'] * 0.0001 * 0.2 +
            self.df['city_tier'] * 10000 * 0.1
        )
        
        # Handle categorical columns
        for col in ['brokered_by', 'street', 'zip_code']:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                except:
                    pass
        
        print(f'Enhanced feature engineering completed. Dataset shape: {self.df.shape}')
        return self.df
    
    def prepare_enhanced_features(self):
        """Prepare enhanced feature matrix"""
        print('\\n=== PREPARING ENHANCED FEATURES ===')
        
        # Comprehensive feature list with enhancements
        feature_cols = [
            # Enhanced mandatory features
            'house_size', 'bed', 'bath', 'acre_lot', 'log_house_size',
            
            # Enhanced size features
            'sqrt_house_size', 'size_efficiency_score', 'property_desirability',
            
            # Enhanced location features
            'city_encoded', 'city_frequency', 'log_city_frequency', 'city_tier',
            
            # Enhanced property ratios
            'bed_bath_ratio', 'total_rooms', 'rooms_per_sqft',
            'sqft_per_room', 'sqft_per_bed', 'bath_per_bed', 'rooms_size_balance',
            
            # Enhanced lot features
            'lot_sqft', 'log_lot_sqft', 'house_to_lot_ratio', 'lot_per_room', 'land_efficiency',
            
            # Enhanced scale features
            'house_size_category', 'lot_size_category', 'size_consistency',
            'is_large_house', 'is_large_lot', 'many_bedrooms', 'many_bathrooms',
            
            # Enhanced status features
            'status_encoded', 'status_for_sale', 'status_sold', 'status_ready_to_build',
            
            # Enhanced statistical features
            'house_size_zscore', 'lot_zscore', 'bed_zscore', 'bath_zscore',
            
            # Other features
            'brokered_by', 'street', 'zip_code'
        ]
        
        # Keep only available columns
        available_cols = [col for col in feature_cols if col in self.df.columns]
        
        print(f'Available enhanced features: {len(available_cols)}')
        print('Enhanced mandatory features:', [f for f in self.mandatory_features if f in available_cols])
        
        # Prepare X and y
        X = self.df[available_cols].copy()
        y = self.df['log_price'].copy()
        
        # Enhanced NaN handling
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_names = list(X.columns)
        print(f'Final enhanced feature matrix: {X.shape}')
        
        return X, y
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data with stratification"""
        print('\\n=== DATA SPLITTING ===')
        
        X, y = self.prepare_enhanced_features()
        
        # Create price bins for stratification
        price_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')  # More bins for better stratification
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=price_bins
        )
        
        # Create new bins for remaining data
        temp_bins = pd.qcut(y_temp, q=8, labels=False, duplicates='drop')
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state,
            stratify=temp_bins
        )
        
        # Convert to numpy arrays for easier indexing
        self.X_train = self.X_train.values
        self.X_val = self.X_val.values
        self.X_test = self.X_test.values
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values
        self.y_test = self.y_test.values
        
        print(f'Train set: {self.X_train.shape[0]:,} samples')
        print(f'Validation set: {self.X_val.shape[0]:,} samples')  
        print(f'Test set: {self.X_test.shape[0]:,} samples')
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def enhanced_feature_selection(self, k_features=25):
        """Enhanced feature selection with mandatory features"""
        print(f'\\n=== ENHANCED FEATURE SELECTION ===')
        
        # Get indices of mandatory features
        mandatory_indices = []
        for feature in self.mandatory_features:
            if feature in self.feature_names:
                mandatory_indices.append(self.feature_names.index(feature))
        
        print(f'Enhanced mandatory features: {[self.feature_names[i] for i in mandatory_indices]}')
        
        # Select additional features
        k_additional = max(0, k_features - len(mandatory_indices))
        
        if k_additional > 0:
            # Get non-mandatory features
            non_mandatory_indices = [i for i in range(len(self.feature_names)) if i not in mandatory_indices]
            non_mandatory_feature_names = [self.feature_names[i] for i in non_mandatory_indices]
            
            # Enhanced feature selection with better scoring
            selector_temp = SelectKBest(score_func=f_regression, k=min(k_additional, len(non_mandatory_indices)))
            
            # Fit on non-mandatory features
            X_train_non_mandatory = self.X_train[:, non_mandatory_indices]
            selector_temp.fit(X_train_non_mandatory, self.y_train)
            
            # Get selected features
            selected_non_mandatory = selector_temp.get_support()
            additional_features = [non_mandatory_feature_names[i] for i in range(len(selected_non_mandatory)) if selected_non_mandatory[i]]
            
            # Combine mandatory and selected additional features
            self.selected_feature_names = [self.feature_names[i] for i in mandatory_indices] + additional_features
        else:
            self.selected_feature_names = [self.feature_names[i] for i in mandatory_indices]
        
        print(f'Enhanced selected features ({len(self.selected_feature_names)}):')
        for i, name in enumerate(self.selected_feature_names, 1):
            mandatory_mark = ' (MANDATORY)' if name in self.mandatory_features else ''
            print(f'  {i:2d}. {name}{mandatory_mark}')
        
        # Create final feature matrices
        selected_indices = [self.feature_names.index(name) for name in self.selected_feature_names]
        
        self.X_train = self.X_train[:, selected_indices]
        self.X_val = self.X_val[:, selected_indices]
        self.X_test = self.X_test[:, selected_indices]
        
        # Enhanced scaling
        print('\\nApplying enhanced RobustScaler...')
        self.scaler = RobustScaler(quantile_range=(10, 90))  # More robust quantile range
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f'Final enhanced feature matrix shape: {self.X_train.shape}')
        
        return self.X_train, self.X_val, self.X_test
    
    def train_enhanced_models(self):
        """Train enhanced models with better hyperparameters"""
        print('\\n=== TRAINING ENHANCED MODELS ===')
        
        models = {
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,  # More trees
                max_depth=7,  # Slightly deeper
                learning_rate=0.08,  # Slower learning
                subsample=0.8,  # Subsampling for regularization
                random_state=self.random_state
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=120,
                max_depth=12,
                min_samples_split=5,  # Better regularization
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Ridge': Ridge(alpha=5.0),  # Stronger regularization
            'Lasso': Lasso(alpha=0.05, max_iter=3000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=3000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f'\\nTraining enhanced {name}...')
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            val_r2 = r2_score(self.y_val, val_pred)
            
            # Overfitting check
            overfitting_ratio = val_rmse / train_rmse
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'overfitting_ratio': overfitting_ratio
            }
            
            print(f'  Train RMSE: {train_rmse:.4f}')
            print(f'  Val RMSE:   {val_rmse:.4f}')
            print(f'  Val R²:     {val_r2:.4f}')
            print(f'  Overfitting: {overfitting_ratio:.2f}')
        
        # Select best model with overfitting consideration
        best_name = min(results.keys(), key=lambda x: results[x]['val_rmse'] * (1 + max(0, results[x]['overfitting_ratio'] - 1.1)))
        
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        self.models = results
        
        print(f'\\nBest Enhanced Model: {best_name}')
        print(f'   Validation RMSE: {results[best_name]["val_rmse"]:.4f}')
        print(f'   Validation R²:   {results[best_name]["val_r2"]:.4f}')
        print(f'   Overfitting:     {results[best_name]["overfitting_ratio"]:.2f}')
        
        return self.best_model
    
    def evaluate_enhanced_model(self):
        """Enhanced model evaluation"""
        print('\\n=== ENHANCED MODEL EVALUATION ===')
        
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        # Test predictions
        test_pred = self.best_model.predict(self.X_test)
        
        # Metrics in log space
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_r2 = r2_score(self.y_test, test_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        
        # Metrics in original price space
        y_test_orig = np.exp(self.y_test)
        test_pred_orig = np.exp(test_pred)
        
        orig_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
        orig_mae = mean_absolute_error(y_test_orig, test_pred_orig)
        orig_mape = np.mean(np.abs((y_test_orig - test_pred_orig) / y_test_orig)) * 100
        
        print(f'Enhanced Test Set Performance ({self.best_model_name}):')
        print(f'  Log Space - RMSE: {test_rmse:.4f}')
        print(f'  Log Space - R²:   {test_r2:.4f}')
        print(f'  Log Space - MAE:  {test_mae:.4f}')
        print(f'  Price Space - RMSE: ${orig_rmse:,.0f}')
        print(f'  Price Space - MAE:  ${orig_mae:,.0f}')
        print(f'  Price Space - MAPE: {orig_mape:.2f}%')
        
        # Enhanced feature importance analysis
        if hasattr(self.best_model, 'feature_importances_'):
            print(f'\\nTop 15 Enhanced Feature Importances:')
            importances = self.best_model.feature_importances_
            feature_importance_pairs = list(zip(self.selected_feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance_pairs[:15], 1):
                mandatory_mark = ' (MANDATORY)' if feature in self.mandatory_features else ''
                print(f'  {i:2d}. {feature:25}: {importance:.4f}{mandatory_mark}')
        
        return {
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'orig_rmse': orig_rmse,
            'orig_mae': orig_mae,
            'orig_mape': orig_mape
        }
    
    def save_enhanced_model(self, filepath='models/trained/final_improved_model.pkl'):
        """Save the enhanced model"""
        print(f'\\n=== SAVING ENHANCED MODEL TO {filepath} ===')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare enhanced model data
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'selector': None,
            'feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names,
            'mandatory_features': self.mandatory_features,
            'models': {name: results['model'] for name, results in self.models.items()},
            'version': 'final_enhanced_v1.0'
        }
        
        # Save
        joblib.dump(model_data, filepath)
        print(f'Enhanced model saved successfully!')
        print(f'   Features: {len(self.selected_feature_names)}')
        print(f'   Mandatory features: {self.mandatory_features}')
        print(f'   Version: {model_data["version"]}')
        
        return filepath

def main():
    """Main enhanced training pipeline"""
    print("FINAL ENHANCED HOUSE PRICE PREDICTION MODEL")
    print("Resolving remaining challenges and applying improvements")
    print("=" * 70)
    
    # Initialize enhanced predictor
    predictor = FinalImprovedHousePricePredictor()
    
    # Enhanced training pipeline
    predictor.load_data(sample_size=60000)  # Slightly more data
    predictor.enhanced_data_cleaning()
    predictor.enhanced_feature_engineering()
    predictor.split_data()
    predictor.enhanced_feature_selection(k_features=25)  # More features
    predictor.train_enhanced_models()
    predictor.evaluate_enhanced_model()
    
    # Save the final enhanced model
    model_path = predictor.save_enhanced_model('models/trained/final_improved_model.pkl')
    
    print("\\n" + "=" * 70)
    print("FINAL ENHANCED MODEL TRAINING COMPLETED!")
    print(f"Model saved to: {model_path}")
    print("All challenges resolved and improvements applied!")
    print("=" * 70)
    
    return model_path

if __name__ == "__main__":
    main()