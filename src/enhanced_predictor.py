"""
Enhanced House Price Prediction Model
With improved feature engineering and outlier handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedHousePricePredictor:
    def __init__(self, data_path=None):
        """Initialize the enhanced house price predictor"""
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.city_state_encoder = None
        
    def load_data(self, sample_size=None):
        """Load and optionally sample the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Sample data if specified for faster processing
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows for faster processing")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def basic_eda(self):
        """Perform basic exploratory data analysis"""
        print("\n=== ENHANCED DATA OVERVIEW ===")
        print(self.df.info())
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        # Check cardinality of categorical features
        print(f"\nCategorical feature cardinality:")
        print(f"Cities: {self.df['city'].nunique()}")
        print(f"States: {self.df['state'].nunique()}")
        print(f"Status: {self.df['status'].nunique()}")
        
        # Plot price distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.df['price'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(np.log1p(self.df['price']), bins=50, alpha=0.7, edgecolor='black')
        plt.title('Log-Transformed Price Distribution')
        plt.xlabel('Log(Price)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        price_by_state = self.df.groupby('state')['price'].median().sort_values(ascending=False).head(10)
        plt.barh(range(len(price_by_state)), price_by_state.values)
        plt.yticks(range(len(price_by_state)), price_by_state.index)
        plt.title('Median Price by State (Top 10)')
        plt.xlabel('Median Price')
        
        plt.tight_layout()
        # plt.show()  # Skip plotting for faster execution
        print("Plots generated but not displayed for faster execution")
        
    def advanced_outlier_handling(self, df, columns, method='iqr', iqr_multiplier=1.5):
        """
        Enhanced outlier handling with IQR capping
        """
        print(f"Handling outliers using {method} method...")
        df_clean = df.copy()
        outlier_summary = {}
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            original_values = df_clean[col].copy()
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
            else:  # quantile method
                lower_bound = df_clean[col].quantile(0.05)
                upper_bound = df_clean[col].quantile(0.95)
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Track changes
            lower_outliers = (original_values < lower_bound).sum()
            upper_outliers = (original_values > upper_bound).sum()
            total_outliers = lower_outliers + upper_outliers
            
            outlier_summary[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_capped': total_outliers,
                'outlier_percentage': (total_outliers / len(df)) * 100
            }
            
            print(f"  {col}: Capped {total_outliers} outliers ({(total_outliers/len(df)*100):.1f}%)")
            print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_clean, outlier_summary
    
    def target_encoding_with_cv(self, df, categorical_col, target_col, n_folds=5, smoothing=10):
        """
        Target encoding with cross-validation to prevent data leakage
        """
        print(f"Performing target encoding for {categorical_col}...")
        
        # Create k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        encoded_values = np.zeros(len(df))
        
        # Global mean for smoothing
        global_mean = df[target_col].mean()
        
        # Cross-validation encoding
        for train_idx, val_idx in kf.split(df):
            # Calculate encoding on training fold
            train_data = df.iloc[train_idx]
            encoding_map = train_data.groupby(categorical_col)[target_col].agg(['mean', 'count'])
            
            # Apply smoothing: (mean * count + global_mean * smoothing) / (count + smoothing)
            encoding_map['smoothed'] = (
                (encoding_map['mean'] * encoding_map['count'] + global_mean * smoothing) /
                (encoding_map['count'] + smoothing)
            )
            
            # Apply to validation fold
            val_data = df.iloc[val_idx]
            encoded_values[val_idx] = val_data[categorical_col].map(
                encoding_map['smoothed']
            ).fillna(global_mean)
        
        # Store encoder for future use
        final_encoding_map = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        final_encoding_map['smoothed'] = (
            (final_encoding_map['mean'] * final_encoding_map['count'] + global_mean * smoothing) /
            (final_encoding_map['count'] + smoothing)
        )
        
        self.city_state_encoder = {
            'encoding_map': final_encoding_map['smoothed'].to_dict(),
            'global_mean': global_mean
        }
        
        return encoded_values
    
    def enhanced_preprocessing(self):
        """Enhanced preprocessing with better feature engineering"""
        print("\n=== ENHANCED PREPROCESSING ===")
        df = self.df.copy()
        
        # Create log-transformed target
        df['price_log'] = np.log1p(df['price'])
        
        # Handle outliers using IQR method with capping
        numerical_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
        df, outlier_summary = self.advanced_outlier_handling(
            df, numerical_cols, method='iqr', iqr_multiplier=1.5
        )
        
        # Create combined city-state feature
        print("Creating city-state combination...")
        df['city_state'] = df['state'].astype(str) + "_" + df['city'].astype(str)
        print(f"Created {df['city_state'].nunique()} unique city-state combinations")
        
        # Apply target encoding with cross-validation
        df['city_state_encoded'] = self.target_encoding_with_cv(
            df, 'city_state', 'price_log', n_folds=5, smoothing=20
        )
        
        # One-hot encode status (low cardinality)
        df = pd.get_dummies(df, columns=['status'], prefix='status', drop_first=True)
        
        # Enhanced feature engineering
        print("Creating enhanced engineered features...")
        
        # Price per square foot
        df['price_per_sqft'] = df['price'] / (df['house_size'] + 1e-5)
        
        # Bed to bath ratio (bathrooms per bedroom)
        df['bed_bath_ratio'] = df['bath'] / (df['bed'] + 1e-5)
        
        # Additional useful features
        df['lot_size_per_bed'] = df['acre_lot'] / (df['bed'] + 1e-5)
        df['size_per_bed'] = df['house_size'] / (df['bed'] + 1e-5)
        df['total_rooms'] = df['bed'] + df['bath']
        df['house_to_lot_ratio'] = df['house_size'] / (df['acre_lot'] * 43560 + 1e-5)  # Convert acres to sqft
        
        # Log transforms for skewed features
        df['log_house_size'] = np.log1p(df['house_size'])
        df['log_acre_lot'] = np.log1p(df['acre_lot'])
        
        # Select features for modeling
        feature_cols = [
            'brokered_by', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 
            'house_size', 'city_state_encoded', 'price_per_sqft', 
            'bed_bath_ratio', 'lot_size_per_bed', 'size_per_bed', 'total_rooms',
            'house_to_lot_ratio', 'log_house_size', 'log_acre_lot'
        ]
        
        # Add status columns if they exist
        status_cols = [col for col in df.columns if col.startswith('status_')]
        feature_cols.extend(status_cols)
        
        # Keep only available features and handle missing values
        available_features = [col for col in feature_cols if col in df.columns]
        
        self.X = df[available_features].fillna(0)  # Fill any remaining NAs
        self.y = df['price_log']
        self.feature_names = available_features
        
        print(f"Features selected: {len(available_features)}")
        print("Feature list:")
        for i, feat in enumerate(available_features):
            print(f"  {i+1}. {feat}")
        
        return self.X, self.y, outlier_summary
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data into train/validation/test sets"""
        print("\n=== DATA SPLITTING ===")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples") 
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def feature_selection(self, k_features=12):
        """Select top k features using univariate selection"""
        print(f"\n=== FEATURE SELECTION (top {k_features}) ===")
        
        selector = SelectKBest(score_func=f_regression, k=k_features)
        
        # Fit on training data only
        self.X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        self.X_val_selected = selector.transform(self.X_val)
        self.X_test_selected = selector.transform(self.X_test)
        
        # Get selected feature names and scores
        selected_features = selector.get_support(indices=True)
        self.selected_feature_names = [self.feature_names[i] for i in selected_features]
        feature_scores = [(self.feature_names[i], selector.scores_[i]) for i in selected_features]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Selected features (by importance):")
        for i, (feature, score) in enumerate(feature_scores):
            print(f"  {i+1}. {feature}: {score:.2f}")
        
        return self.X_train_selected, self.X_val_selected, self.X_test_selected
    
    def train_models(self):
        """Train multiple lightweight models and select the best"""
        print("\n=== MODEL TRAINING ===")
        
        # Define lightweight models with better parameters
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=3000),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=3000)
        }
        
        self.model_results = {}
        
        # Create scaler
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train_selected)
        X_val_scaled = self.scaler.transform(self.X_val_selected)
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, self.y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)
            
            self.model_results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2
            }
            
            print(f"  Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
            print(f"  Val RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
        
        # Select best model based on validation RMSE
        best_model_name = min(self.model_results.keys(), 
                             key=lambda x: self.model_results[x]['val_rmse'])
        self.best_model = self.model_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        
        return self.model_results
    
    def hyperparameter_tuning(self):
        """Enhanced hyperparameter tuning"""
        print(f"\n=== HYPERPARAMETER TUNING FOR {self.best_model_name} ===")
        
        X_train_scaled = self.scaler.transform(self.X_train_selected)
        
        if self.best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            # Tune regularization parameter with wider range
            if self.best_model_name == 'Ridge':
                alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            else:  # Lasso and ElasticNet
                alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            
            if self.best_model_name == 'Ridge':
                model_class = Ridge
                param_name = 'alpha'
            elif self.best_model_name == 'Lasso':
                model_class = Lasso
                param_name = 'alpha'
            else:  # ElasticNet
                model_class = ElasticNet
                param_name = 'alpha'
            
            train_scores, val_scores = validation_curve(
                model_class(random_state=42, max_iter=3000), 
                X_train_scaled, self.y_train,
                param_name=param_name, param_range=alphas,
                cv=3, scoring='neg_root_mean_squared_error'
            )
            
            # Convert to positive RMSE
            train_rmse_mean = -train_scores.mean(axis=1)
            val_rmse_mean = -val_scores.mean(axis=1)
            
            # Find best alpha
            best_alpha_idx = np.argmin(val_rmse_mean)
            best_alpha = alphas[best_alpha_idx]
            
            print(f"Alpha candidates: {alphas}")
            print(f"Validation RMSEs: {val_rmse_mean}")
            print(f"Best alpha: {best_alpha}")
            print(f"Best validation RMSE: {val_rmse_mean[best_alpha_idx]:.4f}")
            
            # Retrain with best alpha
            if self.best_model_name == 'Ridge':
                self.best_model = Ridge(alpha=best_alpha, random_state=42)
            elif self.best_model_name == 'Lasso':
                self.best_model = Lasso(alpha=best_alpha, random_state=42, max_iter=3000)
            else:
                self.best_model = ElasticNet(alpha=best_alpha, l1_ratio=0.5, random_state=42, max_iter=3000)
            
            self.best_model.fit(X_train_scaled, self.y_train)
    
    def final_evaluation(self):
        """Comprehensive evaluation on test set"""
        print("\n=== FINAL EVALUATION ===")
        
        X_test_scaled = self.scaler.transform(self.X_test_selected)
        test_pred = self.best_model.predict(X_test_scaled)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_mae = mean_absolute_error(self.y_test, test_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        # Convert back to original price scale for interpretation
        test_pred_price = np.expm1(test_pred)
        y_test_price = np.expm1(self.y_test)
        
        price_rmse = np.sqrt(mean_squared_error(y_test_price, test_pred_price))
        price_mae = mean_absolute_error(y_test_price, test_pred_price)
        
        # Calculate percentage errors
        percentage_errors = np.abs((test_pred_price - y_test_price) / y_test_price) * 100
        median_pe = np.median(percentage_errors)
        mean_pe = np.mean(percentage_errors)
        
        print(f"Test RMSE (log scale): {test_rmse:.4f}")
        print(f"Test MAE (log scale): {test_mae:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE (price scale): ${price_rmse:,.0f}")
        print(f"Test MAE (price scale): ${price_mae:,.0f}")
        print(f"Median Percentage Error: {median_pe:.1f}%")
        print(f"Mean Percentage Error: {mean_pe:.1f}%")
        
        # Feature importance (for linear models)
        if hasattr(self.best_model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_feature_names,
                'coefficient': self.best_model.coef_,
                'abs_coefficient': np.abs(self.best_model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Predictions vs Actual
        axes[0,0].scatter(self.y_test, test_pred, alpha=0.5)
        axes[0,0].plot([self.y_test.min(), self.y_test.max()], 
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Log(Price)')
        axes[0,0].set_ylabel('Predicted Log(Price)')
        axes[0,0].set_title('Predictions vs Actual (Log Scale)')
        
        # Residuals
        residuals = test_pred - self.y_test
        axes[0,1].scatter(test_pred, residuals, alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Log(Price)')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot')
        
        # Price scale comparison
        axes[1,0].scatter(y_test_price, test_pred_price, alpha=0.5)
        axes[1,0].plot([y_test_price.min(), y_test_price.max()], 
                      [y_test_price.min(), y_test_price.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Price ($)')
        axes[1,0].set_ylabel('Predicted Price ($)')
        axes[1,0].set_title('Predictions vs Actual (Price Scale)')
        
        # Percentage error distribution
        axes[1,1].hist(percentage_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(median_pe, color='r', linestyle='--', label=f'Median: {median_pe:.1f}%')
        axes[1,1].set_xlabel('Absolute Percentage Error (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Percentage Errors')
        axes[1,1].legend()
        
        plt.tight_layout()
        # plt.show()  # Skip plotting for faster execution
        print("Plots generated but not displayed for faster execution")
        
        return {
            'test_rmse_log': test_rmse,
            'test_mae_log': test_mae,
            'test_r2': test_r2,
            'test_rmse_price': price_rmse,
            'test_mae_price': price_mae,
            'median_percentage_error': median_pe,
            'mean_percentage_error': mean_pe
        }
    
    def save_model(self, model_path='enhanced_house_price_model.pkl'):
        """Save the trained model and preprocessing components"""
        print(f"\nSaving enhanced model to {model_path}")
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.selected_feature_names,
            'model_name': self.best_model_name,
            'city_state_encoder': self.city_state_encoder
        }
        
        joblib.dump(model_package, model_path)
        print("Enhanced model saved successfully!")
        
        return model_path
    
    def load_model(self, model_path='enhanced_house_price_model.pkl'):
        """Load a saved model"""
        print(f"Loading enhanced model from {model_path}")
        
        model_package = joblib.load(model_path)
        self.best_model = model_package['model']
        self.scaler = model_package['scaler']
        self.selected_feature_names = model_package['feature_names']
        self.best_model_name = model_package['model_name']
        
        if 'city_state_encoder' in model_package:
            self.city_state_encoder = model_package['city_state_encoder']
        
        print(f"Loaded enhanced {self.best_model_name} model successfully!")
    
    def predict_price(self, house_features):
        """Predict price for new house features"""
        if self.best_model is None:
            raise ValueError("No model trained. Please train a model first or load a saved model.")
        
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(house_features, dict):
                house_features = pd.DataFrame([house_features])
            
            # Check if we have required features
            missing_features = set(self.selected_feature_names) - set(house_features.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Ensure we have the right features
            house_features_selected = house_features[self.selected_feature_names]
            
            # Handle missing values
            house_features_selected = house_features_selected.fillna(0)
            
            # Scale features
            house_features_scaled = self.scaler.transform(house_features_selected)
            
            # Predict (log scale)
            log_price_pred = self.best_model.predict(house_features_scaled)
            
            # Convert back to price scale
            price_pred = np.expm1(log_price_pred)
            
            return price_pred[0] if len(price_pred) == 1 else price_pred
            
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")

def main():
    """Main execution function with enhanced pipeline"""
    print("=== ENHANCED HOUSE PRICE PREDICTION PIPELINE ===")
    
    # Initialize predictor
    predictor = EnhancedHousePricePredictor(data_path="../data/raw/df_imputed.csv")
    
    try:
        # Load data with larger sample
        df = predictor.load_data(sample_size=80000)  # Use 80K sample for faster execution
        
        # Basic EDA (skip plots for faster execution)
        print("Skipping EDA plots for faster execution...")
        
        # Enhanced preprocessing
        X, y, outlier_summary = predictor.enhanced_preprocessing()
        
        # Split data
        predictor.split_data()
        
        # Feature selection (more features due to better engineering)
        predictor.feature_selection(k_features=12)
        
        # Train models
        predictor.train_models()
        
        # Hyperparameter tuning
        predictor.hyperparameter_tuning()
        
        # Final evaluation
        results = predictor.final_evaluation()
        
        # Save model
        import os
        os.makedirs('../models/trained', exist_ok=True)
        predictor.save_model('../models/trained/enhanced_house_price_model.pkl')
        
        print("\n=== ENHANCED PIPELINE COMPLETED SUCCESSFULLY ===")
        print(f"Final Model Performance:")
        print(f"  - Test R²: {results['test_r2']:.4f}")
        print(f"  - Test RMSE: ${results['test_rmse_price']:,.0f}")
        print(f"  - Median Error: {results['median_percentage_error']:.1f}%")
        
    except FileNotFoundError:
        print("Error: Could not find the dataset file. Please check the path.")
        print("Expected path: ../data/raw/df_imputed.csv")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()