"""
Lightweight House Price Prediction Model
Optimized for resource-constrained environments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, data_path=None):
        """Initialize the house price predictor"""
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def load_data(self, sample_size=None):
        """Load and optionally sample the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Sample data if specified for faster processing
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows for faster processing")
        
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def basic_eda(self):
        """Perform basic exploratory data analysis"""
        print("\n=== BASIC DATA OVERVIEW ===")
        print(self.df.info())
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        # Plot price distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['price'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(np.log1p(self.df['price']), bins=50, alpha=0.7, edgecolor='black')
        plt.title('Log-Transformed Price Distribution')
        plt.xlabel('Log(Price)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
    def lightweight_preprocessing(self):
        """Efficient preprocessing for large datasets"""
        print("\n=== PREPROCESSING ===")
        df = self.df.copy()
        
        # Create log-transformed target
        df['price_log'] = np.log1p(df['price'])
        
        # Handle outliers using quantile-based clipping (more efficient than IQR)
        numerical_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
        
        print("Handling outliers...")
        for col in numerical_cols:
            q_low = df[col].quantile(0.05)
            q_high = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=q_low, upper=q_high)
        
        # Efficient categorical encoding using simple mean encoding
        print("Encoding categorical features...")
        
        # Create state_city combination
        df['state_city'] = df['state'] + "_" + df['city']
        
        # Simple target encoding (more efficient than K-fold)
        global_mean = df['price_log'].mean()
        
        # Use sample for encoding if dataset is large
        sample_size = min(50000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        state_city_encoding = sample_df.groupby('state_city')['price_log'].mean().to_dict()
        df['state_city_encoded'] = df['state_city'].map(state_city_encoding).fillna(global_mean)
        
        # One-hot encode status (low cardinality)
        df = pd.get_dummies(df, columns=['status'], prefix='status', drop_first=True)
        
        # Create engineered features
        print("Creating engineered features...")
        df['price_per_sqft'] = df['price'] / (df['house_size'] + 1e-5)
        df['lot_size_per_bed'] = df['acre_lot'] / (df['bed'] + 1e-5)
        df['bed_bath_ratio'] = df['bath'] / (df['bed'] + 1e-5)
        df['size_per_bed'] = df['house_size'] / (df['bed'] + 1e-5)
        
        # Select features for modeling
        feature_cols = [
            'brokered_by', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 
            'house_size', 'state_city_encoded', 'price_per_sqft', 
            'lot_size_per_bed', 'bed_bath_ratio', 'size_per_bed'
        ]
        
        # Add status columns if they exist
        status_cols = [col for col in df.columns if col.startswith('status_')]
        feature_cols.extend(status_cols)
        
        # Keep only available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        self.X = df[available_features]
        self.y = df['price_log']
        self.feature_names = available_features
        
        print(f"Features selected: {len(available_features)}")
        return self.X, self.y
    
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
    
    def feature_selection(self, k_features=10):
        """Select top k features using univariate selection"""
        print(f"\n=== FEATURE SELECTION (top {k_features}) ===")
        
        selector = SelectKBest(score_func=f_regression, k=k_features)
        
        # Fit on training data only
        self.X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        self.X_val_selected = selector.transform(self.X_val)
        self.X_test_selected = selector.transform(self.X_test)
        
        # Get selected feature names
        selected_features = selector.get_support(indices=True)
        self.selected_feature_names = [self.feature_names[i] for i in selected_features]
        
        print("Selected features:")
        for i, feature in enumerate(self.selected_feature_names):
            score = selector.scores_[selected_features[i]]
            print(f"  {feature}: {score:.2f}")
        
        return self.X_train_selected, self.X_val_selected, self.X_test_selected
    
    def train_models(self):
        """Train multiple lightweight models and select the best"""
        print("\n=== MODEL TRAINING ===")
        
        # Define lightweight models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
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
        """Simple hyperparameter tuning for the best model"""
        print(f"\n=== HYPERPARAMETER TUNING FOR {self.best_model_name} ===")
        
        X_train_scaled = self.scaler.transform(self.X_train_selected)
        
        if self.best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            # Tune regularization parameter
            alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
            
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
                model_class(random_state=42, max_iter=2000), 
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
            
            print(f"Best alpha: {best_alpha}")
            print(f"Best validation RMSE: {val_rmse_mean[best_alpha_idx]:.4f}")
            
            # Retrain with best alpha
            if self.best_model_name == 'Ridge':
                self.best_model = Ridge(alpha=best_alpha, random_state=42)
            elif self.best_model_name == 'Lasso':
                self.best_model = Lasso(alpha=best_alpha, random_state=42, max_iter=2000)
            else:
                self.best_model = ElasticNet(alpha=best_alpha, l1_ratio=0.5, random_state=42, max_iter=2000)
            
            self.best_model.fit(X_train_scaled, self.y_train)
    
    def final_evaluation(self):
        """Evaluate the final model on test set"""
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
        
        print(f"Test RMSE (log scale): {test_rmse:.4f}")
        print(f"Test MAE (log scale): {test_mae:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE (price scale): ${price_rmse:,.0f}")
        print(f"Test MAE (price scale): ${price_mae:,.0f}")
        
        # Create prediction vs actual plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, test_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Log(Price)')
        plt.ylabel('Predicted Log(Price)')
        plt.title('Predictions vs Actual (Log Scale)')
        
        plt.subplot(1, 2, 2)
        residuals = test_pred - self.y_test
        plt.scatter(test_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Log(Price)')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'test_rmse_log': test_rmse,
            'test_mae_log': test_mae,
            'test_r2': test_r2,
            'test_rmse_price': price_rmse,
            'test_mae_price': price_mae
        }
    
    def save_model(self, model_path='house_price_model.pkl'):
        """Save the trained model and preprocessing components"""
        print(f"\nSaving model to {model_path}")
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.selected_feature_names,
            'model_name': self.best_model_name
        }
        
        joblib.dump(model_package, model_path)
        print("Model saved successfully!")
        
        return model_path
    
    def load_model(self, model_path='house_price_model.pkl'):
        """Load a saved model"""
        print(f"Loading model from {model_path}")
        
        model_package = joblib.load(model_path)
        self.best_model = model_package['model']
        self.scaler = model_package['scaler']
        self.selected_feature_names = model_package['feature_names']
        self.best_model_name = model_package['model_name']
        
        print(f"Loaded {self.best_model_name} model successfully!")
        
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
    """Main execution function"""
    print("=== LIGHTWEIGHT HOUSE PRICE PREDICTION ===")
    
    # Initialize predictor
    predictor = HousePricePredictor(data_path="../data/raw/df_imputed.csv")
    
    try:
        # Load data (sample for faster processing)
        df = predictor.load_data(sample_size=50000)  # Use smaller sample for speed
        
        # Basic EDA (skip for faster execution)
        # predictor.basic_eda()
        
        # Preprocessing
        X, y = predictor.lightweight_preprocessing()
        
        # Split data
        predictor.split_data()
        
        # Feature selection
        predictor.feature_selection(k_features=8)  # Select top 8 features
        
        # Train models
        predictor.train_models()
        
        # Hyperparameter tuning
        predictor.hyperparameter_tuning()
        
        # Final evaluation
        results = predictor.final_evaluation()
        
        # Save model
        import os
        os.makedirs('../models/trained', exist_ok=True)
        predictor.save_model('../models/trained/house_price_model_optimized.pkl')
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        
        # Example prediction
        print("\n=== EXAMPLE PREDICTION ===")
        sample_features = {
            'brokered_by': 50000,
            'bed': 3,
            'bath': 2,
            'acre_lot': 0.25,
            'street': 1000000,
            'zip_code': 12345,
            'house_size': 1800,
            'state_city_encoded': 12.5,
            'price_per_sqft': 150,
            'lot_size_per_bed': 0.083,
            'bed_bath_ratio': 0.67,
            'size_per_bed': 600
        }
        
        # Only use features that were actually selected
        available_sample = {k: v for k, v in sample_features.items() 
                          if k in predictor.selected_feature_names}
        
        if available_sample:
            predicted_price = predictor.predict_price(available_sample)
            print(f"Predicted price: ${predicted_price:,.0f}")
        
    except FileNotFoundError:
        print("Error: Could not find the dataset file. Please check the path.")
        print("Expected path: ../data/raw/df_imputed.csv")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()