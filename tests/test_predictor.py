"""
Unit tests for the HousePricePredictor class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import the predictor classes
import sys
sys.path.append('../src')
from house_price_predictor import HousePricePredictor
from enhanced_predictor import EnhancedHousePricePredictor


class TestHousePricePredictor:
    """Test cases for the basic HousePricePredictor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'price': np.random.uniform(100000, 1000000, n_samples),
            'bed': np.random.randint(1, 6, n_samples),
            'bath': np.random.randint(1, 4, n_samples),
            'acre_lot': np.random.uniform(0.1, 2.0, n_samples),
            'house_size': np.random.uniform(800, 4000, n_samples),
            'city': np.random.choice(['CityA', 'CityB', 'CityC'], n_samples),
            'state': np.random.choice(['CA', 'TX', 'NY'], n_samples),
            'status': np.random.choice(['for_sale', 'sold'], n_samples),
            'brokered_by': np.random.randint(1000, 9999, n_samples),
            'street': np.random.randint(10000, 99999, n_samples),
            'zip_code': np.random.randint(10000, 99999, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def predictor(self):
        """Create a predictor instance."""
        return HousePricePredictor()
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.data_path is None
        assert predictor.model is None
        assert predictor.feature_names is None
        assert predictor.scaler is None
    
    def test_initialization_with_path(self):
        """Test predictor initialization with data path."""
        path = "/path/to/data.csv"
        predictor = HousePricePredictor(data_path=path)
        assert predictor.data_path == path
    
    def test_load_data(self, predictor, sample_data):
        """Test data loading functionality."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            predictor.data_path = temp_path
            loaded_data = predictor.load_data()
            
            assert loaded_data.shape == sample_data.shape
            assert list(loaded_data.columns) == list(sample_data.columns)
            assert hasattr(predictor, 'df')
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_with_sampling(self, predictor, sample_data):
        """Test data loading with sampling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            predictor.data_path = temp_path
            loaded_data = predictor.load_data(sample_size=500)
            
            assert loaded_data.shape[0] == 500
            assert loaded_data.shape[1] == sample_data.shape[1]
            
        finally:
            os.unlink(temp_path)
    
    def test_lightweight_preprocessing(self, predictor, sample_data):
        """Test the preprocessing pipeline."""
        predictor.df = sample_data
        X, y = predictor.lightweight_preprocessing()
        
        # Check that target is log-transformed
        assert hasattr(predictor, 'X')
        assert hasattr(predictor, 'y')
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert predictor.feature_names is not None
    
    def test_split_data(self, predictor, sample_data):
        """Test data splitting."""
        predictor.df = sample_data
        predictor.lightweight_preprocessing()
        
        train, val, test, y_train, y_val, y_test = predictor.split_data()
        
        total_samples = len(sample_data)
        expected_test = int(total_samples * 0.2)
        expected_val = int(total_samples * 0.2)
        expected_train = total_samples - expected_test - expected_val
        
        # Allow for small rounding differences
        assert abs(len(train) - expected_train) <= 2
        assert abs(len(val) - expected_val) <= 2
        assert abs(len(test) - expected_test) <= 2
    
    def test_feature_selection(self, predictor, sample_data):
        """Test feature selection."""
        predictor.df = sample_data
        predictor.lightweight_preprocessing()
        predictor.split_data()
        
        train_selected, val_selected, test_selected = predictor.feature_selection(k_features=5)
        
        assert train_selected.shape[1] == 5
        assert val_selected.shape[1] == 5
        assert test_selected.shape[1] == 5
        assert len(predictor.selected_feature_names) == 5
    
    def test_train_models(self, predictor, sample_data):
        """Test model training."""
        predictor.df = sample_data
        predictor.lightweight_preprocessing()
        predictor.split_data()
        predictor.feature_selection(k_features=5)
        
        results = predictor.train_models()
        
        assert isinstance(results, dict)
        assert 'Linear' in results
        assert 'Ridge' in results
        assert 'Lasso' in results
        assert 'ElasticNet' in results
        
        # Check that best model is selected
        assert predictor.best_model is not None
        assert predictor.best_model_name is not None
    
    def test_predict_price_without_model(self, predictor):
        """Test prediction without trained model raises error."""
        features = {'bed': 3, 'bath': 2}
        
        with pytest.raises(ValueError, match="No model trained"):
            predictor.predict_price(features)
    
    def test_predict_price_with_dict(self, predictor, sample_data):
        """Test prediction with dictionary input."""
        # Train a simple model
        predictor.df = sample_data
        predictor.lightweight_preprocessing()
        predictor.split_data()
        predictor.feature_selection(k_features=5)
        predictor.train_models()
        
        # Create prediction features matching selected features
        sample_features = {}
        for feature in predictor.selected_feature_names:
            if feature in ['bed', 'bath']:
                sample_features[feature] = 3
            elif feature in ['house_size', 'acre_lot']:
                sample_features[feature] = 1500.0
            else:
                sample_features[feature] = 100.0
        
        prediction = predictor.predict_price(sample_features)
        
        assert isinstance(prediction, (int, float, np.number))
        assert prediction > 0
    
    def test_save_and_load_model(self, predictor, sample_data):
        """Test model saving and loading."""
        # Train a model
        predictor.df = sample_data
        predictor.lightweight_preprocessing()
        predictor.split_data()
        predictor.feature_selection(k_features=5)
        predictor.train_models()
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            saved_path = predictor.save_model(model_path)
            assert saved_path == model_path
            assert os.path.exists(model_path)
            
            # Load model in new predictor
            new_predictor = HousePricePredictor()
            new_predictor.load_model(model_path)
            
            assert new_predictor.best_model is not None
            assert new_predictor.scaler is not None
            assert new_predictor.selected_feature_names is not None
            assert new_predictor.best_model_name is not None
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestEnhancedHousePricePredictor:
    """Test cases for the enhanced predictor."""
    
    @pytest.fixture
    def enhanced_predictor(self):
        """Create an enhanced predictor instance."""
        return EnhancedHousePricePredictor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'price': np.random.uniform(100000, 1000000, n_samples),
            'bed': np.random.randint(1, 6, n_samples),
            'bath': np.random.randint(1, 4, n_samples),
            'acre_lot': np.random.uniform(0.1, 2.0, n_samples),
            'house_size': np.random.uniform(800, 4000, n_samples),
            'city': np.random.choice(['CityA', 'CityB', 'CityC'], n_samples),
            'state': np.random.choice(['CA', 'TX', 'NY'], n_samples),
            'status': np.random.choice(['for_sale', 'sold'], n_samples),
            'brokered_by': np.random.randint(1000, 9999, n_samples),
            'street': np.random.randint(10000, 99999, n_samples),
            'zip_code': np.random.randint(10000, 99999, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_enhanced_initialization(self, enhanced_predictor):
        """Test enhanced predictor initialization."""
        assert enhanced_predictor.data_path is None
        assert enhanced_predictor.model is None
        assert enhanced_predictor.city_state_encoder is None
    
    def test_advanced_outlier_handling(self, enhanced_predictor, sample_data):
        """Test advanced outlier handling."""
        enhanced_predictor.df = sample_data
        
        numerical_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
        cleaned_df, outlier_summary = enhanced_predictor.advanced_outlier_handling(
            sample_data, numerical_cols
        )
        
        assert cleaned_df.shape == sample_data.shape
        assert isinstance(outlier_summary, dict)
        
        for col in numerical_cols:
            assert col in outlier_summary
            assert 'outliers_capped' in outlier_summary[col]
            assert 'outlier_percentage' in outlier_summary[col]
    
    def test_target_encoding_with_cv(self, enhanced_predictor, sample_data):
        """Test cross-validated target encoding."""
        enhanced_predictor.df = sample_data
        sample_data['price_log'] = np.log1p(sample_data['price'])
        sample_data['city_state'] = sample_data['state'] + "_" + sample_data['city']
        
        encoded_values = enhanced_predictor.target_encoding_with_cv(
            sample_data, 'city_state', 'price_log'
        )
        
        assert len(encoded_values) == len(sample_data)
        assert enhanced_predictor.city_state_encoder is not None
        assert 'encoding_map' in enhanced_predictor.city_state_encoder
        assert 'global_mean' in enhanced_predictor.city_state_encoder
    
    def test_enhanced_preprocessing(self, enhanced_predictor, sample_data):
        """Test enhanced preprocessing pipeline."""
        enhanced_predictor.df = sample_data
        X, y, outlier_summary = enhanced_predictor.enhanced_preprocessing()
        
        assert hasattr(enhanced_predictor, 'X')
        assert hasattr(enhanced_predictor, 'y')
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert isinstance(outlier_summary, dict)
        assert enhanced_predictor.feature_names is not None
        
        # Check that enhanced features are created
        expected_features = ['price_per_sqft', 'bed_bath_ratio', 'total_rooms']
        for feature in expected_features:
            if feature in enhanced_predictor.feature_names:
                assert feature in X.columns


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])