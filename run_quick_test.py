"""
Quick test script to verify the project works correctly
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from house_price_predictor import HousePricePredictor
from enhanced_predictor import EnhancedHousePricePredictor

def test_basic_predictor():
    """Test the basic predictor with minimal data"""
    print("=== Testing Basic House Price Predictor ===")
    
    try:
        # Initialize predictor
        predictor = HousePricePredictor(data_path="data/raw/df_imputed.csv")
        
        # Load small sample
        df = predictor.load_data(sample_size=500)
        
        # Preprocessing
        X, y = predictor.lightweight_preprocessing()
        
        # Split data
        predictor.split_data()
        
        # Feature selection
        predictor.feature_selection(k_features=5)  # Use fewer features for speed
        
        # Train models
        predictor.train_models()
        
        # Quick evaluation
        results = predictor.final_evaluation()
        
        print(f"+ Basic predictor test completed successfully!")
        print(f"  - Best model: {predictor.best_model_name}")
        print(f"  - Test R²: {results['test_r2']:.4f}")
        print(f"  - Test RMSE: ${results['test_rmse_price']:,.0f}")
        
        # Test prediction
        sample_features = {}
        for feature in predictor.selected_feature_names:
            if 'bed' in feature or 'bath' in feature:
                sample_features[feature] = 3
            elif 'size' in feature.lower():
                sample_features[feature] = 1500.0
            else:
                sample_features[feature] = 100.0
        
        predicted_price = predictor.predict_price(sample_features)
        print(f"  - Test prediction: ${predicted_price:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"X Basic predictor test failed: {str(e)}")
        return False

def test_enhanced_predictor():
    """Test the enhanced predictor with minimal data"""
    print("\n=== Testing Enhanced House Price Predictor ===")
    
    try:
        # Initialize predictor
        predictor = EnhancedHousePricePredictor(data_path="data/raw/df_imputed.csv")
        
        # Load small sample
        df = predictor.load_data(sample_size=500)
        
        # Enhanced preprocessing
        X, y, outlier_summary = predictor.enhanced_preprocessing()
        
        # Split data
        predictor.split_data()
        
        # Feature selection
        predictor.feature_selection(k_features=8)  # Use fewer features for speed
        
        # Train models
        predictor.train_models()
        
        # Quick evaluation
        results = predictor.final_evaluation()
        
        print(f"+ Enhanced predictor test completed successfully!")
        print(f"  - Best model: {predictor.best_model_name}")
        print(f"  - Test R²: {results['test_r2']:.4f}")
        print(f"  - Test RMSE: ${results['test_rmse_price']:,.0f}")
        print(f"  - Median Error: {results['median_percentage_error']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"X Enhanced predictor test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running Quick Tests for House Price Prediction Project")
    print("=" * 60)
    
    # Test basic predictor
    basic_success = test_basic_predictor()
    
    # Test enhanced predictor
    enhanced_success = test_enhanced_predictor()
    
    print("\n" + "=" * 60)
    if basic_success and enhanced_success:
        print(">> All tests passed! The project is working correctly.")
    else:
        print("!! Some tests failed. Please check the errors above.")
    
    print("\nTo run the full pipeline:")
    print("  python src/house_price_predictor.py")
    print("  python src/enhanced_predictor.py")