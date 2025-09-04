"""
Test the fixed prediction functionality
"""
import sys
sys.path.insert(0, '.')

from app import make_prediction
import joblib
import os

def load_model_safe():
    """Load model safely for testing"""
    try:
        model_path = 'models/trained/advanced_model.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            return model_data, True, None
        else:
            return None, False, "Model file not found"
    except Exception as e:
        return None, False, f"Error loading model: {str(e)}"

def test_prediction():
    print("TESTING FIXED PREDICTION FUNCTIONALITY")
    print("=" * 50)
    
    # Test inputs
    test_inputs = {
        'bed': 3,
        'bath': 2.0,
        'house_size': 2000,
        'acre_lot': 0.25,
        'city': 'Los Angeles',
        'state': 'CA',
        'status': 'for_sale',
        'street': 123,
        'zip_code': 90210,
        'brokered_by': 1
    }
    
    print("Test inputs:")
    for key, value in test_inputs.items():
        print(f"  {key}: {value}")
    
    # Load model
    print("\nLoading model...")
    model_data, success, error = load_model_safe()
    
    if not success:
        print(f"[X] Model loading failed: {error}")
        return
    
    print("[OK] Model loaded successfully")
    print(f"Model components: {list(model_data.keys())}")
    
    # Test prediction
    print("\nTesting prediction...")
    try:
        prediction, success, error = make_prediction(model_data, test_inputs)
        
        if success:
            print(f"[OK] Prediction successful!")
            print(f"Predicted price: ${prediction:,.0f}")
            
            # Additional calculations
            price_per_sqft = prediction / test_inputs['house_size']
            print(f"Price per sq ft: ${price_per_sqft:.0f}")
            
        else:
            print(f"[X] Prediction failed: {error}")
            
    except Exception as e:
        print(f"[X] Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()