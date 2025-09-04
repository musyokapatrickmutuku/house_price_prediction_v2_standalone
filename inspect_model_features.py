"""
Inspect the trained model to see what features it expects
"""
import joblib
import pandas as pd
import numpy as np

def inspect_model_features():
    print("INSPECTING TRAINED MODEL FEATURES")
    print("=" * 50)
    
    try:
        # Load the model
        model_data = joblib.load('models/trained/advanced_model.pkl')
        
        print("Model components found:")
        for key in model_data.keys():
            print(f"  - {key}")
        
        print("\nFeature information:")
        
        # Check selector (feature selection)
        if 'selector' in model_data:
            selector = model_data['selector']
            if hasattr(selector, 'feature_names_in_'):
                print(f"Original features (selector): {len(selector.feature_names_in_)}")
                for i, name in enumerate(selector.feature_names_in_):
                    print(f"  {i+1:2d}. {name}")
            else:
                print("Selector doesn't have feature_names_in_")
        
        # Check scaler
        if 'scaler' in model_data:
            scaler = model_data['scaler']
            if hasattr(scaler, 'feature_names_in_'):
                print(f"\nScaler input features: {len(scaler.feature_names_in_)}")
                for i, name in enumerate(scaler.feature_names_in_):
                    print(f"  {i+1:2d}. {name}")
            else:
                print("\nScaler doesn't have feature_names_in_")
        
        # Check model
        if 'best_model' in model_data:
            model = model_data['best_model']
            if hasattr(model, 'feature_names_in_'):
                print(f"\nModel input features: {len(model.feature_names_in_)}")
                for i, name in enumerate(model.feature_names_in_):
                    print(f"  {i+1:2d}. {name}")
            else:
                print("\nModel doesn't have feature_names_in_")
        
        # Check stored feature names
        if 'selected_feature_names' in model_data:
            print(f"\nStored selected features: {len(model_data['selected_feature_names'])}")
            for i, name in enumerate(model_data['selected_feature_names']):
                print(f"  {i+1:2d}. {name}")
        
        if 'feature_names' in model_data:
            print(f"\nStored original features: {len(model_data['feature_names'])}")
            for i, name in enumerate(model_data['feature_names']):
                print(f"  {i+1:2d}. {name}")
        
        return model_data
        
    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_data = inspect_model_features()