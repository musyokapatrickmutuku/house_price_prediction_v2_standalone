"""
Test Advanced House Price Predictor with Comprehensive Error Analysis
"""
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')
from advanced_predictor import AdvancedHousePricePredictor

def test_advanced_predictor():
    print('ADVANCED HOUSE PRICE PREDICTOR TEST')
    print('=' * 60)
    print('Implementing data scientist recommendations:')
    print('1. Robust data cleaning (remove $0 prices, extreme outliers)')
    print('2. Advanced feature engineering (25+ features)')
    print('3. Ensemble methods (Random Forest, Gradient Boosting)')
    print('4. Stratified sampling and robust scaling')
    print('5. Comprehensive error metrics and bounds')
    print('=' * 60)
    
    start_time = time.time()
    
    try:
        # Initialize advanced predictor
        predictor = AdvancedHousePricePredictor('data/raw/df_imputed.csv')
        
        # Load and clean data
        print('\n[STEP 1] Loading and cleaning data...')
        df_raw = predictor.load_data(sample_size=100000)
        df_clean = predictor.robust_data_cleaning()
        
        cleaning_improvement = len(df_raw) - len(df_clean)
        print(f'Data quality improvement: Removed {cleaning_improvement:,} problematic records')
        
        # Advanced feature engineering
        print('\n[STEP 2] Advanced feature engineering...')
        df_features = predictor.advanced_feature_engineering()
        
        # Split data with stratification
        print('\n[STEP 3] Stratified data splitting...')
        predictor.split_data(test_size=0.2, val_size=0.2)
        
        # Feature selection and scaling
        print('\n[STEP 4] Feature selection and robust scaling...')
        predictor.feature_selection_and_scaling(k_features=20)
        
        # Train advanced models
        print('\n[STEP 5] Training advanced models...')
        model_results = predictor.train_advanced_models()
        
        # Final evaluation
        print('\n[STEP 6] Comprehensive evaluation...')
        results = predictor.final_evaluation()
        
        total_time = time.time() - start_time
        
        # Save model
        os.makedirs('models/trained', exist_ok=True)
        predictor.save_model('models/trained/advanced_model.pkl')
        
        # === COMPREHENSIVE RESULTS ANALYSIS ===
        print('\n' + '=' * 60)
        print('ADVANCED PREDICTOR - FINAL RESULTS')
        print('=' * 60)
        
        print('\n[DATA QUALITY] CLEANING RESULTS:')
        print(f'  Original dataset: {len(df_raw):,} records')
        print(f'  After cleaning: {len(df_clean):,} records')
        print(f'  Records removed: {cleaning_improvement:,} ({cleaning_improvement/len(df_raw)*100:.1f}%)')
        print(f'  Data quality improvement: Addressed $0 prices and extreme outliers')
        
        print('\n[FEATURES] ENGINEERING RESULTS:')
        print(f'  Total features created: {df_features.shape[1]}')
        print(f'  Selected for modeling: {len(predictor.selected_feature_names)}')
        print(f'  Feature types: Location, Price ratios, Property metrics, Statistical')
        
        print('\n[MODEL] SELECTION RESULTS:')
        print(f'  Best algorithm: {predictor.best_model_name}')
        print(f'  Models tested: {len(model_results)}')
        print(f'  Selection criteria: Validation RMSE + Overfitting penalty')
        
        print('\n[PERFORMANCE] COMPREHENSIVE METRICS:')
        print(f'  R² Score: {results["test_r2"]:.4f} (variance explained)')
        print(f'  RMSE (log): {results["test_rmse_log"]:.4f}')
        print(f'  MAE (log): {results["test_mae_log"]:.4f}')
        print()
        print(f'  RMSE (price): ${results["test_rmse_price"]:,.0f}')
        print(f'  MAE (price): ${results["test_mae_price"]:,.0f}')
        print()
        print(f'  Median Error: {results["median_percentage_error"]:.1f}%')
        print(f'  Mean Error: {results["mean_percentage_error"]:.1f}%')
        print(f'  SMAPE: {results["smape"]:.1f}%')
        print(f'  Bounded MAPE: {results["mape_bounded"]:.1f}%')
        
        print('\n[ACCURACY] ERROR DISTRIBUTION:')
        print(f'  Predictions within 10%: {results["error_under_10pct"]*100:.1f}%')
        print(f'  Predictions within 20%: {results["error_under_20pct"]*100:.1f}%')
        print(f'  Predictions within 30%: {results["error_under_30pct"]*100:.1f}%')
        
        print('\n[BUSINESS IMPACT] IMPROVEMENT ANALYSIS:')
        
        # Compare with previous results (Enhanced Predictor baseline)
        baseline_median_error = 28.7  # From previous enhanced predictor
        baseline_rmse_price = 1805021
        baseline_mae_price = 270852
        
        median_improvement = baseline_median_error - results["median_percentage_error"]
        rmse_improvement = (baseline_rmse_price - results["test_rmse_price"]) / baseline_rmse_price * 100
        mae_improvement = (baseline_mae_price - results["test_mae_price"]) / baseline_mae_price * 100
        
        print(f'  Median Error Improvement: {median_improvement:.1f} percentage points')
        print(f'  RMSE Improvement: {rmse_improvement:.1f}%')
        print(f'  MAE Improvement: {mae_improvement:.1f}%')
        
        # Business recommendations
        print('\n[DEPLOYMENT] BUSINESS READINESS:')
        if results["median_percentage_error"] < 15:
            print('  [EXCELLENT] Ready for production deployment')
            print('  [REASON] Median error < 15% meets business standards')
        elif results["median_percentage_error"] < 20:
            print('  [GOOD] Ready for production with monitoring')
            print('  [REASON] Median error < 20% acceptable for most use cases')
        elif results["median_percentage_error"] < 25:
            print('  [ACCEPTABLE] Deploy with caution and user education')
            print('  [REASON] Median error 20-25% requires user awareness')
        else:
            print('  [NEEDS IMPROVEMENT] Additional optimization required')
            print('  [REASON] Median error > 25% too high for business use')
        
        print(f'\n[PERFORMANCE] EXECUTION TIME: {total_time:.1f} seconds')
        
        print('\n[RECOMMENDATIONS] NEXT STEPS:')
        if results["test_r2"] > 0.8:
            print('  - Model shows excellent predictive power')
        elif results["test_r2"] > 0.7:
            print('  - Model shows good predictive power')
        else:
            print('  - Consider additional feature engineering or ensemble methods')
            
        if results["error_under_20pct"] > 0.7:
            print('  - High accuracy rate suitable for most business applications')
        else:
            print('  - Consider focusing on error reduction strategies')
            
        print('  - Implement prediction intervals for uncertainty quantification')
        print('  - Set up model monitoring for production deployment')
        print('  - Consider A/B testing against simpler models for cost-benefit analysis')
        
        print('\n' + '=' * 60)
        print('ADVANCED PREDICTOR TEST COMPLETED SUCCESSFULLY!')
        print('=' * 60)
        
        return results, predictor
        
    except Exception as e:
        print(f'\n[ERROR] Test failed: {str(e)}')
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, predictor = test_advanced_predictor()