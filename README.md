# House Price Prediction Project

A lightweight, resource-efficient machine learning pipeline for predicting house prices in the USA, optimized for systems with limited computational resources.

## 🚀 Quick Start

```bash
# Clone or download the project
cd house_price_prediction_v2

# Install dependencies
pip install -r requirements.txt

# Place your dataset in data/raw/
# Expected file: data/raw/df_imputed.csv

# Run the prediction pipeline
cd src
python house_price_predictor.py
```

## 📁 Project Structure

```
house_price_prediction_v2/
├── CLAUDE.md                    # Detailed project plan and scope
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   ├── house_price_predictor.py # Main prediction pipeline
│   └── enhanced_predictor.py    # Enhanced version with better features
├── data/
│   ├── raw/                     # Original dataset (place df_imputed.csv here)
│   ├── processed/               # Cleaned data
│   └── sample/                  # Sample datasets
├── models/
│   ├── trained/                 # Saved models
│   └── experiments/             # Experiment results
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit tests
├── config/                      # Configuration files
└── docs/                        # Additional documentation
```

## 🎯 Features

- **Resource Efficient**: Optimized for systems with limited RAM (< 4GB)
- **Fast Training**: Complete pipeline runs in under 5 minutes
- **Multiple Models**: Compares Linear, Ridge, Lasso, and ElasticNet regression
- **Smart Preprocessing**: Efficient outlier handling and feature engineering
- **Model Persistence**: Save and load trained models for production use
- **Sample Processing**: Handles large datasets through intelligent sampling

## 📊 Model Performance

- **Target RMSE**: < 0.5 on log-transformed prices
- **Memory Usage**: < 4GB during training
- **Training Time**: < 5 minutes on standard hardware
- **Model Size**: < 10MB for deployment

## 🛠️ Technical Details

### Dataset Requirements
- **Format**: CSV file with headers
- **Expected columns**: price, bed, bath, acre_lot, house_size, city, state, status, brokered_by, street, zip_code
- **Size**: Optimized for 1M+ records

### Model Pipeline
1. **Data Loading**: Optional sampling for faster processing
2. **EDA**: Basic exploratory data analysis with visualizations
3. **Preprocessing**: Outlier handling, target encoding, feature engineering
4. **Feature Selection**: SelectKBest with f_regression scoring
5. **Model Training**: Multiple lightweight models with automatic selection
6. **Hyperparameter Tuning**: Validation curves for optimal parameters
7. **Evaluation**: Comprehensive metrics on holdout test set
8. **Model Saving**: Persistent storage for deployment

### Key Optimizations
- **Quantile-based outlier clipping** instead of IQR method
- **Sample-based target encoding** for high-cardinality features
- **Simple train/val/test split** instead of cross-validation
- **RobustScaler** for better outlier handling
- **Memory-efficient preprocessing** with chunked operations

## 📈 Usage Examples

### Training a Model
```python
from src.house_price_predictor import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor(data_path="data/raw/df_imputed.csv")

# Load and process data
predictor.load_data(sample_size=100000)  # Use sample for speed
predictor.lightweight_preprocessing()
predictor.split_data()
predictor.feature_selection(k_features=8)

# Train and evaluate
predictor.train_models()
predictor.hyperparameter_tuning()
results = predictor.final_evaluation()

# Save model
predictor.save_model("models/trained/my_model.pkl")
```

### Making Predictions
```python
# Load saved model
predictor = HousePricePredictor()
predictor.load_model("models/trained/my_model.pkl")

# Predict price for new house
house_features = {
    'bed': 3,
    'bath': 2,
    'house_size': 1800,
    'acre_lot': 0.25,
    # ... other features
}

predicted_price = predictor.predict_price(house_features)
print(f"Predicted price: ${predicted_price:,.0f}")
```

## 🔧 Configuration

### Memory Settings
- **Default sample size**: 100,000 rows
- **Chunk size**: 50,000 rows for encoding
- **Feature selection**: Top 8 features
- **Cross-validation folds**: 3 (for hyperparameter tuning)

### Model Settings
- **Regularization range**: [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
- **Max iterations**: 2000 for iterative solvers
- **Random state**: 42 for reproducibility

## 📋 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **joblib**: Model serialization

## 🧪 Testing

```bash
# Run unit tests (when implemented)
python -m pytest tests/

# Validate on sample data
python src/house_price_predictor.py
```

## 📚 Documentation

- **CLAUDE.md**: Comprehensive project plan and technical specifications
- **notebooks/**: Jupyter notebooks for detailed analysis
- **docs/**: API documentation and deployment guides

## 🚀 Deployment

### Local Deployment
```bash
# Save trained model
python -c "
from src.house_price_predictor import HousePricePredictor
predictor = HousePricePredictor('data/raw/df_imputed.csv')
# ... train model ...
predictor.save_model('models/trained/production_model.pkl')
"

# Use in production
from src.house_price_predictor import HousePricePredictor
predictor = HousePricePredictor()
predictor.load_model('models/trained/production_model.pkl')
```

### Production Considerations
- **Model monitoring**: Track prediction drift over time
- **Data validation**: Ensure input data quality
- **Error handling**: Graceful failure for invalid inputs
- **Logging**: Track predictions and performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes and add tests
4. Commit changes (`git commit -am 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Create Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Memory Error during training**
- Reduce `sample_size` parameter
- Decrease `k_features` in feature selection
- Use smaller chunks for preprocessing

**File not found error**
- Ensure dataset is in `data/raw/df_imputed.csv`
- Check file path in configuration
- Verify file permissions

**Poor model performance**
- Increase sample size if memory allows
- Add more features to selection
- Try different regularization parameters

### Performance Tips

- **Use SSD storage** for faster data loading
- **Increase RAM** for larger sample sizes
- **Use GPU** for sklearn operations (if available)
- **Profile memory usage** with memory_profiler

## 📞 Support

For issues and questions:
- Check existing documentation in `docs/`
- Review the comprehensive project plan in `CLAUDE.md`
- Create an issue with detailed error information

---

**Last Updated**: 2025-01-28  
**Version**: 2.0.0  
**Python Compatibility**: 3.8+