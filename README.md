# 🏠 House Price Prediction V2

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

A lightweight and efficient machine learning pipeline for predicting house prices, optimized for systems with limited computational resources.

---

### ✨ Key Features

- **⚡️ Efficient**: Optimized for low RAM usage (< 4GB) and fast training (< 5 mins).
- **⚖️ Multiple Models**: Compares Linear, Ridge, Lasso, and ElasticNet regression.
- **🧹 Smart Preprocessing**: Handles outliers, high-cardinality features, and missing data.
- **💾 Model Persistence**: Save and load trained models for easy deployment.
- **⚙️ Configurable**: Easily manage data paths and model parameters via YAML files.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip` for package management

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd house_price_prediction_v2_standalone
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place your data:**
    Ensure your dataset is located at `data/raw/df_imputed.csv`. Sample data can be generated using `create_sample_data.py`.

---

## ⚡️ Run a Quick Test

To verify the setup and run a prediction with the pre-trained model, execute the quick test script:

```bash
python run_quick_test.py
```
This will load the optimized model from `models/trained/` and print a sample prediction.

---

## 📁 Project Structure

```
.
├── config/              # YAML files for data and model configuration
├── data/                # Raw, processed, and sample datasets
├── docs/                # Project documentation
├── models/              # Trained and experimental models
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code for the prediction pipeline
│   ├── house_price_predictor.py  # Main pipeline script
│   └── enhanced_predictor.py     # Enhanced pipeline version
├── tests/               # Unit and integration tests
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

---

## 🛠️ Usage

### Training a New Model

You can run the main pipeline script to process data, train, evaluate, and save a new model.

```bash
python src/house_price_predictor.py
```
The script uses configuration from `config/data_config.yaml` and `config/model_config.yaml`. The final trained model will be saved in the `models/trained/` directory.

### Making Predictions with a Script

The following example shows how to load a trained model and make a prediction.

```python
from src.house_price_predictor import HousePricePredictor
import pandas as pd

# Initialize predictor and load the trained model
predictor = HousePricePredictor()
predictor.load_model("models/trained/house_price_model_optimized.pkl")

# Create a sample DataFrame for prediction
new_data = pd.DataFrame([{
    'bed': 4,
    'bath': 3,
    'acre_lot': 0.5,
    'house_size': 2400,
    'zip_code': '12345',
    # ... add all other required features
}])

# Get the predicted price
price = predictor.predict(new_data)
print(f"Predicted Price: ${price[0]:,.2f}")
```

---

## ⚙️ Configuration

Project settings can be modified in the `config/` directory:
- **`data_config.yaml`**: Manages file paths, feature lists, and sampling settings.
- **`model_config.yaml`**: Controls model parameters, feature selection (`k_features`), and evaluation metrics.

---

## 🧪 Testing

This project uses `pytest` for automated testing. To run the test suite:

```bash
python -m pytest
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
