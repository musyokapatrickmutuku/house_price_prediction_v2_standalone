# Deployment Guide

This guide covers different deployment scenarios for the house price prediction model.

## Local Deployment

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Required packages (see requirements.txt)

### Quick Setup
```bash
# Clone/download the project
cd house_price_prediction_v2

# Install dependencies
pip install -r requirements.txt

# Place your dataset
cp your_dataset.csv data/raw/df_imputed.csv

# Train and save model
cd src
python house_price_predictor.py
```

### Using Pre-trained Model
```python
from src.house_price_predictor import HousePricePredictor

# Load pre-trained model
predictor = HousePricePredictor()
predictor.load_model("models/trained/house_price_model.pkl")

# Make prediction
features = {
    'bed': 3,
    'bath': 2,
    'house_size': 1800,
    'acre_lot': 0.25,
    # ... other required features
}

price = predictor.predict_price(features)
print(f"Predicted price: ${price:,.0f}")
```

## Production Deployment

### Model Serving Options

#### 1. Flask API
Create a simple REST API:

```python
from flask import Flask, request, jsonify
from src.house_price_predictor import HousePricePredictor
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model once at startup
predictor = HousePricePredictor()
predictor.load_model("models/trained/house_price_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction = predictor.predict_price(data)
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### 2. FastAPI (Recommended)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.house_price_predictor import HousePricePredictor
import uvicorn

app = FastAPI(title="House Price Prediction API")

# Load model
predictor = HousePricePredictor()
predictor.load_model("models/trained/house_price_model.pkl")

class HouseFeatures(BaseModel):
    bed: int
    bath: int
    house_size: float
    acre_lot: float
    # Add other required features

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        prediction = predictor.predict_price(features.dict())
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY models/ models/
COPY config/ config/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "src/api_server.py"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  house-price-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Cloud Deployment

#### AWS Lambda
```python
import json
import joblib
import numpy as np
from src.house_price_predictor import HousePricePredictor

# Global variable for model (loaded once)
predictor = None

def lambda_handler(event, context):
    global predictor
    
    # Load model if not already loaded
    if predictor is None:
        predictor = HousePricePredictor()
        predictor.load_model("/opt/model/house_price_model.pkl")
    
    try:
        # Parse input
        features = json.loads(event['body'])
        
        # Make prediction
        prediction = predictor.predict_price(features)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': float(prediction),
                'status': 'success'
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': str(e),
                'status': 'error'
            })
        }
```

#### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/house-price-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/house-price-api']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'house-price-api',
      '--image', 'gcr.io/$PROJECT_ID/house-price-api',
      '--region', 'us-central1',
      '--platform', 'managed',
      '--memory', '2Gi',
      '--cpu', '1'
    ]
```

## Performance Optimization

### Model Optimization
```python
# Use model compression
import joblib
from sklearn.externals import joblib

# Save with compression
joblib.dump(model, 'model.pkl', compress=3)

# Use pickle protocol 4 for better performance
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)
```

### Caching
```python
from functools import lru_cache
import hashlib
import json

class CachedPredictor:
    def __init__(self, model_path):
        self.predictor = HousePricePredictor()
        self.predictor.load_model(model_path)
    
    @lru_cache(maxsize=1000)
    def predict_cached(self, features_hash):
        features = json.loads(features_hash)
        return self.predictor.predict_price(features)
    
    def predict_price(self, features):
        # Create hash of features for caching
        features_str = json.dumps(features, sort_keys=True)
        features_hash = hashlib.md5(features_str.encode()).hexdigest()
        
        return self.predict_cached(features_hash)
```

## Monitoring and Logging

### Basic Logging
```python
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def predict_with_logging(predictor, features):
    start_time = time.time()
    
    try:
        prediction = predictor.predict_price(features)
        duration = time.time() - start_time
        
        logger.info(f"Prediction successful: {prediction:.2f}, Duration: {duration:.3f}s")
        return prediction
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Prediction failed: {str(e)}, Duration: {duration:.3f}s")
        raise
```

### Model Performance Monitoring
```python
import numpy as np
from collections import deque

class ModelMonitor:
    def __init__(self, window_size=1000):
        self.predictions = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        
    def log_prediction(self, prediction, response_time):
        self.predictions.append(prediction)
        self.response_times.append(response_time)
        
    def get_stats(self):
        if not self.predictions:
            return {}
            
        return {
            'prediction_count': len(self.predictions),
            'avg_prediction': np.mean(self.predictions),
            'prediction_std': np.std(self.predictions),
            'avg_response_time': np.mean(self.response_times),
            'p95_response_time': np.percentile(self.response_times, 95)
        }
```

## Security Considerations

### Input Validation
```python
from pydantic import BaseModel, validator

class HouseFeatures(BaseModel):
    bed: int
    bath: int
    house_size: float
    acre_lot: float
    
    @validator('bed')
    def validate_bed(cls, v):
        if v < 0 or v > 20:
            raise ValueError('Bedrooms must be between 0 and 20')
        return v
    
    @validator('house_size')
    def validate_house_size(cls, v):
        if v < 100 or v > 50000:
            raise ValueError('House size must be between 100 and 50,000 sq ft')
        return v
```

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Prediction logic here
    pass
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check file path and permissions
   - Verify model was saved with same sklearn version
   - Ensure all dependencies are installed

2. **Prediction Errors**
   - Validate input features match training data format
   - Check for missing required features
   - Verify data types and ranges

3. **Performance Issues**
   - Monitor memory usage during predictions
   - Consider model compression
   - Implement response caching

4. **Deployment Issues**
   - Check network connectivity and ports
   - Verify environment variables
   - Review application logs

### Health Checks
```python
@app.route('/health')
def health_check():
    try:
        # Test model loading
        test_features = {
            'bed': 3, 'bath': 2, 'house_size': 1800, 
            'acre_lot': 0.25, # ... other features
        }
        
        prediction = predictor.predict_price(test_features)
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'test_prediction': float(prediction)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
```

## Scaling Considerations

- Use load balancers for multiple instances
- Consider model serving frameworks (TensorFlow Serving, MLflow)
- Implement async processing for batch predictions
- Monitor resource usage and auto-scaling triggers
- Use container orchestration (Kubernetes) for complex deployments