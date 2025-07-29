"""
Create sample dataset for testing the house price prediction pipeline
"""
import pandas as pd
import numpy as np

def create_sample_data():
    """Create a sample dataset that matches the expected structure"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate cities and states with realistic distributions
    cities = ['Los Angeles', 'New York', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
              'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    states = ['CA', 'NY', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']
    
    # Create city-state combinations
    city_state_pairs = list(zip(cities, states))
    
    data = {
        'brokered_by': np.random.randint(1000, 9999, n_samples),
        'status': np.random.choice(['for_sale', 'sold', 'ready_to_build'], n_samples),
        'bed': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.25, 0.35, 0.15, 0.05]),
        'bath': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.35, 0.2, 0.05]),
        'acre_lot': np.random.lognormal(np.log(0.25), 0.5, n_samples),
        'street': np.random.randint(100, 99999, n_samples),
        'city': [city_state_pairs[np.random.randint(0, len(city_state_pairs))][0] for _ in range(n_samples)],
        'state': [city_state_pairs[i % len(city_state_pairs)][1] for i in range(n_samples)],
        'zip_code': np.random.randint(10000, 99999, n_samples),
        'house_size': np.random.lognormal(np.log(2000), 0.4, n_samples),
        'price': np.random.lognormal(np.log(300000), 0.6, n_samples)
    }
    
    # Ensure house_size is reasonable
    data['house_size'] = np.clip(data['house_size'], 500, 10000)
    data['acre_lot'] = np.clip(data['acre_lot'], 0.05, 5.0)
    data['price'] = np.clip(data['price'], 50000, 5000000)
    
    # Add some correlation between features and price
    df = pd.DataFrame(data)
    
    # Adjust price based on bed/bath count and house size
    price_multiplier = (
        1 + (df['bed'] - df['bed'].mean()) * 0.1 +
        (df['bath'] - df['bath'].mean()) * 0.15 +
        (df['house_size'] - df['house_size'].mean()) / df['house_size'].mean() * 0.3 +
        (df['acre_lot'] - df['acre_lot'].mean()) / df['acre_lot'].mean() * 0.1
    )
    
    df['price'] = df['price'] * price_multiplier
    df['price'] = np.clip(df['price'], 50000, 5000000)
    
    return df

if __name__ == "__main__":
    # Create sample data
    df = create_sample_data()
    
    # Save to CSV
    df.to_csv('data/raw/df_imputed.csv', index=False)
    print(f"Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", list(df.columns))
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())