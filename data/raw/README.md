# Raw Data Directory

This directory contains the original, unprocessed datasets for the house price prediction project.

## Expected Files

### Primary Dataset
- **df_imputed.csv**: The main USA Real Estate dataset with imputed missing values
  - Size: ~1,048,575 records
  - Features: 11 columns (price, bed, bath, acre_lot, house_size, city, state, status, brokered_by, street, zip_code)
  - Target: price (house prices in USD)

## Data Source

The dataset should be obtained from a reliable real estate data source and placed in this directory before running the prediction pipeline.

## Usage

Place your dataset file named `df_imputed.csv` in this directory, then run:

```bash
# From the project root
cd src
python house_price_predictor.py
```

## Data Format

The CSV file should have the following columns:
- `price`: House price in USD (target variable)
- `bed`: Number of bedrooms
- `bath`: Number of bathrooms  
- `acre_lot`: Lot size in acres
- `house_size`: House size in square feet
- `city`: City name
- `state`: State abbreviation
- `status`: Property status (for sale, sold, etc.)
- `brokered_by`: Broker information
- `street`: Street address
- `zip_code`: ZIP code

## File Size Considerations

- The full dataset may be large (>100MB)
- For testing, you can use a sample of the data
- The pipeline includes automatic sampling options for memory efficiency