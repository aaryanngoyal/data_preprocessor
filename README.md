# Data Preprocessor

A web-based data preprocessing tool built with Streamlit that allows you to clean, transform, and prepare datasets for machine learning without writing any code.

## Features

### Handle Missing Values
- Numerical columns: Mean, Median, Random, KNN Imputation
- Categorical columns: Mode, Add Missing Label
- Supports imputation on individual columns or all columns at once

### Handle Outliers
- Detection methods: Z-Score (for normal distributions), IQR, Percentile (for skewed distributions)
- Removal techniques: Trimming, Capping
- Automatically suggests the appropriate method based on column skewness

### Feature Construction
- Convert numeric columns to bins (uniform, quantile, kmeans strategies)
- Polynomial features
- Ratio, Difference, Addition, Multiplication of two features

### Encoding
- One-Hot Encoding (auto-checks cardinality)
- Label Encoding
- Ordinal Encoding (user-defined category order)

### Scaling
- Standardization (Z-Score scaling)
- Min-Max Normalization
- Robust Scaling

### Feature Selection
- Correlation-based selection
- Variance threshold
- Other selection techniques

## Project Structure

```
data_preprocessor/
├── app.py                  # Main Streamlit application
├── utils.py                # Utility functions (stats, column types, etc.)
├── requirements.txt
└── preprocessing/
    ├── __init__.py
    ├── missing.py          # Missing value imputation
    ├── outlier.py          # Outlier detection and removal
    ├── feature_construction.py
    ├── encoding.py
    ├── scaling.py
    ├── feature_selection.py
    ├── transformation.py
    ├── datetime.py
    └── duplicate.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaryanngoyal/data_preprocessor.git
cd data_preprocessor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Upload a CSV file using the file uploader
2. View dataset statistics, missing values, and correlations in the dashboard
3. Select an operation from the sidebar
4. Apply transformations step by step
5. Download the processed dataset as a CSV when done

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn

See `requirements.txt` for full list.

## Live Demo

[https://data-preprocesorr.streamlit.app/]
