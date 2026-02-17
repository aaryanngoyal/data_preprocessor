import pandas as pd

### DATA INSPECTION HELPER ###

# shape of the df
def shape(df):
    return df.shape

# sample of the df
def struct(df):
    return df.head()

# information of the features
def df_info(df):
    return df.info()

# sum of all missing value in df
def missing_val(df):
    return df.isnull().sum()

# description of df
def description(df):
    return df.describe()

# sum of duplicate value
def duplicate_val(df):
    return df.duplicated().sum()

# correlation between features 
def correlation(df):
    return df.corr(numeric_only=True)

# get numeric columns
def get_numeric_columns(df):
    num_col = df.select_dtypes(include=['number'])
    return num_col

# get categorical columns
def get_categorical_columns(df):
    cat_col = df.select_dtypes(include=['object'])
    return cat_col

# get datetime columns
def get_datetime_columns(df, threshold=0.7):
    datetime_col =[]
    for i in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[i]):
            datetime_col.append(i)
            continue

        if df[i].dtypes == 'object':
            converted = pd.to_datetime(df[i], errors='coerce')

            if converted.notna().mean() >= threshold:
                datetime_col.append(i)
            
    return df[datetime_col]

# get boolean columns
def get_bool_columns(df):
    bool_col = df.select_dtypes(include=['bool'])
    return bool_col

# check if dataset is empty 
def check_empty(df):
    return df.empty

# find column dtypes
def column_dtypes(df):
    return df.dtypes

# find missing value %
def missing_value_per(df):
    return (df.isnull().sum() / df.shape[0]) * 100

# check is dataset is normalized
def check_normalized(df):
    pass