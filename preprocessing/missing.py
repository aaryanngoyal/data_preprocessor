import numpy as np 
import pandas as pd
import utils

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.compose import ColumnTransformer

# imputing using mean all feature
def mean_imputation_all_feature(df):
    
    num_col = utils.get_numeric_columns(df).columns.tolist()
    si = SimpleImputer(strategy='mean')
    df[num_col] = si.fit_transform(df[num_col])
    return df 

# imputing using mean for single feature
def mean_imputation_single_feature(df, column):

    si = SimpleImputer(strategy='mean')
    df[[column]] = si.fit_transform(df[[column]])
    return df

# imputing using median all feature
def median_imputation_all_feature(df):
    
    num_col = utils.get_numeric_columns(df).columns.tolist()
    si = SimpleImputer(strategy='median')
    df[num_col] = si.fit_transform(df[num_col])
    return df

# imputing using median single feature
def median_imputation_single_feature(df, column):

    si = SimpleImputer(strategy='median')
    df[[column]] = si.fit_transform(df[[column]])
    return df

# imputing using random all feature
def random_imputation_all_feature(df):
    
    num_col = utils.get_numeric_columns(df).columns.tolist()
    for col in num_col:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            rnd_sample = df[col].dropna().sample(null_count, replace=True).values
            df.loc[df[col].isnull(), col] = rnd_sample
    return df

# imputing using random single feature
def random_imputation_single_feature(df, column):
    
    null_count = df[column].isnull().sum()
    if null_count > 0:
        rnd_sample = df[column].dropna().sample(null_count, replace=True).values
        df.loc[df[column].isnull(), column] = rnd_sample
    df[column] = df[column].fillna(-999)
    return df

def end_of_dist(df):
    
    df_mean = df.mean()
    df.std = df.std()

    pass

# imputing using mode all feature
def mode_imputation_all_features(df):

    cat_col = utils.get_categorical_columns(df).columns.tolist()
    si = SimpleImputer(strategy='most_frequent')
    df[cat_col] = si.fit_transform(df[cat_col])
    return df

# imputing using mode single feature
def mode_imputation_single_features(df, column):
    si = SimpleImputer(strategy='most_frequent')
    df[[column]] = si.fit_transform(df[[column]])
    return df

# add missing value to all feature
def add_missing_value_all_feature(df):

    cat_col = utils.get_categorical_columns(df).columns.tolist()
    for col in cat_col:
        df[col] = df[col].fillna("Missing") 
    return df

# add missing value to single feature
def add_missing_value_single_feature(df, column):

    df[column] = df[column].fillna("Missing") 
    return df

# KNN imputer
def knn_imputer(df, n, weight):
    
    knn = KNNImputer(n_neighbors=n, weights=weight)
    imputed = knn.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)