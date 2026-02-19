import numpy as np 
import pandas as pd
import utils

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.compose import ColumnTransformer

# imputing using mean all feature
def mean_imputation_all_feature(df):
    
    num_col = utils.get_numeric_columns(df)
    si = SimpleImputer(strategy='mean')
    for i in num_col:
        df[[i]] = si.fit_transform(df[[i]])

    return df 

# imputing using mean for single feature
def mean_imputation_single_feature(column):

    return ColumnTransformer(transformers=[
        ('trf1', SimpleImputer(strategy='mean'), [column])
    ], remainder='passthrough')

# imputing using median all feature
def median_imputation_all_feature(df):
    
    num_col = utils.get_numeric_columns(df)
    si = SimpleImputer(strategy='median')
    for i in num_col:
        df[[i]] = si.fit_transform(df[[i]])

    return df

# imputing using median single feature
def median_imputation_single_feature(column):

    return ColumnTransformer(transformers=[
        ('trf1', SimpleImputer(strategy='median'), [column])
    ], remainder='passthrough')

# imputing using random all feature
def random_imputation_all_feature(df):
    
    num_col = utils.get_numeric_columns(df)
    for i in num_col:
        df[i] = df[i].fillna(-999)

    return df

# imputing using random single feature
def random_imputation_single_feature(df, column):
    
    df[column] = df[column].fillna(-999)
    return df

def end_of_dist(df):
    
    df_mean = df.mean()
    df.std = df.std()

    pass

# imputing using mode all feature
def mode_imputation_all_features(df):

    cat_col = utils.get_categorical_columns(df)
    si = SimpleImputer(strategy='mode')
    for i in cat_col:
        df[[i]] = si.fit_transform(df[[i]])

    return df

# imputing using mode single feature
def mode_imputation_single_features(df, column):

    df[[column]] = si.fit_transform(df[[column]])
    return df

# add missing value to feature
def add_missing_value(df):

    cat_col = utils.get_categorical_columns(df)
    for i in cat_col:
        df[i] = df[i].fillna("Missing") 

    return df

# KNN imputer
def knn_imputer(df, n, weight):
    
    knn = KNNImputer(n_neighbors=n, weights=weight)
    return knn.fit_transform(df)