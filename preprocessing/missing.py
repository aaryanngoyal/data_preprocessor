import numpy as np 
import pandas as pd
import utils

from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from preprocessing.understand_data import shape


def mean_imputation(df):
    
    num_col = utils.get_numeric_columns(df)
    si = SimpleImputer(strategy='mean')
    if num_col in df.columns:
        si.fit_transform(df[num_col])   

def median_imputation(df):
    
    num_col = utils.get_numeric_columns(df)
    si = SimpleImputer(strategy='median')
    if num_col in df.columns:
        si.fit_transform(df[num_col])

def random_imputation(df):
    
    num_col = utils.get_numeric_columns(df)
    return num_col.fillna(-999)

def end_of_dist(df):
    
    df_mean = df.mean()
    df.std = df.std()

    pass

def mode_imputation(df):

    cat_col = utils.get_categorical_columns
    si = SimpleImputer(strategy='mode')
    if cat_col in df.columns:
        si.fit_transform(df[cat_col])

def missing_feature(df):

    df.iloc[:, 0] = cat_col.fillna("Missing") 

def knn_imputer(df):
    
    knn = KNNImputer(n_neighbors=3)
    return knn.fit_transform(df)