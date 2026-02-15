import numpy as np 
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from preprocessing.understand_data import shape


def mean_imputation(df):
    
    si = SimpleImputer(strategy='mean')
    df_shape = shape(df)
    if df_shape[1] == 1:
        return si.fit_transform(df.iloc[:, 0])   

def median_imputation(df):
    
    si = SimpleImputer(strategy='median')
    df_shape = shape(df)
    if df_shape[1] == 1:
        return si.fit_transform(df.iloc[:, 0])

def random_imputation(df):
    
    return df.iloc[:,0].fillna(-999)

def end_of_dist(df):
    
    df_mean = df.mean()
    df.std = df.std()

    pass

def mode_imputation(df):

    si = SimpleImputer(strategy='mode')
    df_shape = shape(df)
    if df_shape[1] == 1:
        return si.fit_transform(df.iloc[:, 0])

def missing_feature(df):

    df.iloc[:, 0] = df.iloc[:, 0].fillna("Missing") 

def knn_imputer(df):
    
    knn = KNNImputer(n_neighbors=3)
    return knn.fit_transform(df)