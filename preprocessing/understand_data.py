import pandas as pd

# shape of the df
def shape(df):
    return df.shape

# sample of the df
def struct(df):
    return df.sample()

# datatype of the features
def data_type(df):
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
    return df.corr