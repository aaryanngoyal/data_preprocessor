import numpy as np
import pandas as pd 

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer

    ### MATHEMATICAL TRANSFORMATION ###

# log transformation
def log_transform(df, column):
    funct = FunctionTransformer(func=np.log1p, validate=False)
    df[column] = funct.fit_transform(df[[column]])
    return df[column]

# reciprocal transformation
def reciprocal(df, column):
    funct = FunctionTransformer(func=lambda x : 1 / x, validate=True)
    df[column] = funct.fit_transform(df[[column]])
    return df[column]

# square root transformation
def square_root(df, column):
    funct = FunctionTransformer(func=lambda x: np.sqrt(x - np.min(x) + 1), validate=False)
    df[column] = funct.fit_transform(df[[column]])
    return df[column]

# square transformation
def square(df, column):
    funct = FunctionTransformer(func=lambda x: x ** 2, validate=False)
    df[column] = funct.fit_transform(df[[column]])
    return df[column]

    ### POWER TRANSFORMATION ###

# box cox transformation
def box_cox_transform(df, column, option):
    powert = PowerTransformer(method='box-cox', standardize=option)
    df[column] = powert.fit_transform(df[[column]])
    return df[column]

# yeo johnson tansformation
def yeo_johnson_transform(df, column, option):
    powert = PowerTransformer(method='yeo-johnson', standardize=option)
    df[column] = powert.fit_transform(df[[column]])
    return df[column]
