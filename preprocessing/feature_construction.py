import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures

# create feature using bins
def bin_numeric(df, column, bins, strat, encoding, ):
    kbin = KBinsDiscretizer(n_bins=bins, strategy=strat, encode=encoding)
    df[column + "_binned"] = kbin.fit_transform(df[column])
    return df

# create feature using polynomial
def poly_feature(df, column, deg):
    poly = PolynomialFeatures(degree=deg)
    df[column + "_polynomial"] = ploy.fit_transform(df[column])
    return df

# create feature using ratio
def ratio_feature(df, col1, col2, name):
    df[col2] = df[col2].replace(0, NaN).ffill()
    df[name] = df[col1] / df[col2]
    return df

# create feature using difference
def difference_feature(df, col1, col2, name):
    df[name] = df[col1] - df[col2]
    return df

# create feature using addition
def add_feature(df, col1, col2, name):
    df[name] = df[col1] + df[col2]
    return df

# create feature using multiplication
def multiply_feature(df, col1, col2, name):
    df[name] = df[col1] * df[col2]
    return df