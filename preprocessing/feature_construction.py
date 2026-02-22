import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures

# create feature using bins
def bin_numeric(df, column, bins, strat, encoding):
    kbin = KBinsDiscretizer(n_bins=bins, strategy=strat, encode=encoding)
    result = kbin.fit_transform(df[[column]])
    if result.shape[1] > 1:
        raise ValueError("Use 'ordinal' encoding for binning — onehot produces multiple columns")
        return df
    df[column + "_binned"] = result.ravel()
    return df

# create feature using polynomial
def poly_feature(df, column, deg):
    poly = PolynomialFeatures(degree=deg)
    result = poly.fit_transform(df[[column]])
    feature_name = poly.get_feature_names_out([column])
    for i, name in enumerate(feature_name):
        df[name] = result[:, i]
    return df

# create feature using ratio
def ratio_feature(df, col1, col2):
    df[col2] = df[col2].replace(0, np.nan).ffill()
    df[col1 + "/" + col2] = df[col1] / df[col2]
    return df

# create feature using difference
def difference_feature(df, col1, col2):
    df[col1 + "-" + col2] = df[col1] - df[col2]
    return df

# create feature using addition
def add_feature(df, col1, col2):
    df[col1 + "+" + col2] = df[col1] + df[col2]
    return df

# create feature using multiplication
def multiply_feature(df, col1, col2):
    df[col1 + "*" + col2] = df[col1] * df[col2]
    return df