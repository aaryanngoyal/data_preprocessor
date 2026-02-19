import numpy as np 
import pandas as pd 
import utils

# remove feature with constant value
def remove_constant_constant_column(df):
    df = df.loc[:, df.nunique() > 1]
    return df

# remove feature with high correlation
def remove_high_correlation(df, threshold):
    corr_matrix = utils.correlation(df)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(to_drop, axis = 1, inplace=True)
    return df
