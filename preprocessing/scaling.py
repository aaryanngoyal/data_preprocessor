import numpy as np 
import pandas as pd
import utils

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

    ### STANDARDIZATION ###

def standardization(df, column, mean, std):
    ss = StandardScaler(with_mean=mean, with_std=std)
    df[column] = ss.fit_transform(df[[column]])
    return df[column]

    ### NORMALIZATION ###

# min max scaling
def min_max_scale(df, column):
    min_max = MinMaxScaler()
    df[column] = min_max.fit_transform(df[[column]])
    return df[column]

# max abs scaling
def max_abs_scale(df, column):
    max_abs = MaxAbsScaler()
    df[column] = max_abs.fit_transform(df[[column]])
    return df[column]

# robust scaling
def robust_scale(df, column):
    robust = RobustScaler()
    df[column] = robust.fit_transform(df[[column]])
    return df[column]