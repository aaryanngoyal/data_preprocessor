import numpy as np 
import pandas as pd 
import utils

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

# one hot encoding
def one_hot_encode(df, column):
    col = utils.cardinality_single_cat(df, column)
    if col <= 50:
        df = pd.get_dummies(df, columns=[column], drop_first=True, dtype='int32')
        return df
    else:
        raise ValueError("Avoid Using One Hot Encoding")

# label encoding
def label_encoding(df, column):
    le = LabelEncoder()
    le.fit(df[column])
    df[column] = le.transform(df[column])
    return df

# ordinal encoding
def ordinal_encoding(df, column, general):
    oe = OrdinalEncoder(categories=[general])
    df[column] = oe.fit_transform(df[[column]]).ravel().astype(int)
    return df
    