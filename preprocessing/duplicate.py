import numpy as np 
import pandas as pd
import utils

# sum of duplicate value
def duplicate_val(df):
    return utils.duplicate_val(df)

# remove duplicates
def remove_duplicate(df):
    return df.drop_duplicates()