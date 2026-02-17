import numpy as np 
import pandas as pd

def finding_outlier_z_score(df, column):
    df_mean = df[column].mean()
    df_std = df[column].std()

    highest_allowed = df_mean + 3 * df_std
    lowest_allowed = df_mean - 3 * df_std

    return df[df[column] > highest_allowed | df[column] < lowest_allowed]

def outlier_removal_zscore_trimming(df, column):
    df_mean = df[column].mean()
    df_std = df[column].std()

    highest_allowed = df_mean + 3 * df_std
    lowest_allowed = df_mean - 3 * df_std

    new_df = df[(df[column] < highest_allowed) & (df[column] > lowest_allowed)]
    return new_df

def outlier_removal_zscore_capping(df, column):
    df_mean = df[column].mean()
    df_std = df[column].std()

    df[column + "_zscore"] = (df[column] - df_mean) / df_std
    return df[(df[column + "_zscore"] < 3) & (df[column + "_zscore"] > -3)] 

def finding_outlier_iqr(df, column):
    per25 = df[column].quantile(0.25)
    per75 = df[column].quantile(0.75)

    iqr = per75 - per25

    up_lim = per75 + 1.5 * iqr
    low_lim = per25 - 1.5 * iqr

    return df[df[column] > up_lim | df[column] < low_lim]

def outlier_removal_iqr_trimming(df, column):
    per25 = df[column].quantile(0.25)
    per75 = df[column].quantile(0.75)

    iqr = per75 - per25

    up_lim = per75 + 1.5 * iqr
    low_lim = per25 - 1.5 * iqr

    return df[df[column] < up_lim & df[column] > low_lim]

def outlier_removal_iqr_capping(df, column):
    per25 = df[column].quantile(0.25)
    per75 = df[column].quantile(0.75)

    iqr = per75 - per25

    up_lim = per75 + 1.5 * iqr
    low_lim = per25 - 1.5 * iqr

    df_copy = df.copy()

    df_cap[column] = np.where(
        df_cap[column] > up_lim, 
        up_lim, 
        np.where(
            df_cap[column] < low_lim,
            low_lim,
            df_cap[column]
        )
    )
    return df_cap

def finding_outlier_percentile(df, column):
    up_lim = df[column].quantile(0.99)
    low_lim = df[column].quantile(0.01)

    return df[df[column] > up_lim | df[column] < low_lim]

def outlier_removal_percentile_trimming(df, column):
    up_lim = df[column].quantile(0.99)
    low_lim = df[column].quantile(0.01)

    return df[(df[column] < up_lim) & df[column] < low_lim]

def outlier_removal_percentile_capping(df, column):
    up_lim = df[column].quantile(0.99)
    low_lim = df[column].quantile(0.01)

    df[column] = np.where(
        df[column] > up_lim,
        up_lim, 
        np.where(
            df[column] < low_lim,
            low_lim,
            df[column]
        )
    )
    return df