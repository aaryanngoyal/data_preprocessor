import numpy as np 
import pandas as pd 
import datetime
import utils

# convert column to datetime
def convert_to_datetime(df, column):
    df[column] = pd.to_datetime(df[column])
    return df[column]

# extract year from column
def extract_year(df, column):
    return df[column].dt.year

# create year column
def create_year_col(df, column):
    df['year'] = extract_year(df, column)
    return df['year']

# extract month from column
def extract_month(df, column):
    return df[column].dt.month

# create month column
def create_month_col(df, column):
    df['month'] = extract_month(df, column)
    return df['month']

# extract month name from column
def extract_month_name(df, column):
    return df[column].dt.month_name()

# create month name column
def create_month_name_col(df, column):
    df['month_name'] = extract_month_name(df, column)
    return df['month_name']

# extract day from column
def extract_day(df, column):
    return df[column].dt.day

# create day column
def create_day_col(df, column):
    df['day'] = extract_day(df, column)
    return df['day']

# extract day of week from column
def extract_day_of_week(df, column):
    return df[column].dt.dayofweek

# create day of week column
def create_day_of_week_col(df, column):
    df['day_of_week'] = extract_day_of_week(df, column)
    return df['day_of_week']

# extract day name from column
def extract_day_name(df, column):
    return df[column].dt.day_name()

# create day name column
def create_day_name_col(df, column):
    df['day_name'] = extract_day_name(df, column)
    return df['day_name']

# extract weekend from column
def extract_isweekend(df, column):
    return pd.DataFrame(extract_day_name(df, column).isin(['Saturday', 'Sunday']).astype(int))

# create is weekend column
def create_isweekend_col(df, column):
    df['is_weekend'] = extract_isweekend(df, column)
    return df['is_weekend']

# extract week of year from column
def extract_week_of_year(df, column):
    return df[column].dt.week
 
# create week of the year column
def create_week_of_year_col(df, column):
    df['week_of_year'] = extract_week_of_year(df, column)
    return df['week_of_year']

# extract quarter from column
def extract_quarter(df, column):
    return df[column].dt.quarter

# create quarter column
def create_quarter_col(df, column):
    df['quarter'] = extract_quarter(df, column)
    return df['quarter']

# extract semester from column
def extract_semester(df, column):
    return pd.DataFrame((extract_quarter(df, column) <= 2).astype(int).replace({0:2}))

# create semester column
def create_semester_col(df, column):
    df['semester'] = extract_semester(df, column)
    return df['semester']

# extract time and date from current day using column
def extract_time_and_date_from_today(df, column):
    today = datetime.datetime.today()
    return today - df[column]

# create time and date from current day column
def create_time_and_date_from_today_col(df, column):
    df['time_and_date_from_today'] = extract_time_and_date_from_today(df, column)
    return df['time_and_date_from_today']

# extract days in between from column
def extract_days_inbetween(df, column):
    today = datetime.datetime.today()
    return (today - df[column]).dt.days

# create days in between column
def create_days_inbetween_col(df, column):
    df['days_inbetween'] = extract_days_inbetween(df, column)
    return df['days_inbetween']

# extract months passed from column
def extract_months_passed(df, column):
    today = datetime.datetime.today()
    return np.round((today - df[column]) / np.timedelta64(1, 'M'), 0)

# create months passed column
def create_months_passed_col(df, column):
    df['months_passed'] = extract_months_passed(df, column)
    return df['months_passed']

# extract hours from column
def extract_hour(df, column):
    return df[column].dt.hour

# create hours column
def create_hour_col(df, column):
    df['hour'] = extract_hour(df, column)
    return df['hour']

# extract minutes from column
def extract_minute(df, column):
    return df[column].dt.minute

# create minutes column
def create_minutes_col(df, column):
    df['minutes'] = extract_minute(df, column)
    return df['minutes']

# extract seconds from column
def extract_second(df, column):
    return df[column].dt.second

# create seconds column
def create_second_col(df, column):
    df['second'] = extract_second(df, column)
    return df['second']

# extract time from column
def extract_time(df, column):
    return df[column].dt.time

# create time column
def create_time_col(df, column):
    df['time'] = extract_time(df, column)
    return df['time']

# extract time difference in seconds from column
def time_diff_sec(df, column):
    today = datetime.datetime.today()
    return (today - df[column]) / np.timedelta64(1, 's')

# create time difference in second column
def create_time_diff_sec_col(df, column):
    df['time_diff_sec'] = time_diff_sec(df, column)
    return df['time_diff_sec']

# extract time difference in minutes from column
def time_diff_min(df, column):
    today = datetime.datetime.today()
    return (today - df[column]) / np.timedelta64(1, 'm')

# create time difference in minute column
def create_time_diff_min_col(df, column):
    df['time_diff_min'] = time_diff_min(df, column)
    return df['time_diff_min']

# extract time difference in hour from column
def time_diff_hour(df, column):
    today = datetime.datetime.today()
    return (today - df[column]) / np.timedelta64(1, 'h')

# create time difference in hour column
def create_time_diff_hour_col(df, column):
    df['time_diff_hour'] = time_diff_hour(df, column)
    return df['time_diff_hour']