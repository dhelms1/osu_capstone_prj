import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Import libraries for sql connection
import mysql.connector
import pickle
import warnings

warnings.filterwarnings('ignore')

def create_pivot(s_id):
    '''
    Function takes a station number to load the data from. Query will be ran within the function
    for the given station number.
        
    Returns a tuple of (pivot table, year splits).
    
    Pivot table - each row is a single day, column is tstamp (15 minutes), and values are temperature readings.
    Year splits - index at which each year changes (need for splitting estimation/validation)
    '''
    # Load credentials for login (hidden and not added to repo)
    with open("login_cred.pkl", "rb") as fp:
        config = pickle.load(fp)

    cnx = mysql.connector.connect(**config) # connection point
    
    # Query DB to get timestamp and temp data for specific station data
    s_q = (f"SELECT station_id, tstamp, temp FROM raw_awn_records r WHERE r.station_id = {s_id}")

    data = pd.read_sql(s_q, cnx) # load query into dataframe
    
    data[['date', 'time']] = [str(x).split('T') for x in data.tstamp.values] # Convert tstamp to date and time
    data['time'] = [x.split('.')[:1][0] for x in data.time.values] # Clean up time column
    data = data.drop(['station_id', 'tstamp'], axis=1) # Drop unused columns

    # Pivot with index being date (single day), columns being 24 hour period, values being temp reading
    data_pivot = data.pivot(index='date', columns='time', values='temp')
    data_pivot.reset_index(inplace=True) # Reset index

    # Set index to be date and remove column name
    data_pivot = data_pivot.set_index('date')
    data_pivot = data_pivot.rename_axis(index=None)
    
    # Find index where year splits
    idx_vals = data_pivot.index.values
    year_splits = [x for x in range (1, len(idx_vals)) if idx_vals[x-1][:4] != idx_vals[x][:4]]
    
    return data_pivot, year_splits


def get_overall_bounds(data, q_val, threshold):
    '''
    Function takes estimation data, a quantile value (0.0 - 1.0), and a threshold (degrees). 
    Returns a estimation for each day (includes leap year).
    '''
    
    # Create dataframe
    df = pd.DataFrame({
        "month_day": [x.split('-', 1)[1] for x in data.index.values], # get month and day, "MM-DD"
        "year": [x.split('-', 1)[0] for x in data.index.values], # get year, "YYYY"
        "daily_min": data.quantile(q = 1-q_val, axis=1).values - threshold, 
        "daily_max": data.quantile(q = q_val,axis=1).values + threshold,
    })

    # Group by month_day, find min/max for each day
    est = df.groupby('month_day').agg({'daily_min': 'min', 'daily_max': 'max'})
    
    # Handle leap year by setting equal to previous day (overwrites if exists, create new if not)
    est.loc['02-29'] = est[est.index == '02-28'].values.flatten()

    return est


def get_delta_est(data, q_val, threshold):
    '''
    Given estimation data, a quantile value (between [0, 1.0]), and a threshold to be included in the
    upper/lower bound estimates, create the bounds for the delta values (across 15 minutes).
    '''
    
    s_iqr = data.copy()
    
    vals = np.diff(data.values.flatten()) # find delta (per 15 minutes)
    vals = np.insert(vals, 0, 0) # insert for first record (no previous)
    
    s_iqr[:] = vals.reshape(data.shape)
    
    s_iqr["month"] = [x.split('-')[1] for x in s_iqr.index.values]
    
    iqr_est = pd.DataFrame(columns=['Month', 'iq_lb', 'iq_ub'])

    for m in s_iqr.month.unique():
        d = s_iqr[s_iqr.month == m].drop(columns=['month']).values.flatten()
        qu = np.nanquantile(d, q_val) # upper quantile (0.75 = q3)
        ql = np.nanquantile(d, 1-q_val) # lower quantile (0.25 = q1)
        iqr = qu - ql # find iqr between lower and upper quartile
        iq_max = qu + abs(1.5*iqr) + threshold # find max (remove outliers) + threshold
        iq_min = ql - abs(1.5*iqr) - threshold # find min (remove outliers) - threshold
        iqr_est.loc[int(m)-1] = [m, iq_min, iq_max]

    iqr_est = iqr_est.set_index('Month')
    return iqr_est
    
def delta_flag_table(data, d_est):
    '''
    Create table that describes the lower/upper bounds for each month's delta values, along
    with the number of lower/upper bound flags that are raised with the current validation data.
    '''
    
    s_tmp_v = data.copy() # create copy (don't alter original)

    s_tmp_v['month'] = [int(x.split('-')[1]) for x in s_tmp_v.index.values] # create month column

    d = pd.DataFrame(columns=['Month', 'LB_EST', 'UB_EST', 'LB_FLAGS', 'UB_FLAGS'])

    for i in s_tmp_v.month.unique():
        v = s_tmp_v[s_tmp_v.month == i] # subset single month
        v = v.drop(columns=['month']).diff(axis=1).values.flatten() # drop column & get deltas
        b = d_est.iloc[i-1,:].values
        lb_f = sum(v < b[0])
        ub_f = sum(v > b[1])
        d.loc[i-1] = [i, b[0], b[1], lb_f, ub_f]

    return d

def temp_bounds_modified(data, q_val, threshold, m_val, n_filter=False):
    '''
    Given data (estimation values), a qunatile value (0.0 - 1.0), a monthly qunatile value 
    (m_val, 0.0 - 1.0) and a threshold to apply to the min/max estimates, find the daily 
    temperature bounds.
    
    **This modified version takes the IQR method from estimates monthly deltas in order to find the 
    monthly temperature estimates. We then iterate over each daily temperature estimate and do the following:
    - Daily min: take the max between daily_lb and IQR method for the given month (remove outliers).
    - Daily max: take the min between daily_ub and IQR method for the given month (remove outliers).
    - Filter: apply a Savitzky-Golay filter to smooth daily estimates (reduce noise).
    '''
    daily_df = pd.DataFrame({
        "month_day": [x.split('-', 1)[1] for x in data.index.values], # get month and day, "MM-DD"
        "year": [x.split('-', 1)[0] for x in data.index.values], # get year, "YYYY"
        "daily_min": data.quantile(q = 1-q_val, axis=1).values - threshold, 
        "daily_max": data.quantile(q = q_val,axis=1).values + threshold,
    })

    # Group by month_day, find min/max for each day
    est = daily_df.groupby('month_day').agg({'daily_min': 'min', 'daily_max': 'max'})

    # Handle leap year by setting equal to previous day (overwrites if exists, create new if not)
    est.loc['02-29'] = est[est.index == '02-28'].values.flatten()

    # Add column for month
    est['month'] = [x.split('-')[0] for x in est.index.values]

    # Copy dataframe to modifiy for monthly estimates
    monthly_df = data.copy()

    # Create month column
    monthly_df['month'] = [x.split('-')[1] for x in data.index.values]

    # Create empty dataframe for monthly estimates
    m_est = pd.DataFrame(columns=['Month', 'iq_lb', 'iq_ub'])

    for m in monthly_df.month.unique(): # iterate over each month
        d = monthly_df[monthly_df.month == m].drop(columns=['month']).values.flatten()
        qu = np.nanquantile(d, m_val) # upper quantile (0.75 = q3)
        ql = np.nanquantile(d, 1-m_val) # lower quantile (0.25 = q1)
        iqr = qu - ql # find iqr between lower and upper quartile
        iq_max = qu + abs(1.5*iqr) + threshold # find max (remove outliers) + threshold
        iq_min = ql - abs(1.5*iqr) - threshold # find min (remove outliers) - threshold
        m_est.loc[int(m)-1] = [m, iq_min, iq_max]

    modified_lb = []
    modified_ub = []

    for m in m_est.Month.unique(): # iterate over each month
        iqr_min = m_est[m_est.Month == m]['iq_lb'].values[0] # get min iqr method value
        iqr_max = m_est[m_est.Month == m]['iq_ub'].values[0] # get max iqr method values

        # Append all replaced daily min/max estimates to corresponding list
        modified_lb += list(est[est.month == m]['daily_min'].apply(lambda x: max(x, iqr_min)).values)
        modified_ub += list(est[est.month == m]['daily_max'].apply(lambda x: min(x, iqr_max)).values)

    # Final dataframe of estimates (replace with modified estimates)
    est = est.drop(columns=['month'])
    est['daily_min'] = modified_lb
    est['daily_max'] = modified_ub
    
    if n_filter: # if filter flag is set, apply Savitzky-Golay filter
        min_est = est.daily_min.values
        max_est = est.daily_max.values
        smoothed_min = savgol_filter(min_est, window_length=15, polyorder=2)
        smoothed_max = savgol_filter(max_est, window_length=15, polyorder=2)
        est['daily_min'] = smoothed_min
        est['daily_max'] = smoothed_max
    
    return est