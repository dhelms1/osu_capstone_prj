# Import libraries for accessing & formatting daymet data
import requests
from datetime import datetime as dt
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import seaborn as sns
from scipy.signal import savgol_filter


# Import libraries for sql connection
import mysql.connector
import pickle
import warnings

warnings.filterwarnings('ignore')


def daymet_daily_est(station, s_d, e_d, threshold = 0):
    '''
    Given a station, the start/end date, and year, this function will return the daily
    min/max estimates for a given station. A threshold of 15 degrees is automatically applied
    to both bounds, as well as the filter being fitted through the estimates the match what is
    produced by our estimates.
    '''
    # Get daymet data from api access function
    dm_data = get_daymet_min_max(station, s_d, e_d, threshold)

    # Group to get daily min and max estimates for each month_day pairing (366 values)
    dm_data = dm_data.groupby('month_day').agg({'daily_min': 'min', 'daily_max': 'max'})  
    
    dm_data.loc['02-29'] = dm_data[dm_data.index == '02-28'].values.flatten()

    # Apply filter to format the same as daily estimates
    min_est = dm_data.daily_min.values
    max_est = dm_data.daily_max.values
    smoothed_min = savgol_filter(min_est, window_length=15, polyorder=2)
    smoothed_max = savgol_filter(max_est, window_length=15, polyorder=2)
    dm_data['daily_min'] = smoothed_min
    dm_data['daily_max'] = smoothed_max
    
    return dm_data # return final estimates

def daymet_vs_das_graph(t_est, d_est, val_data, station_num):
    '''
    Create graph for temperature readings across the validation data, with bounds
    for the upper and lower temperature. Compare Daymet and DAS estimates with 
    bounding lines on validation data.
    '''
    
    v_idx = [x.split('-', 1)[1] for x in val_data.index.values] # get all month_day occurances
    t_est_val = t_est.loc[v_idx] # subset estimates with only val_data dates
    d_est_val = d_est.loc[v_idx] # subset estimates with only val_data dates

    
    # Each day has 96 readings (repeat bounds 96 times since plot has all data)
    t_min_est = np.repeat(t_est_val.daily_min.values, 96)
    t_max_est = np.repeat(t_est_val.daily_max.values, 96)
    
    d_min_est = np.repeat(d_est_val.daily_min.values, 96)
    d_max_est = np.repeat(d_est_val.daily_max.values, 96)
    
    # Find month index (used for plotting)
    tmp = [int(x.split('-')[1]) for x in val_data.index.values] # get month from index
    m_idx = [x*len(val_data.columns.tolist()) for x in range(1,len(tmp)) if tmp[x] != tmp[x-1]] # month change idx
    m_idx.insert(0, 0) # insert start date
    m_idx.append(len(val_data.values.flatten())) # insert end date
    
    vals = val_data.values.flatten() # get all temperature readings from validation data

    # Plot data
    plt.figure(figsize=(10,5))
    plt.plot(vals, alpha=0.7)
    
    # Plot das estimates
    plt.plot(t_max_est, color='green', label='DAS max')
    plt.plot(t_min_est, color='red', label='DAS min')
    
    # Plot daymet estimates
    plt.plot(d_max_est, color='orange', label='Daymet max')
    plt.plot(d_min_est, color='blue', label='Daymet min')
    
    year = val_data.index[0].split('-')[0]
    plt.title(f'{year} DAS vs Daymet Temperature Bounds: Station {station_num}')
    plt.ylabel('Temperature')
    plt.xlabel('Month')
    plt.ylim(-15, 150)
    plt.xticks(m_idx[:-1], range(1,len(m_idx)))
    plt.legend()
    plt.show();
    
    return 

def get_daymet_min_max(station, s_d, e_d, threshold = 0):
    '''
    Function to access the daymet api and acquire min/max estimates for a given timespan.
    A threshold can be applied to the min/max (used for estimates, not for synthetic data).
    '''
    # Load credentials for login (hidden and not added to repo)
    with open("login_cred.pkl", "rb") as fp:
        config = pickle.load(fp)
        
    cnx = mysql.connector.connect(**config) # connection point
    
    # Query for accessing station information
    info_q = (f"""SELECT station_id, station_latdeg, station_lngdeg
             FROM stations_awn
             WHERE station_id = {station} """)

    tmp = pd.read_sql(info_q, cnx)

    # Get latitude and longitude
    lat = tmp.station_latdeg.values[0]
    long = tmp.station_lngdeg.values[0]
    url = "https://daymet.ornl.gov/single-pixel/api/data"

    # Daymet API request (json format)
    r = requests.get(f"{url}?lat={lat}&lon={long}&vars=tmax,tmin&start={s_d}&end={e_d}&format=json")

    # Convert to dataframe if successful, otherwise exit with error code
    if r.status_code == 200:
        dm_data = pd.DataFrame(r.json()['data'])
    else:
        print("Error occurred:", r.status_code) 

    # Convert year + year day into YYYY-MM-DD format
    x = lambda row: dt.strptime(str(int(row['year'])) + "-" + str(int(row['yday'])), "%Y-%j").strftime("%Y-%m-%d")

    dm_data['date'] = dm_data.apply(x, axis=1)

    # Split date into year and month_day pairings
    dm_data[['year', 'month_day']] = [x.split('-',1) for x in dm_data.date.values]

    # Drop unused columns
    dm_data = dm_data.drop(columns=['yday', 'date', 'year'])

    # Rename temperature columns
    dm_data = dm_data.rename(columns={'tmax (deg c)': 'daily_max', 'tmin (deg c)': 'daily_min'})

    # Convert celsius to farenheit
    c_to_f = lambda x: ((9/5) * x) + 32

    # Apply threshold of 15 degrees to both min and max
    dm_data['daily_max'] = dm_data['daily_max'].apply(c_to_f) + threshold
    dm_data['daily_min'] = dm_data['daily_min'].apply(c_to_f) - threshold
    
    return dm_data

def error_injection(daymet, n_flags, temp_inj = 40, delta_inj = 10):
    '''
    Randomly inject errors into the current Daymet temperature synthetic data. Specify the number of 
    flags to be set, and the temperature injection value (degrees). Return a new dataframe of altered 
    values, without modifying the original.
    '''
    tmp = daymet.copy() # create copy
    
    np.random.seed(24) # set random seed

    t_vals = tmp.values.flatten() # get all current values

    t_idx = np.random.choice(np.arange(0, t_vals.size, 8), n_flags, replace=False) # get random index

    p_size = (len(t_idx))//4

    p_arr = [sorted(t_idx[i*p_size: (i+1)*p_size]) for i in range(4)]

    idx = {
        'temp_lb': p_arr[0],
        'temp_ub': p_arr[1],
        'delta_lb': p_arr[2],
        'delta_ub': p_arr[3],
    }

    for i in idx['temp_lb']: # temp drop
        t_vals[i] = t_vals[i] - temp_inj

    for i in idx['temp_ub']: # temp spike
        t_vals[i] = t_vals[i] + temp_inj

    for i in idx['delta_lb']: # step drop
        t_vals[i] = t_vals[i] - delta_inj

    for i in idx['delta_ub']: # step spike
        t_vals[i] = t_vals[i] + delta_inj

    tmp[:] = t_vals.reshape((365, 96)) # create matrix of error injected valued

    return tmp, idx

def eval_metrics(val_data, temp_b, delta_b, idx_dict):
    '''
    Calculate the accuracy, on the synthetic Daymet data with injected errors, for the temperature
    and delta (step) parameters. Accuracy is only based on injected errors, calculated by checking
    the index for the appropriate parameter being set. This is based off the assumption that there
    are 0 errors in the validation data.
    
    Returns the accuracy for the temperature and delta lower / upper bounds.
    '''
   # -------------------- Temperature bounds flag calculation --------------------

    v_idx = [x.split('-', 1)[1] for x in val_data.index.values] # get all month_day occurances
    est_val = temp_b.loc[v_idx] # subset estimates with only val_data dates

    vals = val_data.values.flatten() # get values of temperature data

    t_min_est = np.repeat(est_val.daily_min.values, 96) # repeat for each daily value (96 per day)
    t_max_est = np.repeat(est_val.daily_max.values, 96)

    # Initialize counters for flags (temp & delta)
    temp_lb_flags = 0
    temp_ub_flags = 0
    delta_lb_flags = 0
    delta_ub_flags = 0

    for i in idx_dict['temp_lb']: # temp drop detected
        if vals[i] < t_min_est[i]:
            temp_lb_flags += 1

    for i in idx_dict['temp_ub']: # temp spike detected
        if vals[i] > t_max_est[i]:
            temp_ub_flags += 1
        
     # -------------------- Delta (step) flag calculation --------------------   
    s_tmp = val_data.copy() # create copy (don't alter original)
    s_tmp['month'] = [int(x.split('-')[1]) for x in s_tmp.index.values] # create month column

    # Find lengths of each month
    month_lengths = s_tmp.month.value_counts(sort=False).values

    # Repeat estimates (creates 2d array)
    delta_lb_2d = [[x]*s for x,s in zip(delta_b['iq_lb'].values, month_lengths)]
    delta_ub_2d = [[x]*s for x,s in zip(delta_b['iq_ub'].values, month_lengths)]

    # Flatten into 1d array (length of year)
    daily_lb_1d = list(itertools.chain.from_iterable(delta_lb_2d))
    daily_ub_1d = list(itertools.chain.from_iterable(delta_ub_2d))

    # Repeat each value for an entire day (96 per day)
    d_lb_est = np.repeat(daily_lb_1d, 96)
    d_ub_est = np.repeat(daily_ub_1d, 96)

    # Calculate difference between 15 minute intervals
    vals = np.diff(val_data.values.flatten())
    vals = np.insert(vals, 0, 0) # insert for first observation (no previous)

    for i in idx_dict['delta_lb']: # step drop detected
        if vals[i] < d_lb_est[i]:
            delta_lb_flags += 1

    for i in idx_dict['delta_ub']: # step spike detected
        if vals[i] > d_ub_est[i]:
            delta_ub_flags += 1

    # Calculate accuracy for upper and lower bound temperature + delta
    d_lb_acc, d_ub_acc, t_lb_acc, t_ub_acc = 1.0, 1.0, 1.0, 1.0

    if len(idx_dict['temp_lb']) > 0:
        t_lb_acc = round(temp_lb_flags/len(idx_dict['temp_lb']), 2)
        t_ub_acc = round(temp_ub_flags/len(idx_dict['temp_ub']), 2)
        d_lb_acc = round(delta_lb_flags/len(idx_dict['delta_lb']), 2)
        d_ub_acc = round(delta_ub_flags/len(idx_dict['delta_ub']), 2)
    
    return d_lb_acc, d_ub_acc, t_lb_acc, t_ub_acc