from daymet_funcs import *

def sin14_method(station, s_d, e_d):
    '''
    Implemented method for generating sythetic data using the Sin (14R-1) method proposed by
    algorithm in paper by D.H.C. Chow and G.J. Levermore. This bases the max temp always occuring
    at hour 14:00, and assumes min temp (across all months) occurs at 5:00. Although the minimum will
    not be fully accurate during hotest/coldest months, it is a more accurate method.
    
    Method can generate hourly temp readings (24 in total), and then interpolation using CubicSpline
    is used to fit 96 points between these hourly readings to create 15 minute intervals.
    
    Returned dataframe is 365x96 of temperature readings.
    '''
    # Get daymet data from api access function
    dm_data = get_daymet_min_max(station, s_d, e_d, 0)

    # Create empty dataframe (pivot table) of measurements and dates
    s_t = datetime.strptime('00:00:00', '%H:%M:%S') # start time
    e_t = datetime.strptime('23:45:00', '%H:%M:%S') # end time
    t_i = timedelta(minutes=15) # intervals

    t_list = [(s_t + i * t_i).strftime('%H:%M:%S') for i in range((e_t - s_t) // t_i + 1)]

    t_piv = pd.DataFrame(index = [s_d.split('-')[0] + "-" + x for x in dm_data.month_day.values],
                         columns=t_list)
    
    # Create array of min/max values to interpolate between
    temp_readings = []

    max_vals = dm_data.daily_max.values
    min_vals = dm_data.daily_min.values

    for i in range(len(max_vals)):
        temp_readings.append(min_vals[i])
        temp_readings.append(max_vals[i])    
    
    # Append copy of max/min to use for initial start (discarded later)
    temp_readings.insert(0, max_vals[0])
    temp_readings.append(min_vals[-1])
    
    # Create array of times for min/max temperatures (sin14 assumes hour 14 is max)
    i = 14
    times = []

    for _ in range(366):
        times.append(i)
        i += 15
        times.append(i)
        i += 9 

    # Create hourly estimates from paper method
    tmp = []
    for idx in range(len(times)-1):
        T_next = temp_readings[idx+1]
        T_prev = temp_readings[idx]
        t_next = times[idx+1]
        t_prev = times[idx]

        for t in range(t_prev, t_next):
            tmp += [((T_next + T_prev)/2) - (((T_next - T_prev)/2) * np.cos((np.pi*(t - t_prev))/(t_next - t_prev)))]

    # Pull only 2023 data (remove previous 10 hours of 2022, future 5 hours of 2024)
    tmp = tmp[10:-5]

    # Perform cubic spline interpolation
    x = np.arange(0, len(tmp)) # get all hours
    cs = CubicSpline(x, tmp)

    # Create 15 minute readings
    np.random.seed(42)
    
    x_smooth = np.linspace(0, len(tmp), 365*96)
    y_smooth = cs(x_smooth) + np.random.normal(loc=0, scale=0.1, size=len(x_smooth))
    y_smooth = y_smooth.reshape((365,96))
    
    for i in range(365): # iterate over days
        # Add random (expected) changes in temperature
        np.random.seed(i)
        deg_chng = np.random.choice(np.arange(-1.8, 1.8, 0.1), 20)
        idx_chng = np.random.choice(np.arange(0,96), 20, replace=False)

        for idx, val in zip(idx_chng, deg_chng): # iterate over chosen index values
            y_smooth[i][idx] += val
            
    t_piv[:] = y_smooth # put data in dataframe
    
    return t_piv


def create_daymet_daily(station, s_d, e_d):
    '''
    Create synthetic data using the min/max acquired from the API call. Interpolate data between
    consecutive min/max values (96 in total), to create the 15 minute readings. CubicSpine is 
    used for fitting a line through the points, with slight noise added.
          
    NOTE: we are using Daymet min/max values as "truth" values, expecting that these are all correct.
    '''
    
    # Get daymet data from api access function
    dm_data = get_daymet_min_max(station, s_d, e_d, 0)
    
    # Create array of min/max values to interpolate between
    temp_readings = []

    max_vals = dm_data.daily_max.values
    min_vals = dm_data.daily_min.values
    
    for i in range(len(max_vals)):
        temp_readings.append(min_vals[i])
        temp_readings.append(max_vals[i])    

    temp_readings.append(min_vals[-1]) # append last min value
    
    np.random.seed(42)

    x = np.arange(0, len(temp_readings))

    # Perform cubic spline interpolation
    cs = CubicSpline(x, temp_readings)

    # Create 15 minute readings
    x_smooth = np.linspace(0, len(temp_readings), 366*96)
    y_smooth = cs(x_smooth) + np.random.normal(loc=0, scale=0.1, size=len(x_smooth))
    y_smooth = y_smooth[:-96] # remove extra day (used to avoid diverging estimate)
    
    # Create dataframe (pivot table) of measurements and dates
    tmp = pd.DataFrame(data = y_smooth.reshape((365,96)),
                       index = [s_d.split('-')[0] + "-" + x for x in dm_data.month_day.values],
                       columns=[f't{i}' for i in range(96)])
    
    return tmp
