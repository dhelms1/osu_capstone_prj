from daymet_funcs import *
from method_funcs import *
from syn_gen_funcs import *

from itertools import combinations
from random import sample, seed
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, precision_score, f1_score

def perform_cross_val(s_num, dates, year_idx, s_pivot, n_folds, n_flags, temp_inj, delta_inj):
    '''
    Function will perform cross validation on a given station, with provided information on dates, 
    year index dictionary, and the pivot data table. The average is returned for each subset of 
    years that are used for estimation data. 
    '''
    seed(42) # set random seed

    # Dataframe to hold results from cross validation
    res = pd.DataFrame(columns=['Estimation Years',
                                'TEMP_LB_F1', 'TEMP_UB_F1', 'DELTA_LB_F1', 'DELTA_UB_F1'])
    
    # Create synthetic validation data (2023) and inject errors
    syn_data = sin14_method(s_num, '2023-01-01', '2023-12-31')
    error_data, idx_dict = error_injection(syn_data, n_flags, temp_inj, delta_inj)

    for i in range(1, len(dates)): # iterate over different number of years to use for estimation
        # Get random (repeatable) sample of n_folds of random combinations (estimation years)
        year_combos = sample(list(combinations(dates, i)), n_folds)
        
        # Define variables to hold number of flags being sets
        delta_lb_avg = 0
        delta_ub_avg = 0
        temp_lb_avg = 0
        temp_ub_avg = 0

        for j in range(n_folds): # iterate over each fold (total of n_folds)

            e = year_combos[j] # get estimation years (array of lists - subset by j)

            # Get all estimation and validation data index values
            # np.r_ turns index into range (i.e. 34:45 turns into [34, 35, ..., 44, 45])
            # flatten into single array and use to subset data frame
            e_idx = np.concatenate([np.r_[year_idx[x][0]:year_idx[x][1]] for x in e])

            # Split data (estimation and validation sets)
            s_pivot_e = s_pivot.iloc[e_idx, :]

            # Get daily estimates of min/max (with filter applied)
            temp_bounds = temp_bounds_modified(s_pivot_e, 0.99, 15, 0.8, True)

            # Get monthly estimates of min/max (with filter applied)
            delta_bounds = get_delta_est(s_pivot_e, 0.9, 1)

            # Calculate number of flags being set
            delta_l, delta_u, temp_l, temp_u = eval_metrics_updated(error_data, temp_bounds, delta_bounds, idx_dict)
                        
            # Increment running average of number of flags
            delta_lb_avg += delta_l
            delta_ub_avg += delta_u
            temp_lb_avg += temp_l
            temp_ub_avg += temp_u

        # Divide running average of number of flags
        delta_lb_avg /= n_folds
        delta_ub_avg /= n_folds
        temp_lb_avg /= n_folds
        temp_ub_avg /= n_folds

        # Add new row to data frame
        res.loc[len(res)] = [len(e), temp_lb_avg, temp_ub_avg, delta_lb_avg, delta_ub_avg]
        
    res = res.set_index("Estimation Years")
        
    return res 


def cross_val_ests(s_num, dates, year_idx, s_pivot, n_folds, n_flags, temp_inj, delta_inj):
    '''
    Function will perform cross validation on a given station, with provided information on dates, 
    year index dictionary, and the pivot data table. It will save the estimated parameters for number
    of estimation years that contain, on average, the highest accuracy for temperature. 
    
    NOTE: Delta's consitently acheive 100% accuracy, but are still return in the dictionary for
          the corresponding number of estimation years.
    '''
    seed(42) # set random seed

    # Create dictionary of df's to hold 
    best_est_dfs = {k:{'temp': [], 'delta': []} for k in range(1,9)}
    
    # Create synthetic validation data (2023) and inject errors
    syn_data = create_daymet_daily(s_num, '2023-01-01', '2023-12-31')
    error_data, idx_dict = error_injection(syn_data, n_flags, temp_inj, delta_inj)

    for i in range(1, len(dates)): # iterate over different number of years to use for estimation
        # Get random (repeatable) sample of n_folds of random combinations (estimation years)
        year_combos = sample(list(combinations(dates, i)), n_folds)

        # Define variables to hold number of flags being sets
        cur_temp_l = 0
        cur_temp_u = 0

        for j in range(n_folds): # iterate over each fold (total of n_folds)

            e = year_combos[j] # get estimation years (array of lists - subset by j)

            # Get all estimation and validation data index values
            # np.r_ turns index into range (i.e. 34:45 turns into [34, 35, ..., 44, 45])
            # flatten into single array and use to subset data frame
            e_idx = np.concatenate([np.r_[year_idx[x][0]:year_idx[x][1]] for x in e])

            # Split data (estimation and validation sets)
            s_pivot_e = s_pivot.iloc[e_idx, :]

            # Get daily estimates of min/max (with filter applied)
            temp_bounds = temp_bounds_modified(s_pivot_e, 0.99, 15, 0.8, True)

            # Get monthly estimates of min/max (with filter applied)
            delta_bounds = get_delta_est(s_pivot_e, 0.9, 1)

            # Calculate number of flags being set
            _, _, temp_l, temp_u = eval_metrics_updated(error_data, temp_bounds, delta_bounds, idx_dict)

            # Save the highest (on average) from each fold to represent the estimates
            if ((temp_l + temp_u)/2) > ((cur_temp_l + cur_temp_u)/2):
                cur_temp_l = temp_l 
                cur_temp_u = temp_u
                best_est_dfs[i]['temp'] = temp_bounds
                best_est_dfs[i]['delta'] = delta_bounds
        
    return best_est_dfs 


def cross_val_temp_graph(est, station_num):
    '''
    Create graph for temperature readings across the validation data, with bounds
    for the upper and lower temperature for each number of estimation years.
    '''
    
    val_data = create_daymet_daily(station_num, '2023-01-01', '2023-12-31')
    
    # Find month index (used for plotting)
    tmp = [int(x.split('-')[1]) for x in val_data.index.values] # get month from index
    m_idx = [x*len(val_data.columns.tolist()) for x in range(1,len(tmp)) if tmp[x] != tmp[x-1]] # month change idx
    m_idx.insert(0, 0) # insert start date
    m_idx.append(len(val_data.values.flatten())) # insert end date
    
    vals = val_data.values.flatten() # get all temperature readings from validation data
    
    # Plot data
    plt.figure(figsize=(10,5))
    plt.plot(vals, alpha=0.7)
    
    v_idx = [x.split('-', 1)[1] for x in val_data.index.values] # get all month_day occurances
    
    colors = cm.hsv(np.linspace(0, 1, 9))
    
    for i in range(1, len(est) + 1):
        est_val = est[i]['temp']
        est_val = est_val.loc[v_idx] # subset estimates with only val_data dates
        # Each day has 96 readings (repeat bounds 96 times since plot has all data)
        min_est = np.repeat(est_val.daily_min.values, 96)
        max_est = np.repeat(est_val.daily_max.values, 96)

        plt.plot(max_est, label=f'{i} year(s)', color=colors[i]) #, linestyle=line_styles[i-1])
        plt.plot(min_est, color=colors[i]) #, linestyle=line_styles[i-1])
        
    year = val_data.index[0].split('-')[0]
    plt.title(f'{year} Daily Temperature Bounds: Station {station_num}')
    plt.ylabel('Temperature')
    plt.xlabel('Month')
    plt.ylim(-15, 140)
    plt.xticks(m_idx[:-1], range(1,len(m_idx)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show();

    return 

def cross_val_delta_graph(est, station_num):
    '''
    Function to plot the delta bounds across an entire year. Upper and lower bounds
    placed for each month, with delta's being calculated as difference between current
    and previous record.
    '''

    val_data = sin14_method(station_num, '2023-01-01', '2023-12-31')

    # Find month index (used for plotting)
    tmp = [int(x.split('-')[1]) for x in val_data.index.values] # get month from index
    m_idx = [x*len(val_data.columns.tolist()) for x in range(1,len(tmp)) if tmp[x] != tmp[x-1]] # month change idx
    m_idx.insert(0, 0) # insert start date
    m_idx.append(len(val_data.values.flatten())) # insert end date

    vals = np.diff(val_data.values.flatten()) # get difference
    vals = np.insert(vals, 0, 0) # insert empty initial reading

    # Plot data
    plt.figure(figsize=(10,5))
    plt.plot(vals, alpha=0.7)
    
    colors = cm.hsv(np.linspace(0, 1, 9))
    
    for j in range(1, len(est) + 1):
        est_val = est[j]['delta']

        lb_ests = est_val.iq_lb.values # get all lower bounds
        ub_ests = est_val.iq_ub.values # get all upper bounds

        for i in range(len(m_idx)-1):
            if i == 0: # plot upper and lower bounds (label once)
                plt.hlines(ub_ests[i], m_idx[i], m_idx[i+1], color=colors[j], label=f'{j} year(s)')
                plt.hlines(lb_ests[i], m_idx[i], m_idx[i+1], color=colors[j])
            else:
                plt.hlines(ub_ests[i], m_idx[i], m_idx[i+1], color=colors[j])
                plt.hlines(lb_ests[i], m_idx[i], m_idx[i+1], color=colors[j])

    year = val_data.index[0].split('-')[0]
    plt.title(f'{year} Monthly Delta Bounds: Station {station_num}')
    plt.ylabel('Delta (15 mins)')
    plt.xlabel('Month')
    plt.ylim(-8, 8)
    plt.xticks(m_idx[:-1], range(1,len(m_idx)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show();
    
    return


def eval_metrics_updated(val_data, temp_b, delta_b, idx_dict):
    '''
    Calculate the f1 score, on the synthetic Daymet data with injected errors, for the temperature
    and delta (step) parameters. Labels are generated for both the true values and predicted values,
    then used an input to the evaluation metric.
    
    Returns the f1 for the temperature and delta lower / upper bounds.
    '''
    # -------------------- Temperature bounds flag calculation --------------------

    v_idx = [x.split('-', 1)[1] for x in val_data.index.values] # get all month_day occurances
    est_val = temp_b.loc[v_idx] # subset estimates with only val_data dates

    vals = val_data.values.flatten() # get values of temperature data

    t_min_est = np.repeat(est_val.daily_min.values, 96) # repeat for each daily value (96 per day)
    t_max_est = np.repeat(est_val.daily_max.values, 96)
    
    # Generate true labels (all zeros, except for injected index)
    min_labels = np.zeros(len(vals))
    max_labels = np.zeros(len(vals))

    for i in idx_dict['temp_lb']: # update min temp where error injected
        min_labels[i] = 1

    for i in idx_dict['temp_ub']: # update max temp where error injected
        max_labels[i] = 1

    # Generate predicated labels
    min_preds = [int(x > y) for x,y in zip(t_min_est, vals)]
    max_preds = [int(x < y) for x,y in zip(t_max_est, vals)]
    
    t_lb_eval = round(f1_score(min_labels, min_preds), 3)
    t_ub_eval = round(f1_score(max_labels, max_preds), 3)
        
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
    
    # Generate true labels (all zeros, except for injected index)
    min_labels = np.zeros(len(vals))
    max_labels = np.zeros(len(vals))

    for i in (idx_dict['temp_lb'] + idx_dict['delta_lb']): # update min temp where error injected
        min_labels[i] = 1
        max_labels[i+1] = 1 # catch spike back up

    for i in (idx_dict['temp_ub'] + idx_dict['delta_ub']): # update max temp where error injected
        max_labels[i] = 1
        min_labels[i+1] = 1 # catch spike back down

    # Generate predicated labels
    min_preds = [int(x > y) for x,y in zip(d_lb_est, vals)]
    max_preds = [int(x < y) for x,y in zip(d_ub_est, vals)]
    
    d_lb_eval = round(f1_score(min_labels, min_preds), 3)
    d_ub_eval = round(f1_score(max_labels, max_preds), 3)
    
    return d_lb_eval, d_ub_eval, t_lb_eval, t_ub_eval


def plot_eval(df_cv, save=False):
    '''
    Plot evaluation metrics for each parameter, across all stations used in cross-validation.
    '''
    s_vals = list(df_cv.keys()) # get all station numbers
    p_vals = list(df_cv.values())[0].columns.values # get df column names
    e_y = np.arange(1,9) # array of estimation years

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12,9))
    axs = axs.flatten()
    
    titles = ['Temperature Lower Bound', 'Temperature Upper Bound', 
              'Delta Lower Bound', 'Delta Upper Bound']

    # Plot each eval metric on a subplot
    for i in range(4):
        for s in s_vals:
            axs[i].plot(e_y, df_cv[s][p_vals[i]], label=s)
        axs[i].set_title(titles[i])
        if i % 2 == 0:
            axs[i].set_ylabel('F1 Score')
        if i >= 2:
            axs[i].set_xlabel('Number of Estimation Years')

    # Create legend for station numbers
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), title="Station")

    plt.tight_layout()
    plt.show()
    
    if save:
        fig.savefig('cross_val_results.png', bbox_inches='tight')
    
    return

def cross_val_graphs(est, station_num):
    val_data = sin14_method(station_num, '2023-01-01', '2023-12-31')
    
    # Find month index (used for plotting)
    tmp = [int(x.split('-')[1]) for x in val_data.index.values] # get month from index
    m_idx = [x*len(val_data.columns.tolist()) for x in range(1,len(tmp)) if tmp[x] != tmp[x-1]] # month change idx
    m_idx.insert(0, 0) # insert start date
    m_idx.append(len(val_data.values.flatten())) # insert end date
    
    vals = val_data.values.flatten() # get all temperature readings from validation data
    
    fig, axs = plt.subplots(1, 2, figsize=(13,5))
    
    temp_vals = val_data.values.flatten() # get all temperature readings from validation data
    
    delta_vals = np.diff(val_data.values.flatten()) # get difference
    delta_vals = np.insert(delta_vals, 0, 0) # insert empty initial reading
    
    axs[0].plot(temp_vals)
    axs[1].plot(delta_vals)
    
    year = val_data.index[0].split('-')[0]
    v_idx = [x.split('-', 1)[1] for x in val_data.index.values] # get all month_day occurances
    colors = cm.hsv(np.linspace(0, 1, 9))
    
    # TEMP BOUNDS
    for i in range(1, len(est) + 1):
        est_val = est[i]['temp']
        est_val = est_val.loc[v_idx] # subset estimates with only val_data dates
        # Each day has 96 readings (repeat bounds 96 times since plot has all data)
        min_est = np.repeat(est_val.daily_min.values, 96)
        max_est = np.repeat(est_val.daily_max.values, 96)

        axs[0].plot(max_est, label=f'{i} year(s)', color=colors[i]) #, linestyle=line_styles[i-1])
        axs[0].plot(min_est, color=colors[i]) #, linestyle=line_styles[i-1])
    
    # DELTA BOUNDS
    for j in range(1, len(est) + 1):
        est_val = est[j]['delta']

        lb_ests = est_val.iq_lb.values # get all lower bounds
        ub_ests = est_val.iq_ub.values # get all upper bounds

        for i in range(len(m_idx)-1):
            if i == 0: # plot upper and lower bounds (label once)
                axs[1].hlines(ub_ests[i], m_idx[i], m_idx[i+1], color=colors[j])
                axs[1].hlines(lb_ests[i], m_idx[i], m_idx[i+1], color=colors[j])
            else:
                axs[1].hlines(ub_ests[i], m_idx[i], m_idx[i+1], color=colors[j])
                axs[1].hlines(lb_ests[i], m_idx[i], m_idx[i+1], color=colors[j])
    
    
    axs[0].set_title(f'{year} Daily Temperature Bounds: Station {station_num}')
    axs[0].set_ylabel('Temperature (F)')
    axs[0].set_xlabel('Month')
    axs[0].set_ylim(-15, 140)
    axs[0].set_xticks(m_idx[:-1], range(1,len(m_idx)))
    
    axs[1].set_title(f'{year} Monthly Delta Bounds: Station {station_num}')
    axs[1].set_ylabel('Delta (15 mins)')
    axs[1].set_xlabel('Month')
    axs[1].set_ylim(-8, 8)
    axs[1].set_xticks(m_idx[:-1], range(1,len(m_idx)))
    
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=8)
    
    plt.tight_layout()
    plt.show()
    
    #fig.savefig('cross_val.png', bbox_inches='tight')
    
    return
