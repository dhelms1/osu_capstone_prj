# import general libraries for data exploration and cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import seaborn as sns

def temp_val_graph(est, val_data, station_num):
    '''
    Create graph for temperature readings across the validation data, with bounds
    for the upper and lower temperature.
    '''
    
    v_idx = [x.split('-', 1)[1] for x in val_data.index.values] # get all month_day occurances
    est_val = est.loc[v_idx] # subset estimates with only val_data dates
    
    # Each day has 96 readings (repeat bounds 96 times since plot has all data)
    min_est = np.repeat(est_val.daily_min.values, 96)
    max_est = np.repeat(est_val.daily_max.values, 96)
    
    # Find month index (used for plotting)
    tmp = [int(x.split('-')[1]) for x in val_data.index.values] # get month from index
    m_idx = [x*len(val_data.columns.tolist()) for x in range(1,len(tmp)) if tmp[x] != tmp[x-1]] # month change idx
    m_idx.insert(0, 0) # insert start date
    m_idx.append(len(val_data.values.flatten())) # insert end date
    
    vals = val_data.values.flatten() # get all temperature readings from validation data

    # Plot data
    plt.figure(figsize=(10,5))
    plt.plot(vals, alpha=0.7)
    plt.plot(max_est, color='green', label='max_est')
    plt.plot(min_est, color='red', label='min_est')
    year = val_data.index[0].split('-')[0]
    plt.title(f'{year} Daily Temperature Bounds: Station {station_num}')
    plt.ylabel('Temperature')
    plt.xlabel('Month')
    plt.ylim(-15, 150)
    plt.xticks(m_idx[:-1], range(1,len(m_idx)))
    plt.legend()
    plt.show();
    
    # Calculate what percent of readings were flagged using formulas
    flagged_max = sum(vals > max_est)/len(max_est)
    flagged_min = sum(vals < min_est)/len(min_est)

    print("Flags Raised:")
    print(f"Too Hot: {sum(vals > max_est)}/{len(max_est)} ; {round(100 * flagged_max, 2)}%")
    print(f"Too Cold: {sum(vals < min_est)}/{len(min_est)} ; {round(100 * flagged_min, 2)}%")
    print('---------------------------------')
    
    return 

def plot_monthly_delta(data, d_est):
    '''
    Create 12 subplots, one for each month, that show the delta across 15 minute intervals.
    Lower and upper bounds are plotted as horizontal bars, with the delta's being presented
    as a density plot.
    '''
    s_tmp = data.copy() # create copy (don't alter original)

    s_tmp['month'] = [int(x.split('-')[1]) for x in s_tmp.index.values] # create month column

    m_var = {} 

    for m in s_tmp.month.unique(): # iterate over each month
        tmp = s_tmp[s_tmp.month == m] # subset single month
        tmp = tmp.drop(columns=['month']) # drop column
        m_var[m] = tmp.diff(axis=1).values.flatten() # get all temp changes

    fig, axs = plt.subplots(4, 3, figsize=(10,10))
    axs = axs.flatten()

    for m in s_tmp.month.unique(): # Plot each month
        sns.kdeplot(m_var[m], fill=True, ax=axs[m-1])
        axs[m-1].set_title(f'Month {m}')
        axs[m-1].set_xlabel('Delta')
        axs[m-1].set_ylabel('')

        # Plot estimated bounds (iqr method)
        b = d_est.iloc[m-1,:].values
        axs[m-1].axvline(b[0], color='darkorange', linestyle='-', linewidth=1.6, label = 'IQM - LB')
        axs[m-1].axvline(b[1], color='blue', linestyle='-', linewidth=1.6, label = 'IQM - UB')

        # Plot average
        axs[m-1].axvline(np.nanmean(m_var[m]), color='black', linestyle='--', linewidth=1.6)

    h, l = axs[1].get_legend_handles_labels()

    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=6)
    plt.tight_layout()
    plt.show()
    
    return
    
def plot_yearly_delta(data, d_est, station_num):
    '''
    Function to plot the delta bounds across an entire year (not as subplots using
    density distribution above - plot changes per 15 minutes). Upper and lower bounds
    placed for each month, with delta's being calculated as difference between current
    and previous record.
    '''

    s_tmp = data.copy() # create copy (don't alter original)

    # Find month index (used for plotting)
    tmp = [int(x.split('-')[1]) for x in s_tmp.index.values] # get month from index
    m_idx = [x*len(s_tmp.columns.tolist()) for x in range(1,len(tmp)) if tmp[x] != tmp[x-1]] # month change idx
    m_idx.insert(0, 0) # insert start date
    m_idx.append(len(s_tmp.values.flatten())) # insert end date

    vals = np.diff(s_tmp.values.flatten()) # get difference
    vals = np.insert(vals, 0, 0) # insert empty initial reading

    # Plot data
    plt.figure(figsize=(10,5))
    plt.plot(vals, alpha=0.7)
    year = data.index[0].split('-')[0]
    plt.title(f'{year} Temperature Delta Bounds: Station {station_num}')

    lb_ests = d_est.iq_lb.values # get all lower bounds
    ub_ests = d_est.iq_ub.values # get all upper bounds

    for i in range(len(m_idx)-1):
        if i == 0: # plot upper and lower bounds (label once)
            plt.hlines(ub_ests[i], m_idx[i], m_idx[i+1], color='green', label='ub_est')
            plt.hlines(lb_ests[i], m_idx[i], m_idx[i+1], color='red', label='lb_est')
        else:
            plt.hlines(ub_ests[i], m_idx[i], m_idx[i+1], color='green')
            plt.hlines(lb_ests[i], m_idx[i], m_idx[i+1], color='red')

    plt.ylabel('Delta (15 mins)')
    plt.xlabel('Month')
    plt.ylim(-10, 10)
    plt.xticks(m_idx[:-1], range(1,len(m_idx)))
    plt.legend()
    plt.show();
    
    return