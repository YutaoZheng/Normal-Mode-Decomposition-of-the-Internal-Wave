#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
from s3fs import S3FileSystem, S3Map
from scipy import signal as sg
import pandas as pd
from speccy import sick_tricks as gary

# Read the data from ncfile links
def Open_file_nocache(fname, myfs):
    """
    Load a netcdf file directly from an S3 bucket
    """
    fileobj = myfs.open(fname)
    return xr.open_dataset(fileobj)

def get_temp_qc_aodn(ds,varname='TEMP'):
    """
    Function that returns the QC's variable
    (only works with FV01 of the IMOS convention)
    """
    #find the index of bad quality data
    badidx1 = ds['{}_quality_control'.format(varname)].values <2
    #extra the data from the varnmae
    temp = ds['{}'.format(varname)]
    #replace the bad quality data with nan and find the index of them plus original nan
    badidx2 = np.isnan(temp.where(badidx1,np.nan))          
#     #replace the bad quality with the mean value of the whole period
#     values =  temp.where(badidx2, temp.mean(dim='TIME') )   
#     values_no_mean = values-temp.mean(dim='TIME')
    return temp,badidx2   

def Extract_good_data(raw_data,badidx):
    # Find the index of the last True in the beginning
    index_last_true_beginning = 0
    for i, value in enumerate(badidx):
        if value:
            index_last_true_beginning = i
        else:
            break

    # Find the index of the first True in the end
    index_first_true_end = len(badidx) - 1
    for i in range(len(badidx) - 1, -1, -1):
        if badidx[i]:
            index_first_true_end = i
        else:
            break

    # Select the good data (False values) between the last True in the beginning and the first True in the end
    good_data = raw_data[index_last_true_beginning + 1: index_first_true_end]
    return good_data

def Collect_temp(ncfiles):
    '''
    This function collects the time, depth, temperature, and the idx for bad data point from the ncfiles
    input:  ncfiles
    output: time_list,depths,temp_data_list,temp_badidx_list
    '''
    
    time_list = []
    temp_data_list = []
    temp_data_no_mean_list = []
    temp_badidx_list = []
   
    depths = [] #depth list
    
    fs = S3FileSystem(anon=True)
    
    for i in range(len(ncfiles)):
        ii = ncfiles[i]
        data = Open_file_nocache(ii,fs)
        
        obs_depth = data.instrument_nominal_depth
        depths.append(obs_depth)
    
        #get the raw temp time profile at this depth
        temp_raw,temp_badidx_raw = get_temp_qc_aodn(data, varname='TEMP')
        #remove the bad data occured at begin and end
        temp        = Extract_good_data(temp_raw,temp_badidx_raw)
        temp_badidx = Extract_good_data(temp_badidx_raw,temp_badidx_raw)
        #list append
        temp_data_list.append(temp.values)
        temp_badidx_list.append(temp_badidx.values)
        time_list.append(temp.TIME.values)


    return time_list,depths,temp_data_list,temp_badidx_list

        

#WINDOWING
def Find_window_idx(time_list,window_days):
    '''
    This function find the idx of the start and end point of the window in the time list
    input: time_list, window length in days
    output: idx of the window start and end point for time_list
    '''
    # data recording may not start at the same time
    window_idx_list = []
    for time in time_list:
        start = time[0]
        obs_duration = (time[-1] - start) #days
        n_window = int(obs_duration/np.timedelta64(window_days,'D'))
        if n_window < 1:
            print("duartion is smaller than window length")
        else:
            window_point = []
            window_idx = []
            for i in range(n_window+1):
                window_point.append(start+np.timedelta64(window_days,'D')*i)
            for ii in range(n_window):
                window_idx.append([np.logical_and(time>=window_point[ii] , time<=window_point[ii+1])])
        window_idx_list.append(window_idx) 
    return window_idx_list


def check_length(data_list):
    # Get the length of the first item
    first_item_length = len(data_list[0])

    # Assume all lengths are identical by default
    are_lengths_identical = True

    # Loop through the rest of the items and compare their lengths
    for item in data_list[1:]:
        if len(item) != first_item_length:
            are_lengths_identical = False
            break

    # Check if all lengths are identical
    if are_lengths_identical:
        return data_list
    else:
        print("Not all items in the list have the same length, check window_days")
        return data_list


def Windowing(data_list,window_idx_list):
    '''
    This function window the interested data list accourding to the provided window idx
    input: data list, window idx list
    output: windowed data list
    '''
    data_window_list = []
    for i,data in enumerate(data_list):
        window_idx = window_idx_list[i]
        
        data_window = []
        for idx in window_idx:
            data_window.append(data[idx[0]])
        
        data_window_list.append(data_window)
        
    return check_length(data_window_list)


## Find label list
def Find_window_label(windowed_time_period):
    """
    input: the time window list
    output: window start and end label list
            window year label list
            window only start label list
            window only end label list
    """
    year_list = []
    start_date_list = []
    end_date_list =[]
    
    for data in windowed_time_period:
        year       = []
        start_date = []
        end_date   = []
        
        window_length = len(data)
        for window in range(window_length):
            start_date.append(pd.to_datetime(data[window][0]).date())   #0-start
            end_date.append(pd.to_datetime(data[window][-1]).date())   #0-start
            year.append(pd.to_datetime(data[window][0]).year)  #0-start
            
        start_date_list.append(start_date)
        end_date_list.append(end_date)
        year_list.append(year)
        
    return year_list,start_date_list,end_date_list

## Find mean temp
def Find_mean_temp(windowed_temp_data,cutoff_freq,nyquist_freq):
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = sg.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_temp_avg_list = []
    for data in windowed_temp_data:  #data at each depth
        filtered_temp_avg  = []
        
        window_length = len(data)
        for window in range(window_length):
            filtered_temp = sg.filtfilt(b, a, data[window])
            filtered_temp_avg.append(np.mean(filtered_temp))
            
        filtered_temp_avg_list.append(filtered_temp_avg)
    return filtered_temp_avg_list

## Remove the mean of the whale data for the windowed list
def Remove_mean(data_windowed_list,badidx_windowed_list):
    data_with_mean_list = []
    data_without_mean_list = []

    for order1, data_windowed in enumerate(data_windowed_list):
        data_with_mean = []
        data_without_mean = []
        for order2, data in enumerate(data_windowed):
            #replace bad data with mean value
            data_replaced = np.where(badidx_windowed_list[order1][order2],data.mean(),data)
            #collected the replaced data with mean
            data_with_mean.append(data_replaced)
            #collected the replaced data without mean
            data_without_mean.append(data_replaced-np.mean(data_replaced))                         
        
        data_with_mean_list.append(data_with_mean)
        data_without_mean_list.append(data_without_mean)
        
    return data_with_mean_list,data_without_mean_list






#Sepectrum
#periodogarm
def Cal_periodogram(data,Δ):
    f,p = gary.periodogram(data,Δ)
    return f[f>=0],p[f>=0]

def Cal_var_from_Periodogram(f_list,p_list):  
    var_list = []
    for order1,f in enumerate(f_list):
        #find the freq bin
        freq_bin = f[1]-f[0]
        var = (2*freq_bin*np.sum(p_list[order1]))
        var_list.append(var)
    return var_list

## select the frequency range from corilolis freq to buoyancy freq, and omit M4 frequency
def Subset_freq(ff_list, start_freq,end_freq, bandwidth, Omit=False, omit_freq=None):
    if start_freq>=end_freq:
        raise ValueError("start_freq must be less than end_freq")
    subset_list = []
    for i in range(len(ff_list)):
        ff = ff_list[i]
        subset_range = np.less(start_freq + bandwidth,ff) & np.less(ff,end_freq)
        # If Omit is True, remove omit_freq from the list
        if Omit:
            if omit_freq is None: 
                raise ValueError("omit_freq must be provided when Omit is True")
            else: 
                #omit
                omit_freq_range = np.less(omit_freq-bandwidth, ff) & np.less(ff, omit_freq+bandwidth)
                #AND its opposite
                subset_range = subset_range & ~omit_freq_range
                subset_list.append(subset_range)  
        else:
            subset_list.append(subset_range)   
    return subset_list

def Select_frequency(data_list,index_list):
    selected_data_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        index = index_list[i]
        selected_data =data[index]
        selected_data_list.append(selected_data)
    return selected_data_list



#for find the coherent peak 
def Find_closest_index(data, target):
    min_difference = float('inf')  # Initialize with a large value
    closest_index = None

    for i, value in enumerate(data):
        difference = abs(value - target)
        if difference < min_difference:
            min_difference = difference
            closest_index = i

    return closest_index

def Coherent_peaks(peak_loc,amp_parameters,frequency):
    peaks = 0
    for i in range(len(peak_loc)):
        idx = Find_closest_index(frequency,peak_loc[i])
        amp = np.sqrt(np.power(amp_parameters[i],2)+np.power(amp_parameters[i+len(peak_loc)],2))
        peak = np.abs(amp)*sg.unit_impulse(len(frequency),idx)
        peaks = peaks + peak
    return peaks























