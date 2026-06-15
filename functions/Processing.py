#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
# from s3fs import S3FileSystem, S3Map
from scipy import signal as sg
import pandas as pd
from speccy import sick_tricks as gary
from speccy import ut
from . import Cov
import gsw
import os

import shutil
import sys
#initial
#Aadd packages_Tao into the os 
current_path = os.getcwd()
sys.path.append(os.path.abspath(current_path+'\iwaves_Tao'))
sys.path.append(os.path.abspath(current_path+'\pIMOS_Tao'))
#then import
import iwaves
import pIMOS
import pIMOS.xrwrap.pl2_stacked_mooring as sm
import pIMOS.xrwrap.pl3_stacked_mooring as sm03
import pIMOS.utils.UWA_archive_utils as ai
import pickle

# LV1 data processing
## Group the downloaded nc files
def Group_nc_files(source_dir, dest_dir):
    """
    Groups .nc files from source_dir into subfolders in dest_dir based on trip code
    found after a specified site_name in the filename.

    :param source_dir: Directory where the .nc files are located
    :param dest_dir: Destination directory where grouped files will be moved
    :param site_name: The site name to locate in the filename (the unique code appears after this)
    """
    # Create the destination directory if it doesn't exist
    # dest_dir = os.path.abspath(dest_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Iterate through all files in the source directory   
    current_path = os.getcwd()
    source_directory = os.path.join(current_path,source_dir)
    for filename in os.listdir(source_directory):
        if filename.endswith('.nc'):
            site_name_position = filename.find('FV01')+5
            site_name          = filename[site_name_position:site_name_position+6]
            if site_name_position != -1:
                #using the site_name + trip number as the unique code
                unique_code = filename[site_name_position:site_name_position + len(site_name)+5]
                subfolder = os.path.join(dest_dir, unique_code)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                #Move the file to the corresponding subfolder
                source_path = os.path.join(source_directory,filename)
                destination_path = os.path.join(subfolder, filename)
                #Using shutil.copy() to ensure the file is always replaced, 
                shutil.copy(source_path, destination_path)
            else:
               raise(Exception(f"Site name '{site_name}' not found in {filename}"))
                
    return print("Move and group completed")



## Covert IMOS nc data
def Process_IMOS_source_file(file,stack_variables,attrs_to_join, selected_raw_attrs,
                             window_duration):
    ds = xr.open_dataset(file).copy(deep=True)
    
    #convert to negative depth
    ds['DEPTH'] = -ds['DEPTH'] 
    ds['NOMINAL_DEPTH'] = -ds['NOMINAL_DEPTH']

    #rename data variables
    ds = ds.rename({'LATITUDE':'lat_nom'})
    ds = ds.rename({'LONGITUDE':'lon_nom'})
    ds = ds.rename({'TIME':'time'})
    ds = ds.rename({'NOMINAL_DEPTH':'z_nom'})
    if 'DEPTH' in ds:
        #'DEPTH' in IMOS data = z_hat (knockdown corrected)
        ds = ds.rename({'DEPTH':'z_hat'})
        # print('ds has DEPTH renaming to z_hat')
    else:
        print('no DEPTH, may need knockdown correction')

    #re-define the qc variable
    for var in stack_variables:
        if var in ds and 'ancillary_variables' in ds[var].attrs:
            #remane the attributes of the data variable
            ds[var].attrs['qc_variable'] = ds[var].attrs.pop('ancillary_variables')
            flag_name = ds[var].attrs['qc_variable']
            #IMOS data define the bad data as qc>2 
            #while pIMOS package define the bad as qc>0
            #need to convert the defination 
            ds[flag_name] = ds[flag_name].where(ds[flag_name] >2, -1)
        else:
             raise(Exception('varibale is not in ds'))
    
    #rename attrs
    ds.attrs['author'] = 'AIMS'
    ds.attrs['title'] = ds.attrs['title'][-9:]  #to shorten
    ds.attrs['time_deployment_start'] = ds.attrs['time_deployment_start'][0:10]  #to shorten
    raw_attrs = ds.attrs.copy()
    ds.attrs = {} #reset
    if len(attrs_to_join) == len(selected_raw_attrs):
        for i in range(len(attrs_to_join)):
            ds.attrs[attrs_to_join[i]]= raw_attrs.get(selected_raw_attrs[i])
    else:
        raise(Exception('len of mapping is not equal, check'))

    #remove bad data in the beginning and end of deployment
    #according to the first stack_variables (assume 2nd is z_hat)
    ds = Extract_good_data(ds,stack_variables[0])
    #check whether ds is too short
    ds_duration = (ds['time'][-1]-ds['time'][0])/ np.timedelta64(1, 'D')
    if ds_duration.values >= window_duration:
        #store the ds into nc file
        current_directory = os.getcwd() #Get the current working directory
        # Define the folder path and create folder
        folder_path = os.path.join(current_directory, 'PL1 Data')
        os.makedirs(folder_path, exist_ok=True)
        subfolder_path = os.path.join(folder_path, ds.attrs['trip_deployed'])
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        file_path = os.path.join(subfolder_path, 'Converted_IMOS_dataset_{}_{}.nc'.format(ds.attrs['trip_deployed'],ds.nominal_instrument_height_asb))
        # Save the data array to a NetCDF file
        ds.to_netcdf(file_path)
    else:
        print('data of {} at depth of {} m has duration of {} days so short that excluding'.format(
            ds.attrs['trip_deployed'],np.round(ds['z_nom'].values,1),np.round(ds_duration.values)))


def Process_Gourp_IMOS_source_file(Group_source_file,stack_variables,attrs_to_join, selected_raw_attrs,
                                   window_duration = 160):
    '''
    input: destination directory
    '''
    current_path = os.getcwd()
    Group_source_file_dir = os.path.join(current_path,Group_source_file)
    for root, dirs, files in os.walk(Group_source_file_dir):
        # Modify dirs in-place to exclude any directory starting with "_"
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        for file in files:
            if file.endswith('.nc'):
                # Full path of the .nc file
                nc_file_path = os.path.join(root, file)
                try:
                    # Call the read_IMOS_data function for each .nc file
                    Process_IMOS_source_file(nc_file_path,stack_variables,attrs_to_join, selected_raw_attrs,window_duration)
                    # print(f"Processed: {nc_file_path}")
                except Exception as e:
                    print(f"Failed to process {nc_file_path}: {e}")
                
    return print('IMOS Lv1 data processing completed, name as PL1 Data')

# Lv2 data processing
def Compare_folder_counts(folder_path1, folder_path2):
    # Check if the paths exist
    if not os.path.exists(folder_path1):
        print(f"Warning: {folder_path1} does not exist.")
        return
    if not os.path.exists(folder_path2):
        print(f"Warning: {folder_path2} does not exist.")
        return
    
    # Get the number of folders in each path
    folders1 = [name for name in os.listdir(folder_path1)]
    folders2 = [name for name in os.listdir(folder_path2)]
    
    # Count the number of folders
    folder1_count = len(folders1)
    folder2_count = len(folders2)
    
    # Compare the counts and print a warning if they are not equal
    if folder1_count != folder2_count:
        print("Warning: The number of folders in 1 ({}) and 2 ({}) are not equal.".format(folder1_count,folder2_count))
    else:
        print("The number of folders in 1 ({}) and 2 ({}) are equal.".format(folder1_count,folder2_count))

def Construct_PL2_data(PL1_folder_name,PL2_folder_name,stack_variables,dt_sec=60, start=None, end=None, z_method = 'z_nom'):
    '''
    ddd
    '''
    # Find the PL1 folder path
    current_directory = os.getcwd()
    PL1_folder_path = os.path.join(current_directory, PL1_folder_name)
    #create a new folder for PL2 data
    PL2_folder_path = os.path.join(current_directory,PL2_folder_name)
    os.makedirs(PL2_folder_path, exist_ok=True)
    
    #Find all the PL2 nc files
    for root, dirs, files in os.walk(PL1_folder_path):
        nc_files = [file for file in files if file.endswith('.nc')]
        # Print out files that do not end with .nc
        non_nc_files = [file for file in files if not file.endswith('.nc')]
        if non_nc_files:
            print("Files not ending with .nc:")
            for non_nc_file in non_nc_files:
                print(non_nc_file)
        #Full paths to .nc files    
        nc_file_path = [os.path.join(root, nc_file) for nc_file in nc_files]
        #lv1 to lv2 processing by pIMOS
        if nc_file_path:
            try:
                stacked_lv2 = sm.from_fv01_archive(nc_file_path,stack_variables, dt_sec = dt_sec, start=start,end=end,z_method=z_method)
                #save the file in a new folder
                PL2_file_name = '\stacked_LV2_{}.nc'.format(';'.join(set(stacked_lv2.ds.attrs['trip_deployed'].split(';'))))
                stacked_lv2.ds.to_netcdf(path = PL2_folder_path+PL2_file_name)
            except Exception as e:
                print(f"Failed to process {nc_file_path}: {e}")

    Compare_folder_counts(PL1_folder_path, PL2_folder_path)
    return print('IMOS Lv2 data processing completed, named as{}'.format(PL2_folder_name))

# Lv3 data processing
def Construct_PL3_data(PL2_folder_name,PL3_folder_name, stack_variables,
                       nmodes=4, z_method = 'z_nom',fit=True,density_func='double_tanh_new'):
    '''
    ddd
    '''
    # Find the PL2 folder path
    current_directory = os.getcwd()
    PL2_folder_path = os.path.join(current_directory,PL2_folder_name)
    #create a new folder for PL3 data
    PL3_folder_path = os.path.join(current_directory,PL3_folder_name)
    os.makedirs(PL3_folder_path, exist_ok=True)
    
    #PL2 processing
    for root, dirs, files in os.walk(PL2_folder_path):
        nc_files = [file for file in files if file.endswith('.nc')]
        # Print out files that do not end with .nc
        non_nc_files = [file for file in files if not file.endswith('.nc')]
        if non_nc_files:
            print("Files not ending with .nc:")
            for non_nc_file in non_nc_files:
                print(non_nc_file)
        #Full paths to .nc files    
        PL2_nc_file_path = [os.path.join(root, nc_file) for nc_file in nc_files]

        if PL2_nc_file_path:
            for PL2_nc_file in PL2_nc_file_path:
                #lv2 to lv3 processing by pIMOS
                stacked_lv3 = sm03.from_fv02(PL2_nc_file,S=35.6,)
                #fill the nans in stack_variables
                for var in stack_variables:
                    nan_values = np.isnan(stacked_lv3.ds[var]).sum(axis=1)
                    critical = len(stacked_lv3.ds['time'])*0.25
                    nan_index = nan_values > critical
                    if np.any(nan_index):
                        print("Too many nan appearing at {}!! check depths at {}".format(var,stacked_lv3.ds['z_nom'][nan_index].values))
                        stacked_lv3.fill_nans(tvar=var)
                    else:
                        stacked_lv3.fill_nans(tvar=var)
                #constant salinity
                stacked_lv3.calc_salinity(method='constant_sal', S=35.6,tvar='TEMP')
                #background density (rho_bar) 
                #assume stack_variable is TEMP
                stacked_lv3.calc_density(tvar=stack_variables[0],z_method=z_method)
                # Remove entire column if any data <1020 or >1030
                stacked_lv3.ds = stacked_lv3.ds.where((stacked_lv3.ds['sea_water_potential_density'] > 1020) 
                                                    & (stacked_lv3.ds['sea_water_potential_density'] < 1030), drop=True)
                #find the index when density > 0 
                idx = np.sum(np.isnan(stacked_lv3.ds['sea_water_potential_density']), axis=0).values > 0
                stacked_lv3.ds = stacked_lv3.ds.isel(time=~idx)
                #create timeslow window
                stacked_lv3.calc_timeslow(slow_tstep_h=3)
                #estimate density by fitting to density_func then compute amp
                stacked_lv3.calc_bmodes(bar_tstep_h=24*3,nmodes=nmodes,density_func=density_func,fit=fit)
                #save the file in a new folder
                PL3_file_name = '\stacked_LV3_{}.nc'.format(';'.join(set(stacked_lv3.ds.attrs['trip_deployed'].split(';'))))
                stacked_lv3.ds.to_netcdf(path = PL3_folder_path+PL3_file_name)
                print('lv2 to lv3 processing done')

    Compare_folder_counts(PL2_folder_path, PL3_folder_path)
    
    return print('All IMOS Lv3 data processing completed, named as {}'.format(PL3_folder_name))


# Read the data from ncfile links
def Extract_good_data(ds,var):
    #perform qaqc first
    flag_name = ds[var].attrs['qc_variable']
    badidx = ds[flag_name].values > 0
    ds[var][badidx] = np.nan
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
    good_data = ds.isel(time = slice(index_last_true_beginning + 1, index_first_true_end))
    return good_data


        
#WINDOWING
def windowing(ds,window_size_days):
    windowed_ds_list = []
    time = ds.time
    obs_duration = (time[-1] - time[0]).values.astype('float')/1e9/86400 
    n_window     = int(obs_duration/window_size_days)
    if n_window < 1:
        print("duartion is smaller than window length")
    else:
        for i in range(n_window):
            start_time = time[0]+np.timedelta64(window_size_days,'D')*i
            end_time   = start_time+np.timedelta64(window_size_days,'D')
            if end_time > time[-1]:
                raise ValueError(f"Error: end_time {end_time.values} exceeds the dataset's last time value {time[-1].values}")
            else:
                windowed_ds = ds.sel(time=slice(start_time, end_time))
                windowed_ds_list.append(windowed_ds.sel(timeslow=slice(start_time, end_time)))
                
    return windowed_ds_list

def Cal_density_fit_percentage(ds):
    rho    = ds['sea_water_potential_density'] # time, z_nom
    rhobar = ds['rhobar'] # time_slow, z_interp    
    rhofit = ds['rhofit'] # time, z_interp   #the density fit summering all modes
    #coverting
    rho_new    = rho.interp(time=ds.timeslow)
    rhofit_new = rhofit.interp(time=ds.timeslow)
    diff1 = rho_new.interp(z_nom=ds.z_interp) - rhobar
    diff2 = rhofit_new - rhobar
    #drop nan 
    diff1 = diff1.dropna(dim="z_interp", how='all')
    diff2 = diff2.dropna(dim="z_interp", how='all')
    #compute the variance and then integrate over depth
    detph_integral1 = np.power(diff1,2).integrate("z_interp")
    detph_integral2 = np.power(diff2,2).integrate("z_interp")
    return detph_integral2/detph_integral1*100

def Read_PL3_data(PL3_file, percentage=0.8, window_size_days=80):
    current_directory = os.getcwd()
    PL3_folder_path = os.path.join(current_directory, PL3_file)
    nc_files = [f for f in os.listdir(PL3_folder_path) if f.endswith('.nc')]
    Dataset_list = []
    for nc_file in nc_files:
        filepath = os.path.join(PL3_folder_path, nc_file)
        ds = xr.open_dataset(filepath)
        Dataset_list.append(ds)
        mode = Dataset_list[0].modes.values
    print(f"There are {len(Dataset_list)} raw PL3 data (no windowing)")
    # Create dict and windowing
    ds_dict = {}
    site_summary = {}  # Dictionary to store removal and kept counts by site
    for order1, ds in enumerate(Dataset_list):
        ds_windowed_list = windowing(ds, window_size_days)
        for order2, ds_windowed in enumerate(ds_windowed_list):
            site = ds_windowed.attrs['trip_deployed'][0:6]
            ds_name = f"{site}_P{order1}_{order2}"
            # Initialize site entry in summary dictionary
            if site not in site_summary:
                site_summary[site] = {"kept": 0, "removed": 0, 
                                      "removed_overfitting": 0, "removed_large_amplitude": 0,
                                      "removed_nan":0,
                                      "start_date": None, "end_date": None}
            # Update observation duration
            start_time = ds_windowed.time.values.min()
            end_time = ds_windowed.time.values.max()
            if site_summary[site]["start_date"] is None or start_time < site_summary[site]["start_date"]:
                site_summary[site]["start_date"] = start_time
            if site_summary[site]["end_date"] is None or end_time > site_summary[site]["end_date"]:
                site_summary[site]["end_date"] = end_time
            # Check for overfitting
            performance = Cal_density_fit_percentage(ds_windowed).quantile(percentage, skipna=True).values
            if performance < 100:
                ds_dict[ds_name] = ds_windowed
                site_summary[site]["kept"] += 1  # Increment kept count
            else:
                site_summary[site]["removed"] += 1
                site_summary[site]["removed_overfitting"] += 1
                print(f"{ds_name} has over-fitting {performance}%")

    # Read this dict and extrac the info
    A_n_dict = {}
    Time_dict = {}
    Start_date_dict = {}
    for i, ds in ds_dict.items():
        # Read ds using read_dict logic
        A_obs_list = []
        Time_list = []
        Start_date_list = []
        exclude_dataset = False
        site = i.split("_")[0]
        for mode_number in mode:
            A_n_ds = xr.concat([ds['A_n'][:, mode_number]], dim='time')
            Time_ds = A_n_ds.time
            Start_date = Time_ds[0].values.astype('datetime64[D]')
            # Check absolute amplitudes > 100m and decide action
            count_large_amplitude = (np.abs(A_n_ds) >= 100).sum().values
            if 0 < count_large_amplitude <= 200:
                print(f"{i} mode {mode_number + 1} has {count_large_amplitude} amplitudes > 100m. Setting them to NaN.")
                A_n_ds = A_n_ds.where(np.abs(A_n_ds) < 100, np.nan)
            elif count_large_amplitude > 200:
                print(f"Warning: {i} mode {mode_number + 1} has {count_large_amplitude} amplitudes > 100m. Dataset excluded.")
                exclude_dataset = True
                site_summary[site]["kept"] -= 1  # Decrement kept count
                site_summary[site]["removed"] += 1
                site_summary[site]["removed_large_amplitude"] += 1
            else:
                A_n_ds = A_n_ds
            # Handle NaN values
            nan_count = np.isnan(A_n_ds).sum().values
            if nan_count > 0:
                nan_percentage = nan_count / len(A_n_ds)
                if nan_percentage > 0.10: #10%
                    print(f"{i}: {nan_count} ({nan_percentage:.2%}) NaN values. Dataset excluded.")
                    exclude_dataset = True
                    # nan_removed_list.append(i)
                    site_summary[site]["kept"] -= 1  # Decrement kept count
                    site_summary[site]["removed"] += 1
                    site_summary[site]["removed_nan"] += 1
                    break  # Exit loop for this dataset if exclusion condition is met
                else:
                    A_n_ds = A_n_ds.interpolate_na('time', fill_value="extrapolate")
             # Add processed data to lists
            A_obs_list.append(A_n_ds)
            Time_list.append(Time_ds)
            Start_date_list.append(Start_date)
        if not exclude_dataset:
            # Add dataset to the dictionaries
            A_n_dict[i] = A_obs_list
            Time_dict[i] = Time_list
            Start_date_dict[i] = Start_date_list

    # Print results
    print(f"\nThere are {len(A_n_dict)} 'good' data")
    print("Summary of segments by site:")
    for site, counts in site_summary.items():
        start_date = np.datetime64(counts['start_date']).astype('datetime64[D]').item().strftime('%Y-%m-%d')
        end_date = np.datetime64(counts['end_date']).astype('datetime64[D]').item().strftime('%Y-%m-%d')
        print(f"Site {site}: Kept = {counts['kept']}, "
              f"Removed = {counts['removed']} (Overfitting = {counts['removed_overfitting']}, "
              f"Large Amplitude = {counts['removed_large_amplitude']} High NaN Percentage = {counts['removed_nan']}), "
              f"Observation Duration = {start_date} to {end_date}")
    print("Read lv3 data completed")
    
    return ds_dict, A_n_dict, Time_dict, Start_date_dict

def read_pickle(file_name):
    # Load data from the file
    with open('{}'.format(file_name), 'rb') as f:
        result = pickle.load(f)
    return result[0]

def Extract_site_info(ds_dict, tolerance = 0.1):
    lat_lon_site = []  # To store tuples of (latitude, longitude, site_name)
    for i in ds_dict:
        lat = float(ds_dict[i]['lat_nom'].values)
        lon = float(ds_dict[i]['lon_nom'].values)
        site = ds_dict[i].attrs['site']
        # Check if the lat/lon pair already exists within the tolerance
        match_found = False
        for lat_lon in lat_lon_site:
            # Calculate the absolute difference
            lat_diff = abs(lat_lon[0] - lat)
            lon_diff = abs(lat_lon[1] - lon)
            # If the differences are smaller than the tolerance, consider it a match
            if lat_diff < tolerance and lon_diff < tolerance:
                match_found = True
                break
        # If no match is found, add the new pair
        if not match_found:
            lat_lon_site.append((lat, lon, site))
    # Extract the lists back
    lat_list = [item[0] for item in lat_lon_site]
    lon_list = [item[1] for item in lat_lon_site]
    site_name = [item[2] for item in lat_lon_site]
        
    return site_name,lat_list,lon_list


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
                for freq in omit_freq:
                    omit_freq_range = np.less(freq-bandwidth, ff) & np.less(ff, freq+bandwidth)
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



#estimate thermocline depth from Temperature data
def find_max_η_depth_from_temp(temp_df,lat,min_threshold):
    '''
    data: from Whole_Soln_df_M1P2_clean with selected site and season
    '''
    N2_list = []
    p_mid_list = []  
    CT_list = []
    p_list = [] 
    for year in temp_df['year'].unique():
        b = temp_df[temp_df['year']==year].sort_values(by='depth_round').copy()
        p = gsw.conversions.p_from_z(b['depth_round'].unique(),lat)
        SA = np.array([34.6]*len(p))
        CT = gsw.conversions.CT_from_t(SA,b['mean_temp'],p)
        N2,p_mid = gsw.stability.Nsquared(SA,CT,p,lat)

        N2_list.append(N2)
        p_mid_list.append(p_mid)
        CT_list.append(CT)
        p_list.append(p)
         
    p_mid = np.concatenate(p_mid_list)
    N2 = np.concatenate(N2_list) 
    #df_N2
    data ={'N2':N2,'p_mid':p_mid}
    df = pd.DataFrame(data).sort_values(by='p_mid')
    n2_mean = df.groupby('p_mid')['N2'].mean().reset_index()
    #find the idx for max
    max_index = n2_mean['N2'].idxmax()
    max_value_depth_up   = gsw.conversions.z_from_p(n2_mean.loc[max_index, 'p_mid'],lat)+min_threshold
    max_value_depth_down = gsw.conversions.z_from_p(n2_mean.loc[max_index, 'p_mid'],lat)-min_threshold
        
    return max_value_depth_up, max_value_depth_down

def Find_avg_thermocline_depth(dataset,loc,lat,seasons,min_threshold=40):
    depth_list=[]
    for j, season in enumerate(seasons):
        data = dataset.loc[(dataset['site'] =='{}'.format(loc)) & (dataset['season'] =='{}'.format(season))].copy()
        #find the depth of max η_c
        max_value_depth_up, max_value_depth_down = find_max_η_depth_from_temp(data,lat,min_threshold)
        thermocline_depth = (max_value_depth_up+max_value_depth_down)/2
        depth_list.append(thermocline_depth)
    
    return np.average(depth_list)

def Add_season(result_df): 
    season_label_list = []# = ['Feb-Apr','May-Jul','Aug-Oct','Nov-Jan']
    for i in result_df['Start_date']:
        month = i.month
        if   month in [1,2,3]:
            season = 'Feb-Apr'
        elif month in [4,5,6]:
            season = 'May-Jul'
        elif month in [7,8,9]:
            season = 'Aug-Oct'
        else: 
            season = 'Nov-Jan'
        
        season_label_list.append(season)
    result_df['season'] = season_label_list
        
    return result_df

# Constract the pd dataframe for results
## PANDA DATAFRAME
def transfer_dict_to_df(dict,dict_name):
    df = pd.DataFrame.from_dict(dict,orient='index')
    #add column of Mode
    df = df.melt(var_name='Mode', value_name=dict_name, ignore_index=False)
    return df

def Construct_soln_df_from_dict(Start_date_dict, HA_var_dict,HA_M2_var_dict,HA_S2_var_dict,
                                Var_residual_dict,Var_model_fit_dict,
                                Soln_model_fit_dict,Whittle_dict,
                                parameter_name):
    HA_var_df         = transfer_dict_to_df(HA_var_dict,'HA_var')
    HA_M2_var_df      = transfer_dict_to_df(HA_M2_var_dict,'HA_M2_var')
    HA_S2_var_df      = transfer_dict_to_df(HA_S2_var_dict,'HA_S2_var')
    Start_date_df     = transfer_dict_to_df(Start_date_dict,'Start_date')
    Var_residual_df   = transfer_dict_to_df(Var_residual_dict,'Var_residual')
    Var_model_fit_df  = transfer_dict_to_df(Var_model_fit_dict,'Var_model_fit')
    Whittle_list_df   = transfer_dict_to_df(Whittle_dict,'Whittle_value')

    #extract the model parameter from the solution df
    Soln_model_fit_df = transfer_dict_to_df(Soln_model_fit_dict,'S')
    Soln_model_fit_df[parameter_name] = pd.DataFrame(Soln_model_fit_df["S"].tolist(), index=Soln_model_fit_df.index)
    Soln_model_fit_df.drop(columns=["S"],inplace=True)
    #concatdf
    Final_df_all = pd.concat([HA_var_df,HA_M2_var_df,HA_S2_var_df,
                              Start_date_df,Var_residual_df,Var_model_fit_df,
                               Whittle_list_df,Soln_model_fit_df],axis=1)
    
    #remove duplicated columns
    Final_df_all = Final_df_all.loc[:,~Final_df_all.columns.duplicated()]
    #add site name & parameter name
    Final_df_all.insert(0,'Site',Final_df_all.index.str[:6])
    #sort
    Final_df_all = Add_season(Final_df_all)
    Final_df_all.sort_values(by=['Site','Start_date','Mode'],inplace=True)
    Final_df_all.reset_index(inplace=True)
    Final_df_all.rename(columns={"index": "Dict_name"}, inplace=True)
    return Final_df_all

def find_index_significant_LR(final_df,model_type,tolerance = 2, cutoff_freq=10):
    index_list_of_significant_LR = []
    # Analytically define the ACF
    n = 600000
    delta = 600 / 86400
    tt = ut.taus(n, delta)
    for index, item in final_df.iterrows():
        if model_type == 'M1P2':
            covparams = item.loc['η_c':'γ_M2'].values
            acf_true_LR2 = Cov.LR_2(tt, covparams[2:], l_cos=1.9992) \
                         + Cov.LR_2(tt, covparams[5:], l_cos=1.932)   #M2+S2
        elif model_type == 'M1P1':
            covparams = item.loc['η_c':'γ_D2'].values
            acf_true_LR2 = Cov.LR_2(tt, covparams[2:], l_cos=1.962)
        else:
            raise ValueError("model_type must be 'M1P2' or 'M1P1'")
        acf_true_Matern = Cov.Matern(tt, covparams, lmbda=3) 
        # Numerically calculate the spectrum from ACF
        ff_LR2, S_bias_LR2 = gary.bochner(acf_true_LR2, delta, bias=True)
        ff_Matern, S_bias_Matern = gary.bochner(acf_true_Matern, delta, bias=True)
        interpolate_LR2 = interp1d(ff_LR2, S_bias_LR2, bounds_error=False, fill_value="extrapolate")
        interpolate_Matern = interp1d(ff_Matern, S_bias_Matern, bounds_error=False, fill_value="extrapolate")
        S_LR2_at_cutoff = interpolate_LR2(cutoff_freq)
        S_Matern_at_cutoff = interpolate_Matern(cutoff_freq)
        #when matern is 2 time greater than LR
        if S_Matern_at_cutoff >= tolerance * S_LR2_at_cutoff:
            pass
        else:
            index_list_of_significant_LR.append(index)

    return final_df.loc[index_list_of_significant_LR]


def Select_df_from_sites(final_df,site_list,model_type):
    final_df = final_df[final_df['Site'].isin(site_list)].copy()
    #add model type
    final_df['Model_type'] = model_type
    #order the df
    final_df['Site'] = pd.Categorical(final_df['Site'], categories=site_list, ordered=True)
    final_df.sort_values(['Dict_name','Mode'], inplace=True)
    final_df.reset_index(drop = 'True',inplace=True)
    return final_df

# def Clean_up(final_df,model_type = 'M1P1'):
#     #extreme value
#     eliminated_η_c_df  = final_df[(final_df['η_c']<=0.01) | (final_df['η_c']>=1e3)]
#     small_α_c_df  = final_df[final_df['α_c']<=0.5]
#     if model_type == 'M1P2':
#         large_τ_D2_df = final_df[(final_df['τ_M2'] >= 15) & (final_df['τ_S2'] >= 15)]
#         small_τ_D2_df = final_df[(final_df['τ_M2'] <= 1) & (final_df['τ_S2'] <= 1)]
#         eliminated_η_D2_df = final_df[(final_df['η_M2'] <= 0.01) & (final_df['η_S2'] <= 0.01)
#                                       | (final_df['η_M2'] > 50) | (final_df['η_S2'] > 50)]
#     elif model_type == 'M1P1':
#         large_τ_D2_df = final_df[(final_df['τ_D2'] >= 15)]
#         small_τ_D2_df = final_df[(final_df['τ_D2'] <= 1)]
#         eliminated_η_D2_df = final_df[(final_df['η_D2'] <= 0.01) | (final_df['η_D2'] >50)]
#     else:
#         raise ValueError("model_type must be 'M1P2' or 'M1P1'")   
#     #significant LR_D2
#     significant_LR_df = find_index_significant_LR(final_df,model_type,)
#     #concat
#     drop_indices_df = pd.concat([eliminated_η_c_df, 
#                                  large_τ_D2_df, 
#                                  small_τ_D2_df,
#                                  small_α_c_df,
#                                  eliminated_η_D2_df,
#                                  significant_LR_df,]).drop_duplicates()
#     final_df_clean  = final_df.drop(index=drop_indices_df.index)
#     print('{}({}%) of {} fits are removed'.format(len(drop_indices_df),np.round(len(drop_indices_df)/len(final_df)*100,2),len(final_df)))

#     return drop_indices_df,final_df_clean.reset_index(drop=True)

def Clean_up(final_df, model_type='M1P1'):
    # extreme value filters
    eliminated_η_c_df  = final_df[(final_df['η_c'] <= 0.01) | (final_df['η_c'] >= 1e3)] #greater than 0
    small_α_c_df  = final_df[final_df['α_c'] <= 0.5]
    # significant LR_D2
    significant_LR_df = find_index_significant_LR(final_df, model_type)
    if model_type == 'M1P2':
        # examine η
        eliminated_η_D2_df = final_df[((final_df['η_M2'] <= 0.01) & (final_df['η_S2'] <= 0.01))
                                      | (final_df['η_M2'] > 50) | (final_df['η_S2'] > 50)]
         # --- Set insignificant peaks to NaN separately ---
        low_S2_mask = final_df['η_S2'] < 0.01
        final_df.loc[low_S2_mask, ['τ_S2', 'γ_S2']] = np.nan
        low_M2_mask = final_df['η_M2'] < 0.01
        final_df.loc[low_M2_mask, ['τ_M2', 'γ_M2']] = np.nan
        # examine τ
        large_τ_D2_df = final_df[(final_df['τ_M2'] >= 30) | (final_df['τ_S2'] >= 30)]
        small_τ_D2_df = final_df[(final_df['τ_M2'] <= 1) & (final_df['τ_S2'] <= 1)] #cannot be both zero
    elif model_type == 'M1P1':
        large_τ_D2_df = final_df[(final_df['τ_D2'] >= 30)]
        small_τ_D2_df = final_df[(final_df['τ_D2'] <= 1)]
        eliminated_η_D2_df = final_df[(final_df['η_D2'] <= 0.01) | (final_df['η_D2'] > 50)]
    else:
        raise ValueError("model_type must be 'M1P2' or 'M1P1'")   
    # concat all indices to drop
    drop_indices_df = pd.concat([eliminated_η_c_df, 
                                 large_τ_D2_df, 
                                 small_τ_D2_df,
                                 small_α_c_df,
                                 eliminated_η_D2_df,
                                 significant_LR_df]).drop_duplicates()
    # drop them from the original df
    final_df_clean  = final_df.drop(index=drop_indices_df.index)
    print('{} ({:.2f}%) of {} fits are removed'.format(
        len(drop_indices_df),
        len(drop_indices_df)/len(final_df)*100,
        len(final_df)))
    return drop_indices_df, final_df_clean.reset_index(drop=True)


def Calc_Parameter_median_and_std(df, selected_columns, by_mode=False):
    # Select only numerical columns
    numeric_columns = df.select_dtypes(include='number').columns
    
    # Filter selected columns to include only numerical ones
    selected_columns = [col for col in selected_columns if col in numeric_columns]
    
    # Group by 'Site' (and 'Mode' if by_mode is True)
    group_columns = ['Site']
    if by_mode:
        group_columns.append('Mode')
    
    # Compute the median and standard deviation using 'Site' (and 'Mode' if by_mode is True) for grouping
    # skip nan
    median_df = df.groupby(group_columns, observed=False)[selected_columns].median().round(8)
    std_df    = df.groupby(group_columns, observed=False)[selected_columns].std().round(2)

    # Merge the two dataframes on 'Site' (and 'Mode' if by_mode is True) and format values as 'median (std)'
    df_final = median_df.copy()
    for col in selected_columns:
        df_final[col] = median_df[col].astype(str) + " (" + std_df[col].astype(str) + ")"
        
    return df_final

def Compute_timescales(df, model_type):
    delta = 600 / 86400
    N = 600000
    max_threshold = 50.0
    def compute_T(params):
        eta, tau, gamma = params
        if eta < 0.5:
            return np.nan
        tt = ut.taus(N, delta)
        acf = Cov.LR_2_no_cos(tt, params)
        # Calculate the integral timescale
        T = 1./eta**2*np.trapz(acf, tt) 
        return T if T <= max_threshold else np.nan
    if model_type == 'M1P2':
        df[['T_S2', 'T_M2']] = df.apply(lambda row: pd.Series([
                compute_T((row['η_S2'], row['τ_S2'], row['γ_S2'])),
                compute_T((row['η_M2'], row['τ_M2'], row['γ_M2']))]), axis=1)
    elif model_type == 'M1P1':
        df['T_D2'] = df.apply(lambda row: compute_T((row['η_D2'], row['τ_D2'], row['γ_D2'])),
            axis=1)
    return df


def Compute_T_D2_M1P2(df):
    df = df.copy()  # avoid modifying original

    # Weighted average
    df['T_D2'] = (df['T_M2'] * df['η_M2'] + df['T_S2'] * df['η_S2']) / (df['η_M2'] + df['η_S2'])

    # Handle cases where one component is NaN
    df['T_D2'] = df['T_D2'].combine_first(df['T_M2'])
    df['T_D2'] = df['T_D2'].combine_first(df['T_S2'])

    # Print warning if both are NaN
    both_nan = df['T_M2'].isna() & df['T_S2'].isna()
    if both_nan.any():
        print("Rows where both T_M2 and T_S2 are NaN:")
        print(df.loc[both_nan, ['T_M2','T_S2']])
        df.loc[both_nan, 'T_D2'] = np.nan

    return df

def Clean_and_Process_DF(final_df_all,model_type = 'M1P1'):
    drop_df, final_df_clean = Clean_up(final_df_all,model_type = model_type)
    #cal var
    final_df_clean['Subset_Var']     = final_df_clean['HA_var']+final_df_clean['Var_residual']
    final_df_clean['HA_var%']        = final_df_clean['HA_var']/final_df_clean['Subset_Var']*100
    final_df_clean['Var_model_fit%'] = final_df_clean['Var_model_fit']/final_df_clean['Var_residual']*100
    final_df_clean['2α_c']           = 2*final_df_clean['α_c']
    final_df_clean['Mode']           = final_df_clean['Mode']+1
    #cal integral timescale
    final_df_clean = Compute_timescales(final_df_clean, model_type)
    if model_type == 'M1P2':
        final_df_clean = Compute_T_D2_M1P2(final_df_clean)
    return drop_df, final_df_clean


def cal_median_by_mode(final_df):
    median_df = final_df.groupby('Mode').median()
    std_df = final_df.groupby('Mode').std()
    # Combine median and std into one DataFrame
    result_df = median_df.copy()
    # Format the values to show both median and std in the desired format
    for column in median_df.columns:
        result_df[column] = median_df[column].round(2)
    # Display the final result
    return result_df



#from the model parameters from df to compute APE
def Compute_APE(ds_dict,time_dict,amp_dict,df,):
    #constant
    ρ_0 = 1024. 
    APE_dict = {}
    APE_obs_avg_list = []
    APE_NPL_avg_list  = []
    APE_c_avg_list = []
    APE_HA_avg_list = []
    dict_list = df['Dict_name'].drop_duplicates()
    for i in dict_list:
        ds = ds_dict[i].copy()
        ds['modes'] = ds['modes'] +1
        #read stratification
        ϕ  = ds['phi']
        N2 = ds['N2']
        depth_integral = -(N2*ϕ**2).integrate(coord="z_interp")
        #estimate from obs
        time_array =np.stack(time_dict[i])[0]   
        ϵ_A_n_array = xr.DataArray(np.stack(amp_dict[i]).T,
                           dims=["timelong", "modes"],coords={"timelong": time_array, "modes": ds.modes})
        ϵ_A_n_array = ϵ_A_n_array.interp(timelong = ds.timeslow, kwargs={"fill_value": "extrapolate"})
        APE_obs = 0.5*ρ_0* ϵ_A_n_array**2 *depth_integral
        #compute APE from npl η parameters
        df_i = df.query("Dict_name=='{}'".format(i))
        model_type = df_i['Model_type'].iloc[0] if not df_i.empty else None
        if model_type == 'M1P1':
            η_NPL = np.power(df_i['η_D2'].values, 2)
        elif model_type == 'M1P2':
            η_NPL = np.power(df_i['η_S2'].values, 2)+ np.power(df_i['η_M2'].values, 2)
        else:
            η_NPL = np.nan
            print('no such model type')
        mode_number = df_i['Mode'].values    #df has adjusted mode number    
        APE_NPL = 0.5*ρ_0*η_NPL*depth_integral.sel(modes=mode_number) 
        #compute APE from c VAR
        c_var = np.power(df_i['η_c'].values,2)
        APE_c   = 0.5*ρ_0*c_var*depth_integral.sel(modes=mode_number) 
        #compute APE from HA VAR
        HA_var = df_i['HA_M2_var'].values + df_i['HA_S2_var'].values  #just semidiurnal
        APE_HA = 0.5*ρ_0*HA_var*depth_integral.sel(modes=mode_number) 
        APE = xr.Dataset({'APE_obs':APE_obs,'APE_NPL':APE_NPL,'APE_c':APE_c,'APE_HA':APE_HA,})
        # #compute APE over 12h period
        APE_dict[i] = APE #add a negative to correct the sign

        #faltten the lists
        APE_obs_avg = APE_obs.sel(modes=mode_number) #match the length
        APE_obs_avg_list.extend(APE_obs_avg.mean(dim='timeslow').values.tolist())
        APE_NPL_avg_list.extend(APE_NPL.mean(dim='timeslow').values.tolist())
        APE_c_avg_list.extend(APE_c.mean(dim='timeslow').values.tolist())
        APE_HA_avg_list.extend(APE_HA.mean(dim='timeslow').values.tolist())

    return APE_dict,APE_obs_avg_list,APE_NPL_avg_list,APE_c_avg_list,APE_HA_avg_list





#select dict by its keywords
def filter_dict_by_partial_keywords(data_dict, keywords):
    """
    Filter dictionary to include only items with specified keywords in their keys.
    
    Parameters:
        data_dict (dict): Dictionary to filter.
        keywords (list of str): List of keywords to search for in the keys.
    
    Returns:
        dict: Filtered dictionary with keys containing any of the specified keywords.
    """
    return {k: v for k, v in data_dict.items() if any(keyword in k[0] for keyword in keywords)}


# Convert parameters into other types, e.g. temperature
def Convert_to_temp_params(ds_dict,final_df,parameter_name):
    Temp_parameter_dict = {}
    for i in ds_dict:
        dtdz = ds_dict[i]['TEMP'].diff('z_nom')/ds_dict[i]['z_nom'].diff('z_nom')
        phi_interp = ds_dict[i]['phi'].interp(timeslow=ds_dict[i]['TEMP'].time)
        multiplier = phi_interp.interp(z_interp = dtdz ['z_nom'])*dtdz
        selected_df = final_df[final_df['index']==i].copy()
        modes       = selected_df['Mode'].values-1
        for parameter in parameter_name:
            if len(selected_df) !=0:
                Temp_parameter_dict[i, parameter] = (multiplier[:, modes, :]\
                                                      * np.array(selected_df[parameter])[np.newaxis,:, np.newaxis])\
                                                        .sum(dim='modes').median(dim='time')
    return Temp_parameter_dict



def Transfer_to_plot_df(parameter_dict,depth_round_ratio=10):
    data_list = []
    for key, data_array in parameter_dict.items():
        # Get the parameter values and depth (z_nom)
        values = data_array.values
        depths_round = np.round(data_array['z_nom'].values/depth_round_ratio)*depth_round_ratio
        
        #Create a DataFrame with the values and depth for this entry
        df = pd.DataFrame({
            'depths_round': depths_round,
            'parameter': key[1],  # Store the parameter identifier (e.g., 'P1_1'),
            'values': values,     
        })
        # Append to the list
        data_list.append(df)
        
    # Concatenate all the dataframes into a single dataframe for plotting
    plot_df = pd.concat(data_list).reset_index(drop=True)
    return plot_df





