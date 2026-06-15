#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import pandas as pd
from . import Processing
from speccy import sick_tricks as gary

def Mean_X(x):
    """
    the mean function for harmonic analysis 
    composing of 4 astronomical tidal harmonics
    input: a time-series np array 
    """
    p1 = 0.93  #cpd  Principal lunar diurnal       O1
    p2 = 1  #cpd, Lunisolar diurnal             K1
    p3 = 1.93  #cpd  Principal lunar semidiurnal   M2
    p4 = 2  #cpd, Principal solar semidiurnal   S2
    
    time_values = x 
    angular_frequencies = 2 * np.pi * np.array([p1, p2, p3, p4])
    cosine_constants = np.cos(angular_frequencies[:,None] * time_values)
    sine_constants = np.sin(angular_frequencies[:,None] * time_values)
    
    # Create X1 DataArray
    X1 = xr.DataArray(
        np.vstack((cosine_constants, sine_constants)).T,
        dims=('time', 'parameters'))
    
    return X1.values  

def OLS(X,y):
    """
    Ordinary least squares
    input: x,y
    output: the parameters
    """
    inverse = np.linalg.inv(np.transpose(X)@X)
    β = inverse@np.transpose(X)@y
    return β

def Prior_mean_function(x,params):
    a1,a2,a3,a4, b1,b2,b3,b4= params
    p1 = 0.93    #cpd  Principal lunar diurnal       O1
    p2 = 1       #cpd, Lunisolar diurnal             K1 
    p3 = 1.93    #cpd  Principal lunar semidiurnal   M2
    p4 = 2       #cpd, Principal solar semidiurnal   S2 
    peak1 = a1*np.cos(2*np.pi*p1*x)+b1*np.sin(2*np.pi*p1*x)
    peak2 = a2*np.cos(2*np.pi*p2*x)+b2*np.sin(2*np.pi*p2*x)
    peak3 = a3*np.cos(2*np.pi*p3*x)+b3*np.sin(2*np.pi*p3*x)
    peak4 = a4*np.cos(2*np.pi*p4*x)+b4*np.sin(2*np.pi*p4*x)
    return peak1+peak2+peak3+peak4

def Cal_HA(time_list,obs_list):
    Mean_params_list = []
    Yd_mean_list     = []
    ϵ_list           = []
    F_ϵ_list   = []
    Puu_ϵ_list = []
    
    for order,obs in enumerate(obs_list):
        t = time_list[order]
        y = obs.values
        N = len(y)
        Δ = (t[1]- t[0]).astype('float')/1e9/86400 #in day
        Δ = Δ.values
        x  = np.linspace(0,Δ*N,N)
        Xd = Mean_X(x)
        mean_params = OLS(Xd,y)
        yd_mean = Prior_mean_function(x,mean_params)
        ϵ =  y - yd_mean #residual
        
        F_ϵ,Puu_ϵ = Processing.Cal_periodogram(ϵ,Δ)
        #append results
        Mean_params_list.append(mean_params)
        Yd_mean_list.append(yd_mean)    
        ϵ_list.append(ϵ)
        F_ϵ_list.append(F_ϵ)
        Puu_ϵ_list.append(Puu_ϵ)
        
    return Mean_params_list, Yd_mean_list,ϵ_list,F_ϵ_list,Puu_ϵ_list

def Cal_var_from_mean_params(mean_params_list):
    '''
    output: phaselocked variability
    '''
    HA_var_list = []
    for mean_params in mean_params_list:
        #HA_var.append(np.sqrt(np.sum(mean_param**2))/2)
        HA_var = np.sum(mean_params**2)/2
        HA_var_list.append(HA_var) 
    return HA_var_list


def Cal_var_from_mean_funcs(mean_funcs_list):
    '''
    output: phaselocked variability
    '''
    HA_var_list = []
    for mean_funcs in mean_funcs_list:
        HA_var_list.append(mean_funcs.var()) 
    return HA_var_list






