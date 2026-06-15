import xarray as xr
import numpy as np
from gptide import GPtideScipy
import numpy as np
from . import Processing
from . import Cov
from speccy import sick_tricks as gary
from speccy import utils as ut
import string
import pandas as pd

def Sampling(covparams,sample_length,sample_freq,mean_func,mean_params,cov_model,sample_number=1,noise=0):
    xd = np.arange(0,sample_length*sample_freq,sample_freq)[:,None]
    print('the sampling duration is {} days'.format(sample_length*sample_freq))
    GP  = GPtideScipy(xd, xd, noise, cov_model, covparams,mean_func = mean_func,mean_params = mean_params)
    GP_samples = GP.prior(samples=sample_number)
    GP_samples_array = xr.DataArray(GP_samples.T,
                                  dims=('sample','time'),
                                  coords={"sample": np.arange(sample_number), "time": xd.flatten()*86400}) #time in second
    GP_samples_dataset = xr.Dataset({"Amplitude": GP_samples_array})
    
    return GP_samples_dataset



def Predicting(obs_x, obs_y,
               predict_x,covparams,cov_model,
               mean_func,mean_params,sample_number=1,noise=0):
    
    print('the prediction is for {} days'.format(predict_x[-1]-obs_x[-1]))
    OI = GPtideScipy(obs_x[:,None], predict_x[:,None], noise, cov_model, covparams,mean_func = mean_func,mean_params = mean_params)
    y_predict = OI.conditional(obs_y.values[:,None],samples=sample_number)
    GP_predict_array = xr.DataArray(y_predict.T,
                                  dims=('sample','time'),
                                  coords={"sample": np.arange(sample_number), "time": predict_x}) #time in second
    GP_predict_dataset = xr.Dataset({"Amplitude": GP_predict_array})
    
    return GP_predict_dataset

def Predict_set_up(A_obs,obs_start,obs_len,predict_len,freq_scale):
    #select a period of obs
    obs_freq = (A_obs.time[1]-A_obs.time[0]).values.astype('float')/1e9/86400 #in days
    x_obs = np.arange(0,obs_len*obs_freq,obs_freq)
    y_obs = A_obs[obs_start:obs_start+obs_len]
    #set up prediction length
    predict_freq = obs_freq*freq_scale
    print('prediction interva is {}s'.format(predict_freq*86400)) #in seconds
    x_predict = np.arange(0,predict_len*predict_freq,predict_freq)
    #set up the true
    x_true = np.arange(obs_len*obs_freq,predict_len*predict_freq,predict_freq)
    y_true = A_obs[obs_start+obs_len:obs_start+int(predict_len*predict_freq/obs_freq)][::int(predict_freq/obs_freq)]
    return x_obs,y_obs,x_predict,x_true,y_true



def M1P1(x,xpr,params):
    D2_freq = 1.93 #cpd
    M2_freq = 2 #cpd
    D2_freq = (D2_freq+M2_freq)/2
    
    η_matern1 = params[0]
    α_matern1 = params[1]
    eta_D2    = params[2]
    tau_D2    = params[3]
    gamma_D2  = params[4]
    
    dx    = np.sqrt((x-xpr)*(x-xpr))
    #background energy continuum  
    matern1 = Cov.Matern(dx, (η_matern1,α_matern1),lmbda=3,sigma=1e-6)   
    #peak
    peak2   = Cov.LR_2(dx, (eta_D2,tau_D2,gamma_D2),l_cos=D2_freq) 
    COV = matern1 + peak2 #+ noise
    return COV


def M1P2_2(x,xpr,params):
    S2_freq = 1.93 #cpd
    M2_freq = 2 #cpd

    η_matern1 = params[0]
    α_matern1 = params[1]
    eta_S2    = params[2]
    tau_S2    = params[3]
    gamma_S2  = params[4]
    eta_M2    = params[5]
    tau_M2    = params[6]
    gamma_M2  = params[7]
    
    dx    = np.sqrt((x-xpr)*(x-xpr))
    matern1 = Cov.Matern(dx, (η_matern1,α_matern1),lmbda=3,sigma=1e-6)              #background energy continuum  
    peak2   = Cov.LR_2(dx, (eta_S2,tau_S2,gamma_S2),l_cos=S2_freq) + \
              Cov.LR_2(dx, (eta_M2,tau_M2,gamma_M2),l_cos=M2_freq)
    COV = matern1 + peak2 #+ noise
    return COV




