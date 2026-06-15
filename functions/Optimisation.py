import numpy as np
from speccy import utils as ut
from speccy import sick_tricks as gary
from scipy.optimize import minimize
from . import Processing

def whittle_function(params,n,delta,subset,I_ω,
                     covfunct):
    """
    Compute the Whittle likelihood function.

    Parameters:
        params (list or array): Model parameters.
        n: length of the time series
        delta: sampling internal
        index: index for selected frequency range
        I_ω: positive periodogoram
        covfunct: kernel for acf
        covfunct_spectra: kernel for spectra

    Returns:s
        float: Negative log-likelihood value.
    """
    tt = ut.taus(n, delta)
    I_ω_subset = I_ω[subset]
    #Nnumerical
    acf = covfunct(tt,params)                              #analytic acf
    ff,P_hat = gary.bochner(acf, delta=delta, bias = True) #numerical spectrum
    P_hat = P_hat[ff>=0]   
    P_hat = P_hat[subset]
    whittle = -np.sum(np.log(P_hat) + I_ω_subset/P_hat)  
    return -whittle                                #negative whittle likelihood

def logistic(x, low, high):
    """Logistic transformation."""
    return low + (high - low) / (1 + np.exp(-x))
def inverse_logistic(y, low, high):
    """Inverse logistic transformation."""
    return np.log((y - low) / (high - y))

def transform_params(params_log, bound_indices=None, bounds_list=None):
    """
    Transform parameters from log space to their appropriate scale.
    - Exponential transformation for unbounded parameters.
    - Logistic transformation for bounded parameters.
    """
    params = np.exp(params_log)
    if bound_indices and bounds_list:
        for idx, (low, high) in zip(bound_indices, bounds_list):
            params[idx] = logistic(params_log[idx], low, high)
    return params
def inverse_transform_params(params, bound_indices=None, bounds_list=None):
    """
    Inverse transform parameters from their appropriate scale back to log space.
    - Logarithmic transformation for unbounded parameters.
    - Inverse logistic transformation for bounded parameters.
    """
    params_log = np.log(params)
    if bound_indices and bounds_list:
        for idx, (low, high) in zip(bound_indices, bounds_list):
            params_log[idx] = inverse_logistic(params[idx], low, high)
    return params_log

# def transform_params(params_latent, bound_indices=None, bounds_list=None):
#     """
#     Transform parameters from unconstrained latent space to physical space.
#     - exp() for positive unbounded parameters
#     - logistic() for bounded parameters
#     """
#     params = np.zeros_like(params_latent)
#     bound_indices = bound_indices or []
#     bounds_dict = {idx: (low, high) for idx, (low, high) in zip(bound_indices, bounds_list or [])}
#     for i, x in enumerate(params_latent):
#         if i in bounds_dict:
#             low, high = bounds_dict[i]
#             params[i] = logistic(x, low, high)
#         else:
#             params[i] = np.exp(x)
#     return params

# def inverse_transform_params(params, bound_indices=None, bounds_list=None):
#     """
#     Transform parameters from physical space back to unconstrained latent space.
#     - log() for positive unbounded parameters
#     - inverse_logistic() for bounded parameters
#     """
#     params_latent = np.zeros_like(params)
#     bound_indices = bound_indices or []
#     bounds_dict = {idx: (low, high) for idx, (low, high) in zip(bound_indices, bounds_list or [])}
#     for i, y in enumerate(params):
#         if i in bounds_dict:
#             low, high = bounds_dict[i]
#             params_latent[i] = inverse_logistic(y, low, high)
#         else:
#             params_latent[i] = np.log(y)
#     return params_latent

def whittle_fitting(params_ic, n, delta, subset, I_ω, covfunct, 
                    bound_indices=None, bounds_list=None):
    """
    Minimize the Whittle likelihood function with mixed transformations:
    - Logistic for specified parameters with bounds.
    - Logarithmic for other parameters.
    Parameters:
        params_ic (list or array): Initial guess for model parameters.
        n (int): Length of the time series.
        delta (float): Sampling interval.
        subset (array-like): Index for selected frequency range.
        I_ω (array-like): Periodogram.
        covfunct (callable): Kernel for the autocorrelation function (ACF).
        bound_indices (list of int, optional): Indices of parameters to constrain with bounds.
        bounds_list (list of tuple, optional): List of (lower, upper) bounds for each bounded parameter. 
    Returns:
        f (array): Frequencies of the fitted model spectrum.
        P (array): Power spectral density of the fitted model.
        soln (OptimizeResult): Optimization result object.
    """
    method = 'Nelder-Mead'
    myargs = (n, delta, subset, I_ω, covfunct)

    # Transform initial guess
    params_log_ic = inverse_transform_params(params_ic, 
                                             bound_indices=bound_indices, 
                                             bounds_list=bounds_list)

    # Wrapper for the Whittle likelihood with mixed transformations
    def whittle_function_mixed(params_log, *args):
        # Transform parameters to the original scale
        params = transform_params(params_log.copy(), 
                                  bound_indices=bound_indices, 
                                  bounds_list=bounds_list)
        return whittle_function(params, *args)

    # Minimize the Whittle likelihood
    soln_whittle = minimize(
        whittle_function_mixed,
        x0=params_log_ic,
        args=myargs,
        method=method,
        options={'maxiter': 10000})

    # Transform best solution back to the original scale
    params = transform_params(soln_whittle['x'], 
                              bound_indices=bound_indices, 
                              bounds_list=bounds_list)

    # Numerical ACF and model spectrum
    tt = ut.taus(n, delta)
    acf = covfunct(tt, params)
    f_model, P_model = gary.bochner(acf, delta=delta, bias=True)
    P_model = P_model[f_model >= 0]
    f_model = f_model[f_model >= 0]

    return f_model[subset], P_model[subset], soln_whittle


def Model_fit(P_ϵ_list, Time_obs_list, subset_list,
              inital_guess,kernel, bound_indices=None, bounds_list=None):
    
    F_model_fit_list =[]
    P_model_fit_list = []
    Soln_model_fit_list = []
    Whittle_value_list = []
    
    for i in range(len(P_ϵ_list)):
        Time = Time_obs_list[i]
        Subset = subset_list[i]
        P_ϵ = P_ϵ_list[i]
        time_length = len(Time)
        delta_days = (Time[1]-Time[0]).astype('float')/1e9/86400
        delta_days = delta_days.values
        f_model_fit, p_model_fit, soln_model_fit = whittle_fitting(inital_guess,time_length,delta_days,                  
                                                                    Subset,P_ϵ, kernel,
                                                                    bound_indices=bound_indices, bounds_list=bounds_list)
        if soln_model_fit['success']:            
            F_model_fit_list.append(f_model_fit)
            P_model_fit_list.append(p_model_fit)
            Soln_model_fit_list.append(transform_params(soln_model_fit['x'], 
                              bound_indices=bound_indices, 
                              bounds_list=bounds_list))
            Whittle_value_list.append(soln_model_fit['fun'])
        else:
            Soln_model_fit_list.append(len(inital_guess)*[np.nan])
            print('Unsuccessful at',soln_model_fit['message'])

    return F_model_fit_list, P_model_fit_list, Soln_model_fit_list, Whittle_value_list









