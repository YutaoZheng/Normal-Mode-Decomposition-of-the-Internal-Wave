import xarray as xr
import numpy as np


def Cal_SE(prediction,obs):
    """
    prediiction (xr.dataset)
    Astfalck2023 eq 7
    """
    m = len(prediction.sample)
    μ = prediction.sum(dim='sample')/m
    SE = np.power(μ.values-obs.values,2)
    
    return SE

def Cal_DSS_1D(prediction,obs):
    """
    prediiction (xr.dataset) (1D)
    Astfalck2023 eq 8
    """
    m = len(prediction.sample)
    μ = prediction.sum(dim='sample')/m
    Σ = np.power(prediction-μ,2).sum(dim='sample')/(m-1)
    
    return np.log(Σ.values) + Cal_SE(prediction,obs)/Σ.values

def Cal_ES(prediction,obs):
    """
    prediiction (xr.dataset) (1D)
    Astfalck2023 eq 10
    """
    m = len(prediction.sample)
    #1st term - for accuracy
    first_term = np.sqrt(np.power(prediction.values - obs.values,2)).sum(axis=0)/m
    #2ed term - precision
    sum_list = []
    for t in range(len(prediction.time)):
        k = prediction[:, t].values
        k_sum = np.sum(np.sqrt((k[:, None] - k[None, :]) ** 2))
        sum_list.append(k_sum)
    second_term = np.array(sum_list)/(2*m**2)
    
    return first_term + second_term    

#calulate avg skill score
def Moving_average(arr, sample_freq,window_size,):
    # Calculate the number of points in the window size
    window_size_points = int(window_size / sample_freq)
    # Create a window of ones
    window = np.ones(window_size_points) / window_size_points
    # Apply the moving average
    return np.convolve(arr, window, mode='valid')