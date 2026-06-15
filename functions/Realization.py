import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from . import Processing
from . import Cov
from speccy import sick_tricks as gary
from speccy import utils as ut
import string
import pandas as pd

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


