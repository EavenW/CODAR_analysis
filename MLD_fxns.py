import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
from netCDF4 import Dataset

import cmocean
import pandas as pd
import xarray as xr
import scipy
from scipy.optimize import leastsq
import matplotlib.dates as mdates
import datetime
import dask.array as da
import scipy.ndimage as nd

#### HERE WE GOOOOOO!!!! ####

def coriolis_frequency(lat):
    """ 
    f(latitude in degrees) = 2*omega*sin(lat)
    [f] = sec^(-1)
    """
    omega = 2.0*np.pi/(24.0*60*60)
    return 2.0*omega*np.sin(np.deg2rad(lat))

def inertial_period(lat):
    return 2*np.pi/((60*60)*coriolis_frequency(lat))

def wind_forcing(Wamp_i,Wangle_i,Zo):
    # Return F,G = wind_forcing()
    # [F] = kg / hr2
    # Wamp2 = cm2 / sec2
    Cd = 2e-3 
    rho_a = 1.225 # kg/m3
    rho_w = 1.025e3  # kg/m3
    Num = rho_a*Cd*(Wamp_i**2) # Wamp2 = m / hr2
    Denom = rho_w*Zo
    return (np.sin(Wangle_i) + 1j*np.cos(Wangle_i))*Num/Denom

def medfilt_2D(dataset,i,j,i_err,j_err):
    dataset = dataset[:,i-i_err:i+i_err,j-j_err:j+j_err]
    dataset = np.nanmedian(dataset, axis = 1)
    dataset = np.nanmedian(dataset, axis = 1)
    return dataset

def maxfilt_2D(dataset,i,j,i_err,j_err):
    dataset = dataset[:,i-i_err:i+i_err,j-j_err:j+j_err]
    dataset = np.nanmax(dataset, axis = 1)
    dataset = np.nanmax(dataset, axis = 1)
    return dataset

def wind_forcing(Wamp_i,Wangle_i,Zo):
    # Return F,G = wind_forcing()
    # [F] = kg / hr2
    # Wamp2 = cm2 / sec2
    Cd = 2e-3 
    rho_a = 1.225 # kg/m3
    rho_w = 1.025e3  # kg/m3
    Num = rho_a*Cd*(Wamp_i**2) # Wamp2 = m / hr2
    Denom = rho_w*Zo
    return (np.sin(Wangle_i) + 1j*np.cos(Wangle_i))*Num/Denom

def rk4_Pollard_1d(x, dt, W, Zo, f, c):
    k1 = np.sum(Pollard_1d(x, W, Zo, f, c))*dt
    k2 = np.sum(Pollard_1d((x + k1/2.), W, Zo, f, c))*dt
    k3 = np.sum(Pollard_1d((x + k2/2.), W, Zo, f, c))*dt
    k4 = np.sum(Pollard_1d((x + k3), W, Zo, f, c))*dt
    return (k1 + 2*(k2+k3) + k4)/6.0

def Pollard_1d(V, W, Zo, f, c):
    Wamp = np.sqrt((W.real**2)+(W.imag**2))
    Wangle = np.arctan2(W.imag, W.real)
    Fwind = wind_forcing(Wamp, Wangle, Zo)
    dudt =  -1.0*c*V.real + f*V.imag + Fwind.real
    dvdt =  -1.0*c*V.imag + -1.0*f*V.real + Fwind.imag
    return dudt + 1j*dvdt

def Pollard_2(V, W, Zo, f, c):
    Wamp = np.sqrt((W.real**2)+(W.imag**2))
    Wangle = np.arctan2(W.imag, W.real)
    Fwind = wind_forcing(Wamp, Wangle, Zo)
    #V = V*c
    dudt = -(1./c)*V.real + 1.0*f*V.imag + Fwind.real
    dvdt = -(1./c)*V.imag - 1.0*f*V.real + Fwind.imag
    return np.array(dudt + 1j*dvdt)
     
def rk4_Pollard_2(x, dt, W, Zo, f, c):
    k1 = Pollard_1d(x, W, Zo, f, c)*dt
    k2 = Pollard_1d((x + k1/2.), W, Zo, f, c)*dt
    k3 = Pollard_1d((x + k2/2.), W, Zo, f, c)*dt
    k4 = Pollard_1d((x + k3), W, Zo, f, c)*dt
    return (k1 + 2*(k2+k3) + k4)/6.0

def lanczos(per,window_width):
    w_vec = np.linspace(1,window_width,window_width)
    R_f = np.linspace(1,(2*window_width)+1,(2*window_width)+1) * np.nan
    n_s = np.linspace(-1*window_width,window_width,(2*window_width)+1) 
    H0 = (1./per) * 2.0
    omt = w_vec * ((2.0*np.pi)/per)
    omc = (w_vec * 2. * np.pi)/ ((2.*window_width)+1.)
    w_k = ((H0*np.sin(omt))/(omt)) * (np.sin((omc))/omc)
    R_f_tmp = H0 + (2. * np.sum(w_k))
    R_f_tmp_2 = (w_k/R_f_tmp)
    R_f[0:window_width] = R_f_tmp_2[::-1,...]
    R_f[window_width] = H0 / R_f_tmp
    R_f[window_width+1:(2*window_width)+1] = R_f_tmp_2
    return  R_f

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    from scipy.signal import firwin
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def bandpass_fir_CODAR(variable, times, low_period, high_period):
    dt = times[1]-times[0]
    lowcut = 1./high_period
    highcut = 1./low_period
    taps_window = int(high_period*10)
    taps_hamming = bandpass_firwin(taps_window, lowcut, highcut, fs=1.0/dt)
    var_ham = scipy.signal.convolve(variable,taps_hamming,'same')
    return var_ham

def wind_forcing_2d(Wind_u,Wind_v,Zo):
    # Return F,G = wind_forcing()
    # [F] = kg / hr2
    # Wamp2 = cm2 / sec2
    Cd = 2e-3 
    rho_a = 1.225 # kg/m3
    rho_w = 1.025e3  # kg/m3
    Num = rho_a*Cd*(Wind_u**2+Wind_v**2) # Wamp2 = m / hr2
    Denom = rho_w*Zo
    theta = np.arctan2(Wind_v,Wind_u)
    return np.sin(theta)*Num/Denom + 1j*np.cos(theta)*Num/Denom

def slab_ocean_response(z, W, time, f, c):
    length = time.shape[0]
    Vi = 0.001 + 1j*(0.001)
    dt = time[1]-time[0]
    try:
        del data[:]
    except:
        data = np.ones(length,dtype='complex')
    data[0] = Vi
    for i in range(length-1):
        dVR = np.sum(rk4_Pollard_1d(data[i],dt,W[i],Zo=10,f=f,c=c));
        Vn  = data[i] + dVR # #rk4[i]
        data[i+1] = Vn
    #v_in = bandpass_fir_CODAR(V.imag, time, 18.2, 19.0)
    #u_in = bandpass_fir_CODAR(V.real, time, 18.2, 19.0)
    #Vamp_in = np.abs(data)
    return data