## SCRIPT 1: DONE!!!!! :-)
## EXTRACT USABLE SAILDRONE DATA
## Subhatra Sivam, Eli Lichtblau
print('EXTRACT USABLE SAILDRONE DATA')

## SAILDRONE INPUT VARIABLES
## OBS: time (seconds), longitude (degrees), latitude (degrees)
## RH_MEAN (%), TEMP_AIR_MEAN (C), TEMP_CTD_RBR_MEAN (C)
## wind_speed (m/s), BARO_PRES_MEAN (hPa)
## FLUX: time (seconds), longitude (degrees), latitude (degrees)
## QL (W/m^2), QS (W/m^2)

# import packages
import netCDF4 as nc
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
from typing import Tuple

# functions
def solution(X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Params:
        X1: (1D array_like)
        X2: (1D array_like)
    Returns:
        X1_indices where value exists in X2 as well
        X2_indices where value exists in X1 as well
    Note: the returned indices array are ordered smallest to greatest. by the value they correspond to
    that is to say X1[X1_indices] is a sorted list, u could do X1[X1_indices.sort()] to get the values in 
    the order they appear in the orignal X1
    
    """
    inter = np.intersect1d(X1, X2)
    def helper(inter: np.ndarray, x: np.ndarray):
        sorter = np.argsort(x)
        searchsorted_left = np.searchsorted(x, inter, sorter=sorter,side='left')
        searchsorted_right = np.searchsorted(x, inter, sorter=sorter,side='right')
        values = vrange(searchsorted_left, searchsorted_right) 
        return sorter[values] # optional to sort this if u care?
        

    return helper(inter, X1), helper(inter, X2)
def vrange(starts: np.ndarray, stops: np.ndarray):
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

### 2017 ###
print('start 2017...')

## observations - 1 file, no individual saildrone files. ##
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2017/2017obs/sd_tot.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
time: np.ndarray = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat: np.ndarray  = f_obs.variables['latitude'][:] # degrees
lon: np.ndarray  = f_obs.variables['longitude'][:] # degrees
rh: np.ndarray  =  f_obs.variables['RH_MEAN'][:] # %
airtemp: np.ndarray  = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp: np.ndarray  = f_obs.variables['TEMP_CTD_MEAN'][:] # C
wind: np.ndarray  = f_obs.variables['wind_speed'][:] # m/s^2
sp: np.ndarray  = f_obs.variables['BARO_PRES_MEAN'][:] # hPa
bundle = np.stack((time,lat,lon,rh,airtemp,skntemp,wind,sp),axis=1)
bundle = np.ma.masked_where(bundle == 9.969209968386869E36, bundle)
bundle = np.ma.compress_rows(bundle)

# remove duplicate rows
[_, index] = np.unique(bundle, axis=0, return_index=True)
index.sort() # sort by index so we get in same order
bundle = bundle[index] # np unqiue sorts the values whihc we don't want

# remove duplicate times, where time[i]==time[i+1]
time = bundle[:, 0]
time_repeats = np.argwhere(time[1:]==time[:-1])
bundle = np.delete(bundle, time_repeats, axis=0)
time = bundle[:,0]
lat = bundle[:,1]
lon = bundle[:,2]
rh = bundle[:,3]
airtemp = bundle[:,4] 
skntemp = bundle[:,5] 
wind = bundle[:,6] 
sp = bundle[:,7]

# split saildrone data
timestep = np.arange(0,len(time)+1,1)
min_ind = [0]
localmin = argrelextrema(time, np.less)
for min in localmin[0]:
    min_ind.append(min)
max_ind = []
localmax = argrelextrema(time, np.greater)
for max in localmax[0]:
    max_ind.append(max+1)
max_ind.append(timestep[-1])
spl_time_sd = np.split(time,max_ind)
spl_lon_sd = np.split(lon,max_ind)
spl_lat_sd = np.split(lat,max_ind)
spl_rh_sd = np.split(rh,max_ind)
spl_airtemp_sd = np.split(airtemp,max_ind)
spl_skntemp_sd = np.split(skntemp,max_ind)
spl_wind_sd = np.split(wind,max_ind)
spl_sp_sd = np.split(sp,max_ind)

## flux ##
# 1001 #
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2017/2017calc/sd1001.nc'
f_calc = nc.Dataset(filename_calc,mode='r')
time = f_calc.variables['time'][:] 
lat = f_calc.variables['latitude'][:] 
lon = f_calc.variables['longitude'][:] 
eflux = f_calc.variables['QL'][:] 
hflux = f_calc.variables['QS'][:]

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# intersect with time and location
[idx_obs,idx_calc] = solution(spl_time_sd[0],time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(spl_rh_sd[0][idx_obs]))
airtemp = np.array(np.copy(spl_airtemp_sd[0][idx_obs]))
skntemp = np.array(np.copy(spl_skntemp_sd[0][idx_obs]))
wind = np.array(np.copy(spl_wind_sd[0][idx_obs]))
sp = np.array(np.copy(spl_sp_sd[0][idx_obs]))
print('start averaging: SD2017: 1001')

# convert time
yeardiff = 414888 
hours = time/60/60 - yeardiff #hours since 05/01/2017 00:00

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables ##
hours_sd2017 = dff['hours_sd_rounded'].values
eflux_sd2017 = dff['eflux'].values
hflux_sd2017 = dff['hflux'].values
rh_sd2017 = dff['rh'].values
airtemp_sd2017 = dff['airtemp'].values
skntemp_sd2017 = dff['skntemp'].values
sp_sd2017 = dff['sp'].values
wind_sd2017 = dff['wind'].values
roundedhour_sd2017 = hours[preciseidx]
indlat_sd2017 = lat[preciseidx]
indlon_sd2017 = lon[preciseidx]
id_sd2017 = np.full(np.shape(hours_sd2017), 1001)

print('done with SD2017: 1001')

# 1002 #
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2017/2017calc/sd1002.nc'
f_calc = nc.Dataset(filename_calc,mode='r')
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# intersect with time and location
[idx_obs,idx_calc] = solution(spl_time_sd[1],time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(spl_rh_sd[1][idx_obs]))
airtemp = np.array(np.copy(spl_airtemp_sd[1][idx_obs]))
skntemp = np.array(np.copy(spl_skntemp_sd[1][idx_obs]))
wind = np.array(np.copy(spl_wind_sd[1][idx_obs]))
sp = np.array(np.copy(spl_sp_sd[1][idx_obs]))
print('start averaging: SD2017: 1002')

# convert time
hours = time/60/60 - yeardiff #hours since 05/01/2017 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables ##
hours_sd2017 = np.concatenate((hours_sd2017,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2017 = np.concatenate((eflux_sd2017,dff['eflux'].values),axis=None)
hflux_sd2017 = np.concatenate((hflux_sd2017,dff['hflux'].values),axis=None)
rh_sd2017 = np.concatenate((rh_sd2017,dff['rh'].values),axis=None)
airtemp_sd2017 = np.concatenate((airtemp_sd2017,dff['airtemp'].values),axis=None)
skntemp_sd2017 = np.concatenate((skntemp_sd2017,dff['skntemp'].values),axis=None)
sp_sd2017 = np.concatenate((sp_sd2017,dff['sp'].values),axis=None)
wind_sd2017 = np.concatenate((wind_sd2017,dff['wind'].values),axis=None)
roundedhour_sd2017 = np.concatenate((roundedhour_sd2017,hours[preciseidx]),axis=None)
indlat_sd2017 =  np.concatenate((indlat_sd2017,lat[preciseidx]),axis=None)
indlon_sd2017 =  np.concatenate((indlon_sd2017,lon[preciseidx]),axis=None)
id_sd2017 = np.concatenate((id_sd2017,np.full(np.shape(dff['hours_sd_rounded'].values), 1002)))

print('done with SD2017: 1002')

# 1003 #
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2017/2017calc/sd1003.nc'
f_calc = nc.Dataset(filename_calc,mode='r')
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# intersect with time and location
[idx_obs,idx_calc] = solution(spl_time_sd[2],time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(spl_rh_sd[2][idx_obs]))
airtemp = np.array(np.copy(spl_airtemp_sd[2][idx_obs]))
skntemp = np.array(np.copy(spl_skntemp_sd[2][idx_obs]))
wind = np.array(np.copy(spl_wind_sd[2][idx_obs]))
sp = np.array(np.copy(spl_sp_sd[2][idx_obs]))
print('start averaging: SD2017: 1003')

# convert time
hours = time/60/60 - yeardiff #hours since 05/01/2017 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

hours_sd2017 = np.concatenate((hours_sd2017,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2017 = np.concatenate((eflux_sd2017,dff['eflux'].values),axis=None)
hflux_sd2017 = np.concatenate((hflux_sd2017,dff['hflux'].values),axis=None)
rh_sd2017 = np.concatenate((rh_sd2017,dff['rh'].values),axis=None)
airtemp_sd2017 = np.concatenate((airtemp_sd2017,dff['airtemp'].values),axis=None)
skntemp_sd2017 = np.concatenate((skntemp_sd2017,dff['skntemp'].values),axis=None)
sp_sd2017 = np.concatenate((sp_sd2017,dff['sp'].values),axis=None)
wind_sd2017 = np.concatenate((wind_sd2017,dff['wind'].values),axis=None)
roundedhour_sd2017 = np.concatenate((roundedhour_sd2017,hours[preciseidx]),axis=None)
indlat_sd2017 =  np.concatenate((indlat_sd2017,lat[preciseidx]),axis=None)
indlon_sd2017 =  np.concatenate((indlon_sd2017,lon[preciseidx]),axis=None)
id_sd2017 = np.concatenate((id_sd2017,np.full(np.shape(dff['hours_sd_rounded'].values), 1003)))

print('done with SD2017: 1003')

# conversions for sea
essea_sd = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(skntemp_sd2017+273.15)))) # sea saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
esea_sd = essea_sd * (rh_sd2017/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qskn = 0.622*(esea_sd/(sp_sd2017-esea_sd))*1000 # specific humidity of sea surface : Wallace and Hobbs (pg. 80)

# conversions for air
esair_sd = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(airtemp_sd2017+273.15)))) # air saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
eair_sd = esair_sd * (rh_sd2017/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qair = 0.622*(eair_sd/(sp_sd2017-eair_sd))*1000 # specific humidity of atmosphere : Wallace and Hobbs (pg. 80)

# export file
for_df = np.array([id_sd2017,hours_sd2017,indlat_sd2017, indlon_sd2017,eflux_sd2017,hflux_sd2017,qair,qskn,airtemp_sd2017,skntemp_sd2017,wind_sd2017,sp_sd2017])
df = pd.DataFrame(for_df.T, columns=['id','hours','lat','lon','eflux','hflux','qair','qskn','airtemp','skntemp','wind','sp'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2017E5avghr.csv')
print('...end 2017.')
###################################################################################################
### 2018 ###
print('start 2018...')

## 1020
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018obs/sd1020.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018calc/sd1020.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2018: 1020')

# average by time
yeardiff = 423648
hours = time/60/60 - yeardiff #hours since 05/01/2018 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables ##
hours_sd2018 = dff['hours_sd_rounded'].values
eflux_sd2018 = dff['eflux'].values
hflux_sd2018 = dff['hflux'].values
rh_sd2018 = dff['rh'].values
airtemp_sd2018 = dff['airtemp'].values
skntemp_sd2018 = dff['skntemp'].values
sp_sd2018 = dff['sp'].values
wind_sd2018 = dff['wind'].values
roundedhour_sd2018 = hours[preciseidx]
indlat_sd2018 = lat[preciseidx]
indlon_sd2018 = lon[preciseidx]
id_sd2018 = np.full(np.shape(hours_sd2018), 1020)

print('done with SD2018: 1020')

## 1021
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018obs/sd1021.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018calc/sd1021.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2018: 1021')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2018 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2018 = np.concatenate((hours_sd2018,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2018 = np.concatenate((eflux_sd2018,dff['eflux'].values),axis=None)
hflux_sd2018 = np.concatenate((hflux_sd2018,dff['hflux'].values),axis=None)
rh_sd2018 = np.concatenate((rh_sd2018,dff['rh'].values),axis=None)
airtemp_sd2018 = np.concatenate((airtemp_sd2018,dff['airtemp'].values),axis=None)
skntemp_sd2018 = np.concatenate((skntemp_sd2018,dff['skntemp'].values),axis=None)
sp_sd2018 = np.concatenate((sp_sd2018,dff['sp'].values),axis=None)
wind_sd2018 = np.concatenate((wind_sd2018,dff['wind'].values),axis=None)
roundedhour_sd2018 = np.concatenate((roundedhour_sd2018,hours[preciseidx]),axis=None)
indlat_sd2018 =  np.concatenate((indlat_sd2018,lat[preciseidx]),axis=None)
indlon_sd2018 =  np.concatenate((indlon_sd2018,lon[preciseidx]),axis=None)
id_sd2018 = np.concatenate((id_sd2018,np.full(np.shape(dff['hours_sd_rounded'].values), 1021)))

print('done with SD2018: 1021')

## 1022
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018obs/sd1022.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018calc/sd1022.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2018: 1022')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2018 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2018 = np.concatenate((hours_sd2018,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2018 = np.concatenate((eflux_sd2018,dff['eflux'].values),axis=None)
hflux_sd2018 = np.concatenate((hflux_sd2018,dff['hflux'].values),axis=None)
rh_sd2018 = np.concatenate((rh_sd2018,dff['rh'].values),axis=None)
airtemp_sd2018 = np.concatenate((airtemp_sd2018,dff['airtemp'].values),axis=None)
skntemp_sd2018 = np.concatenate((skntemp_sd2018,dff['skntemp'].values),axis=None)
sp_sd2018 = np.concatenate((sp_sd2018,dff['sp'].values),axis=None)
wind_sd2018 = np.concatenate((wind_sd2018,dff['wind'].values),axis=None)
roundedhour_sd2018 = np.concatenate((roundedhour_sd2018,hours[preciseidx]),axis=None)
indlat_sd2018 =  np.concatenate((indlat_sd2018,lat[preciseidx]),axis=None)
indlon_sd2018 =  np.concatenate((indlon_sd2018,lon[preciseidx]),axis=None)
id_sd2018 = np.concatenate((id_sd2018,np.full(np.shape(dff['hours_sd_rounded'].values), 1022)))

print('done with SD2018: 1022')

## 1023
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018obs/sd1023.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2018/2018calc/sd1023.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2018: 1023')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2018 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2018 = np.concatenate((hours_sd2018,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2018 = np.concatenate((eflux_sd2018,dff['eflux'].values),axis=None)
hflux_sd2018 = np.concatenate((hflux_sd2018,dff['hflux'].values),axis=None)
rh_sd2018 = np.concatenate((rh_sd2018,dff['rh'].values),axis=None)
airtemp_sd2018 = np.concatenate((airtemp_sd2018,dff['airtemp'].values),axis=None)
skntemp_sd2018 = np.concatenate((skntemp_sd2018,dff['skntemp'].values),axis=None)
sp_sd2018 = np.concatenate((sp_sd2018,dff['sp'].values),axis=None)
wind_sd2018 = np.concatenate((wind_sd2018,dff['wind'].values),axis=None)
roundedhour_sd2018 = np.concatenate((roundedhour_sd2018,hours[preciseidx]),axis=None)
indlat_sd2018 =  np.concatenate((indlat_sd2018,lat[preciseidx]),axis=None)
indlon_sd2018 =  np.concatenate((indlon_sd2018,lon[preciseidx]),axis=None)
id_sd2018 = np.concatenate((id_sd2018,np.full(np.shape(dff['hours_sd_rounded'].values), 1023)))

print('done with SD2018: 1023')

# conversions for sea
essea_sd = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(skntemp_sd2018+273.15)))) # sea saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
esea_sd = essea_sd * (rh_sd2018/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qskn = 0.622*(esea_sd/(sp_sd2018-esea_sd))*1000 # specific humidity of sea surface : Wallace and Hobbs (pg. 80)

# conversions for air
esair_sd = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(airtemp_sd2018+273.15)))) # air saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
eair_sd = esair_sd * (rh_sd2018/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qair = 0.622*(eair_sd/(sp_sd2018-eair_sd))*1000 # specific humidity of atmosphere : Wallace and Hobbs (pg. 80)

# export file
for_df = np.array([id_sd2018,hours_sd2018,indlat_sd2018, indlon_sd2018,eflux_sd2018,hflux_sd2018,qair,qskn,airtemp_sd2018,skntemp_sd2018,wind_sd2018,sp_sd2018])
df = pd.DataFrame(for_df.T, columns=['id','hours','lat','lon','eflux','hflux','qair','qskn','airtemp','skntemp','wind','sp'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2018E5avghr.csv')
print('...end 2018.')
###################################################################################################
### 2019 ###
print('start 2019...')

## 1033
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019obs/sd1033.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019calc/sd1033.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_RBR_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat_calc[idx_calc]))
lon = np.array(np.copy(lon_calc[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2019: 1033')

# average by time
yeardiff = 432408
hours = time/60/60 - yeardiff #hours since 05/01/2019 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables ##
hours_sd2019 = dff['hours_sd_rounded'].values
eflux_sd2019 = dff['eflux'].values
hflux_sd2019 = dff['hflux'].values
rh_sd2019 = dff['rh'].values
airtemp_sd2019 = dff['airtemp'].values
skntemp_sd2019 = dff['skntemp'].values
sp_sd2019 = dff['sp'].values
wind_sd2019 = dff['wind'].values
roundedhour_sd2019 = hours[preciseidx]
indlat_sd2019 = lat[preciseidx]
indlon_sd2019 = lon[preciseidx]
id_sd2019 = np.full(np.shape(hours_sd2019), 1033)

print('done with SD2019: 1033')

## 1034
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019obs/sd1034.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019calc/sd1034.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_RBR_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2019: 1034')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2019 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2019 = np.concatenate((hours_sd2019,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2019 = np.concatenate((eflux_sd2019,dff['eflux'].values),axis=None)
hflux_sd2019 = np.concatenate((hflux_sd2019,dff['hflux'].values),axis=None)
rh_sd2019 = np.concatenate((rh_sd2019,dff['rh'].values),axis=None)
airtemp_sd2019 = np.concatenate((airtemp_sd2019,dff['airtemp'].values),axis=None)
skntemp_sd2019 = np.concatenate((skntemp_sd2019,dff['skntemp'].values),axis=None)
sp_sd2019 = np.concatenate((sp_sd2019,dff['sp'].values),axis=None)
wind_sd2019 = np.concatenate((wind_sd2019,dff['wind'].values),axis=None)
roundedhour_sd2019 = np.concatenate((roundedhour_sd2019,hours[preciseidx]),axis=None)
indlat_sd2019 =  np.concatenate((indlat_sd2019,lat[preciseidx]),axis=None)
indlon_sd2019 =  np.concatenate((indlon_sd2019,lon[preciseidx]),axis=None)
id_sd2019 = np.concatenate((id_sd2019,np.full(np.shape(dff['hours_sd_rounded'].values), 1034)))
print('done with SD2019: 1034')

## 1035
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019obs/sd1035.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019calc/sd1035.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_RBR_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2019: 1035')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2019 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2019 = np.concatenate((hours_sd2019,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2019 = np.concatenate((eflux_sd2019,dff['eflux'].values),axis=None)
hflux_sd2019 = np.concatenate((hflux_sd2019,dff['hflux'].values),axis=None)
rh_sd2019 = np.concatenate((rh_sd2019,dff['rh'].values),axis=None)
airtemp_sd2019 = np.concatenate((airtemp_sd2019,dff['airtemp'].values),axis=None)
skntemp_sd2019 = np.concatenate((skntemp_sd2019,dff['skntemp'].values),axis=None)
sp_sd2019 = np.concatenate((sp_sd2019,dff['sp'].values),axis=None)
wind_sd2019 = np.concatenate((wind_sd2019,dff['wind'].values),axis=None)
roundedhour_sd2019 = np.concatenate((roundedhour_sd2019,hours[preciseidx]),axis=None)
indlat_sd2019 =  np.concatenate((indlat_sd2019,lat[preciseidx]),axis=None)
indlon_sd2019 =  np.concatenate((indlon_sd2019,lon[preciseidx]),axis=None)
id_sd2019 = np.concatenate((id_sd2019,np.full(np.shape(dff['hours_sd_rounded'].values), 1035)))

print('done with SD2019: 1035')

## 1036
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019obs/sd1036.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019calc/sd1036.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_RBR_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2019: 1036')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2019 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2019 = np.concatenate((hours_sd2019,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2019 = np.concatenate((eflux_sd2019,dff['eflux'].values),axis=None)
hflux_sd2019 = np.concatenate((hflux_sd2019,dff['hflux'].values),axis=None)
rh_sd2019 = np.concatenate((rh_sd2019,dff['rh'].values),axis=None)
airtemp_sd2019 = np.concatenate((airtemp_sd2019,dff['airtemp'].values),axis=None)
skntemp_sd2019 = np.concatenate((skntemp_sd2019,dff['skntemp'].values),axis=None)
sp_sd2019 = np.concatenate((sp_sd2019,dff['sp'].values),axis=None)
wind_sd2019 = np.concatenate((wind_sd2019,dff['wind'].values),axis=None)
roundedhour_sd2019 = np.concatenate((roundedhour_sd2019,hours[preciseidx]),axis=None)
indlat_sd2019 =  np.concatenate((indlat_sd2019,lat[preciseidx]),axis=None)
indlon_sd2019 =  np.concatenate((indlon_sd2019,lon[preciseidx]),axis=None)
id_sd2019 = np.concatenate((id_sd2019,np.full(np.shape(dff['hours_sd_rounded'].values), 1036)))

print('done with SD2019: 1036')

## 1037
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019obs/sd1037.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019calc/sd1037.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_RBR_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2019: 1037')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2019 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2019 = np.concatenate((hours_sd2019,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2019 = np.concatenate((eflux_sd2019,dff['eflux'].values),axis=None)
hflux_sd2019 = np.concatenate((hflux_sd2019,dff['hflux'].values),axis=None)
rh_sd2019 = np.concatenate((rh_sd2019,dff['rh'].values),axis=None)
airtemp_sd2019 = np.concatenate((airtemp_sd2019,dff['airtemp'].values),axis=None)
skntemp_sd2019 = np.concatenate((skntemp_sd2019,dff['skntemp'].values),axis=None)
sp_sd2019 = np.concatenate((sp_sd2019,dff['sp'].values),axis=None)
wind_sd2019 = np.concatenate((wind_sd2019,dff['wind'].values),axis=None)
roundedhour_sd2019 = np.concatenate((roundedhour_sd2019,hours[preciseidx]),axis=None)
indlat_sd2019 =  np.concatenate((indlat_sd2019,lat[preciseidx]),axis=None)
indlon_sd2019 =  np.concatenate((indlon_sd2019,lon[preciseidx]),axis=None)
id_sd2019 = np.concatenate((id_sd2019,np.full(np.shape(dff['hours_sd_rounded'].values), 1037)))

print('done with SD2019: 1037')

## 1041
filename_obs = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019obs/sd1041.nc'
filename_calc = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/2019/2019calc/sd1041.nc'
f_obs = nc.Dataset(filename_obs,mode='r')
f_calc = nc.Dataset(filename_calc,mode='r')

# observations
time = f_obs.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_obs.variables['latitude'][:] # degrees
lon = f_obs.variables['longitude'][:] # degrees
rh =  f_obs.variables['RH_MEAN'][:] # %
airtemp = f_obs.variables['TEMP_AIR_MEAN'][:] # C
skntemp = f_obs.variables['TEMP_CTD_RBR_MEAN'][:] # C
wind = f_obs.variables['wind_speed'][:] # m/s^2
sp = f_obs.variables['BARO_PRES_MEAN'][:] # hPa

# remove NAN
bundle = np.stack((time,lat,lon,airtemp,skntemp,sp,rh,wind),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_obs = bundle[:,0]
lat_obs = bundle[:,1]
lon_obs = bundle[:,2]
rh_obs = bundle[:,6]
airtemp_obs = bundle[:,3]
skntemp_obs = bundle[:,4]
wind_obs = bundle[:,7]
sp_obs = bundle[:,5]

# flux
time = f_calc.variables['time'][:] # seconds since 01/01/1970 00:00:00
lat = f_calc.variables['latitude'][:] # degrees
lon = f_calc.variables['longitude'][:] # degrees
eflux = f_calc.variables['QL'][:] # W/m^2
hflux = f_calc.variables['QS'][:] # W/m^2

# remove NAN
bundle = np.stack((time,lat,lon,eflux,hflux),axis=1)
bundle = np.ma.masked_where(bundle == -1.0E34, bundle)
bundle = np.ma.compress_rows(bundle)
time_calc = bundle[:,0]
lat_calc = bundle[:,1]
lon_calc = bundle[:,2]
eflux_calc = bundle[:,3]
hflux_calc = bundle[:,4]

# match by time
[idx_obs,idx_calc] = solution(time_obs,time_calc)
idx_calc = np.array(idx_calc)
time = np.array(np.copy(time_calc[idx_calc]))
lat = np.array(np.copy(lat[idx_calc]))
lon = np.array(np.copy(lon[idx_calc]))
eflux = np.array(np.copy(eflux_calc[idx_calc]))
hflux = np.array(np.copy(hflux_calc[idx_calc]))
idx_obs = np.array(idx_obs)
rh = np.array(np.copy(rh_obs[idx_obs]))
airtemp = np.array(np.copy(airtemp_obs[idx_obs]))
skntemp = np.array(np.copy(skntemp_obs[idx_obs]))
wind = np.array(np.copy(wind_obs[idx_obs]))
sp = np.array(np.copy(sp_obs[idx_obs]))
print('start averaging: SD2019: 1041')

# average by time
hours = time/60/60 - yeardiff #hours since 05/01/2019 00:30

# find average of variables
hours_sd_rounded = np.int64(np.round(hours))
for_df = np.array([hours_sd_rounded,eflux,hflux,rh,airtemp,skntemp,wind,sp])
df = pd.DataFrame(for_df.T, columns=['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp'])
df['hours_sd_rounded'] = np.int64(df["hours_sd_rounded"])
dff = df.groupby(['hours_sd_rounded'])[['hours_sd_rounded','eflux','hflux','rh','airtemp','skntemp','wind','sp']].mean()
hours_rounded = dff['hours_sd_rounded'].values
preciseidx = []
for hour in hours_rounded:
    preciseidx.append(find_nearest(hours, hour))

## final variables
hours_sd2019 = np.concatenate((hours_sd2019,dff['hours_sd_rounded'].values),axis=None)
eflux_sd2019 = np.concatenate((eflux_sd2019,dff['eflux'].values),axis=None)
hflux_sd2019 = np.concatenate((hflux_sd2019,dff['hflux'].values),axis=None)
rh_sd2019 = np.concatenate((rh_sd2019,dff['rh'].values),axis=None)
airtemp_sd2019 = np.concatenate((airtemp_sd2019,dff['airtemp'].values),axis=None)
skntemp_sd2019 = np.concatenate((skntemp_sd2019,dff['skntemp'].values),axis=None)
sp_sd2019 = np.concatenate((sp_sd2019,dff['sp'].values),axis=None)
wind_sd2019 = np.concatenate((wind_sd2019,dff['wind'].values),axis=None)
roundedhour_sd2019 = np.concatenate((roundedhour_sd2019,hours[preciseidx]),axis=None)
indlat_sd2019 =  np.concatenate((indlat_sd2019,lat[preciseidx]),axis=None)
indlon_sd2019 =  np.concatenate((indlon_sd2019,lon[preciseidx]),axis=None)
id_sd2019 = np.concatenate((id_sd2019,np.full(np.shape(dff['hours_sd_rounded'].values), 1041)))

print('done with SD2019: 1041')

# conversions for sea
essea_sd = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(skntemp_sd2019+273.15)))) # sea saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
esea_sd = essea_sd * (rh_sd2019/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qskn = 0.622*(esea_sd/(sp_sd2019-esea_sd))*1000 # specific humidity of sea surface : Wallace and Hobbs (pg. 80)

# conversions for air
esair_sd = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(airtemp_sd2019+273.15)))) # air saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
eair_sd = esair_sd * (rh_sd2019/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qair = 0.622*(eair_sd/(sp_sd2019-eair_sd))*1000 # specific humidity of atmosphere : Wallace and Hobbs (pg. 80)

# export file
for_df = np.array([id_sd2019,hours_sd2019,indlat_sd2019, indlon_sd2019,eflux_sd2019,hflux_sd2019,qair,qskn,airtemp_sd2019,skntemp_sd2019,wind_sd2019,sp_sd2019])
df = pd.DataFrame(for_df.T, columns=['id','hours','lat','lon','eflux','hflux','qair','qskn','airtemp','skntemp','wind','sp'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2019E5avghr.csv')
print('...end 2019.')

print('end SAILDRONE DATA EXTRACTION for ERA5.')