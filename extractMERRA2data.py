### MERRA2 DATASET INTERPOLATION ####
### SUBHATRA SIVAM  ###

# import packages #
from math import radians
import netCDF4 as nc
import numpy as np
import pandas as pd
from typing import Tuple

# functions #
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
def latLongL2(original, secondary, k=4):
    o_lat = original[:,1]
    o_long = original[:,0]
    s_lat = secondary[:,1]
    s_long = secondary[:,0]
    
    diffs = (o_lat[:, None] - s_lat[None, :])**2 + (o_long[:, None] - s_long[None, :])**2
    indices = np.argpartition(diffs, k, axis=1)[:, :k]
    return indices
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def square(list):
    return [i ** 2 for i in list]
def squared(list):
    return [i ** 1/2 for i in list]

### MERRA 2 INPUT VARIABLES ###
# time, lon, lat
# QV10M, T10M, TSKINWTR
# U10, V10, EFLUXWTR, HFLUXWTR

# import MERRA2 data #
print('start MERRA2 import...')

## 2017
filename_prod = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/MERRA2_2017.nc'
f = nc.Dataset(filename_prod,mode='r')
time_prod2017 = f.variables['time'][:] # hours since 05/01/2017 00:30:00
lon_prod2017 = f.variables['lon'][:] # degrees
lat_prod2017 = f.variables['lat'][:] # degrees
q_prod2017 = f.variables['QV10M'][:] # kg/kg
airtemp_prod2017 = f.variables['T10M'][:] # K
skntemp_prod2017 = f.variables['TSKINWTR'][:] # K
uwind_prod2017 = f.variables['U10M'][:] # m*s^-1
vwind_prod2017 = f.variables['V10M'][:] # m*s^-1
eflux_prod2017 = f.variables['EFLUXWTR'][:] # W*m^-2
hflux_prod2017 = f.variables['HFLUXWTR'][:] # W*m^-2
print('done with MERRA2 2017')

## 2018
filename_prod = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/MERRA2_2018.nc'
f = nc.Dataset(filename_prod,mode='r')
time_prod2018 = f.variables['time'][:] # hours since 05/01/2018 00:30:00
lon_prod2018 = f.variables['lon'][:] # degrees
lat_prod2018 = f.variables['lat'][:] # degrees
q_prod2018 = f.variables['QV10M'][:] # kg/kg
airtemp_prod2018 = f.variables['T10M'][:] # K
skntemp_prod2018 = f.variables['TSKINWTR'][:] # K
uwind_prod2018 = f.variables['U10M'][:] # m*s^-1
vwind_prod2018 = f.variables['V10M'][:] # m*s^-1
eflux_prod2018 = f.variables['EFLUXWTR'][:] # W*m^-2
hflux_prod2018 = f.variables['HFLUXWTR'][:] # W*m^-2
print('done with MERRA2 2018')

## 2019
filename_prod = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/MERRA2_2019.nc'
f = nc.Dataset(filename_prod,mode='r')
time_prod2019 = f.variables['time'][:] # hours since 05/01/2019 00:30:00
lon_prod2019 = f.variables['lon'][:] # degrees
lat_prod2019 = f.variables['lat'][:] # degrees
q_prod2019 = f.variables['QV10M'][:] # kg/kg
airtemp_prod2019 = f.variables['T10M'][:] # K
skntemp_prod2019 = f.variables['TSKINWTR'][:] # K
uwind_prod2019 = f.variables['U10M'][:] # m*s^-1
vwind_prod2019 = f.variables['V10M'][:] # m*s^-1
eflux_prod2019 = f.variables['EFLUXWTR'][:] # W*m^-2
hflux_prod2019 = f.variables['HFLUXWTR'][:] # W*m^-2
print('done with MERRA2 2019')

print('...end MERRA2 import.')

# import saildrone data #
sd2017 = pd.read_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2017avghr.csv')
sd2018 = pd.read_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2018avghr.csv')
sd2019 = pd.read_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2019avghr.csv')

# interpolate 2017 data #
print('start 2017 interpolation...')

## MERRA2 VARIABLES ##
time_prod = time_prod2017
lon_prod = lon_prod2017
lat_prod = lat_prod2017
q_prod = q_prod2017
airtemp_prod = airtemp_prod2017
skntemp_prod = skntemp_prod2017
uwind_prod = uwind_prod2017
vwind_prod = vwind_prod2017
eflux_prod = eflux_prod2017
hflux_prod = hflux_prod2017

## SAILDRONE VARIABLES ##
time_sd = sd2017['hours'].values
lon_sd = sd2017['lon'].values
lat_sd = sd2017['lat'].values

location_sd = np.stack((lon_sd,lat_sd),axis=1)

# get MERRA location grid
gridlocation = []
grididx = []
for idx_lon, lon in enumerate(lon_prod):
    for idx_lat,lat in enumerate(lat_prod):
        gridlocation.append([lon,lat])
        grididx.append([idx_lon,idx_lat])
gridlocation = np.asarray(gridlocation)
grididx = np.asarray(grididx)

# find the corresponding times between both MERRA2 and saildrone  files
[idx_prod,idx_sd] = solution(time_prod,time_sd)
idx_prod = np.array(idx_prod)
timeidx_prod = np.copy(time_prod[idx_prod])
efluxidx_prod = np.copy(eflux_prod[idx_prod])
hfluxidx_prod = np.copy(hflux_prod[idx_prod])
airtempidx_prod = np.copy(airtemp_prod[idx_prod])
skntempidx_prod = np.copy(skntemp_prod[idx_prod])
qidx_prod = np.copy(q_prod[idx_prod])
uwindidx_prod = np.copy(uwind_prod[idx_prod])
vwindidx_prod = np.copy(vwind_prod[idx_prod])

# get distance estimations for the MERRA2 model (has everything)
close4 = latLongL2(location_sd,gridlocation)
percentdist = []
percentdistog = []
for grid_idx, point in enumerate(location_sd):
    ind_close4 = close4[grid_idx]
    ind_gridlocations = []
    distancegrid = []
    distancegridog = []
    for ind in ind_close4:
        gridpoint = gridlocation[ind]
        ind_gridlocations.append(gridpoint)
        distancepixog =((point[0] - gridpoint[0])**2 + (point[1] - gridpoint[1])**2)**1/2
        lata, lona, latb, lonb, R = radians(gridpoint[1]),radians(gridpoint[0]), radians(point[1]), radians(point[0]), 6378.0
        lat_diff = latb-lata
        lon_diff = lonb-lona
        a = np.sin(lat_diff/2)**2+np.cos(latb)*np.cos(lata)*np.sin(lon_diff/2)**2
        c = 2*np.arctan2(a**(1/2),(1-a)**(1/2))
        distancepix = R * c
        distancegrid.append(np.divide(1,distancepix))
        distancegridog.append(np.divide(1,distancepixog))
    sumdist = np.sum(distancegrid)
    sumdistog = np.sum(distancegridog)
    percentdist.append(distancegrid/sumdist)
    percentdistog.append(distancegridog/sumdistog)
curiousdist = percentdist
curiousdistog = percentdistog

# get the MERRA flux values for every saildrone time/locationeflux_prod_withsd = []
eflux_prod_withsd = []
hflux_prod_withsd = []
for time_s in time_sd:
    for idx_time,time_m in enumerate(timeidx_prod):
        if time_s == time_m:
            eflux_prod_withsd.append(efluxidx_prod[idx_time])
            hflux_prod_withsd.append(hfluxidx_prod[idx_time])
eflux_prod_withsd = np.array(eflux_prod_withsd)
hflux_prod_withsd = np.array(hflux_prod_withsd)

# get the MERRA observation values for every saildrone time/location
airtemp_prod_withsd = []
q_prod_withsd = []
skntemp_prod_withsd = []
uwind_prod_withsd = []
vwind_prod_withsd = []
skntemplatlot = []
uwindtemplatlot = []
vwindtemplatlot = []
for time_s in time_sd:
    for idx_time,time_m in enumerate(timeidx_prod):
        if time_s == time_m:
            airtemp_prod_withsd.append(airtempidx_prod[idx_time])
            q_prod_withsd.append(qidx_prod[idx_time])
            skntemp_prod_withsd.append(skntempidx_prod[idx_time])
            uwind_prod_withsd.append(uwindidx_prod[idx_time])
            vwind_prod_withsd.append(vwindidx_prod[idx_time])
airtemp_prod_withsd = np.array(airtemp_prod_withsd)
q_prod_withsd = np.array(q_prod_withsd)
skntemp_prod_withsd = np.array(skntemp_prod_withsd)
uwind_prod_withsd = np.array(uwind_prod_withsd)
vwind_prod_withsd = np.array(vwind_prod_withsd)

# get longitude and latitude values
timeidx = np.arange(0,len(time_sd),1)
lat1 = []
lat2 = []
lat3 = []
lat4 = []
lon1 = []
lon2 = []
lon3 = []
lon4 = []
for k in close4:
    lat1.append(gridlocation[k[0]][1])
    lon1.append(gridlocation[k[0]][0])
    lat2.append(gridlocation[k[1]][1])
    lon2.append(gridlocation[k[1]][0])
    lat3.append(gridlocation[k[2]][1])
    lon3.append(gridlocation[k[2]][0])
    lat4.append(gridlocation[k[3]][1])
    lon4.append(gridlocation[k[3]][0])
arraylon1 = []
arraylon2 = []
arraylon3 = []
arraylon4 = []
arraylat1 = []
arraylat2 = []
arraylat3 = []
arraylat4 = []
for idx,lon in enumerate(lon1):
    arraylon1.append(find_nearest(lon_prod,lon))
    arraylon2.append(find_nearest(lon_prod,lon2[idx]))
    arraylon3.append(find_nearest(lon_prod,lon3[idx]))
    arraylon4.append(find_nearest(lon_prod,lon4[idx]))
    arraylat1.append(find_nearest(lat_prod,lat1[idx]))
    arraylat2.append(find_nearest(lat_prod,lat2[idx]))
    arraylat3.append(find_nearest(lat_prod,lat3[idx]))
    arraylat4.append(find_nearest(lat_prod,lat4[idx]))  
latlon1 = np.stack((arraylat1,arraylon1),axis=1)
latlon2 = np.stack((arraylat2,arraylon2),axis=1)   
latlon3 = np.stack((arraylat3,arraylon3),axis=1)   
latlon4 = np.stack((arraylat4,arraylon4),axis=1)

# get the closest 4 values
timeidx = np.arange(0,len(time_sd),1)
efluxlatlon1 = []
hfluxlatlon1 = []
efluxlatlon2 = []
hfluxlatlon2 = []
efluxlatlon3 = []
hfluxlatlon3 = []
efluxlatlon4 = []
hfluxlatlon4 = []
airtemplatlon1 = []
airtemplatlon2 = []
airtemplatlon3 = []
airtemplatlon4 = []
qlatlon1 = []
qlatlon2 = []
qlatlon3 = []
qlatlon4 = []
skntemplatlot1 = []
skntemplatlot2 = []
skntemplatlot3 = []
skntemplatlot4 = []
uwindtemplatlot1 = []
uwindtemplatlot2 = []
uwindtemplatlot3 = []
uwindtemplatlot4 = []
vwindtemplatlot1 = []
vwindtemplatlot2 = []
vwindtemplatlot3 = []
vwindtemplatlot4 = []
for j in timeidx:
    efluxlatlon1.append(eflux_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    hfluxlatlon1.append(hflux_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    efluxlatlon2.append(eflux_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    hfluxlatlon2.append(hflux_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    efluxlatlon3.append(eflux_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    hfluxlatlon3.append(hflux_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    efluxlatlon4.append(eflux_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    hfluxlatlon4.append(hflux_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    airtemplatlon1.append(airtemp_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    qlatlon1.append(q_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    skntemplatlot1.append(skntemp_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    uwindtemplatlot1.append(uwind_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    vwindtemplatlot1.append(vwind_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    airtemplatlon2.append(airtemp_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    qlatlon2.append(q_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    skntemplatlot2.append(skntemp_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    uwindtemplatlot2.append(uwind_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    vwindtemplatlot2.append(vwind_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    airtemplatlon3.append(airtemp_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    qlatlon3.append(q_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    skntemplatlot3.append(skntemp_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    uwindtemplatlot3.append(uwind_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    vwindtemplatlot3.append(vwind_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    airtemplatlon4.append(airtemp_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    qlatlon4.append(q_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    skntemplatlot4.append(skntemp_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    uwindtemplatlot4.append(uwind_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    vwindtemplatlot4.append(vwind_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
efluxlatlon1 = np.array(efluxlatlon1)
hfluxlatlon1 = np.array(hfluxlatlon1)
efluxlatlon2 = np.array(efluxlatlon2)
hfluxlatlon2 = np.array(hfluxlatlon2)
efluxlatlon3 = np.array(efluxlatlon3)
hfluxlatlon3 = np.array(hfluxlatlon3)
efluxlatlon4 = np.array(efluxlatlon4)
hfluxlatlon4 = np.array(hfluxlatlon4)
airtemplatlon1 = np.array(airtemplatlon1)
qlatlon1 = np.array(qlatlon1)
skntemplatlot1 = np.array(skntemplatlot1)
uwindtemplatlot1 = np.array(uwindtemplatlot1)
vwindtemplatlot1 = np.array(vwindtemplatlot1)
airtemplatlon2 = np.array(airtemplatlon2)
qlatlon2 = np.array(qlatlon2)
skntemplatlot2 = np.array(skntemplatlot2)
uwindtemplatlot2 = np.array(uwindtemplatlot2)
vwindtemplatlot2 = np.array(vwindtemplatlot2)
airtemplatlon3 = np.array(airtemplatlon3)
qlatlon3 = np.array(qlatlon3)
skntemplatlot3 = np.array(skntemplatlot3)
uwindtemplatlot3 = np.array(uwindtemplatlot3)
vwindtemplatlot3 = np.array(vwindtemplatlot3)
airtemplatlon4 = np.array(airtemplatlon4)
qlatlon4 = np.array(qlatlon4)
skntemplatlot4 = np.array(skntemplatlot4)
uwindtemplatlot4 = np.array(uwindtemplatlot4)
vwindtemplatlot4 = np.array(vwindtemplatlot4)
percentdist = np.array(percentdist)

# get weighted average
eflux = efluxlatlon1*percentdist[:,0] + efluxlatlon2*percentdist[:,1] + efluxlatlon3*percentdist[:,2] + efluxlatlon4*percentdist[:,3]
hflux = hfluxlatlon1*percentdist[:,0] + hfluxlatlon2*percentdist[:,1] + hfluxlatlon3*percentdist[:,2] + hfluxlatlon4*percentdist[:,3]
airtemp = airtemplatlon1*percentdist[:,0] + airtemplatlon2*percentdist[:,1] + airtemplatlon3*percentdist[:,2] + airtemplatlon4*percentdist[:,3]
skntemp = skntemplatlot1*percentdist[:,0] + skntemplatlot2*percentdist[:,1] + skntemplatlot3*percentdist[:,2] + skntemplatlot4*percentdist[:,3]
q = qlatlon1*percentdist[:,0] + qlatlon2*percentdist[:,1] + qlatlon3*percentdist[:,2] + qlatlon4*percentdist[:,3]
uwind = uwindtemplatlot1*percentdist[:,0] + uwindtemplatlot2*percentdist[:,1] + uwindtemplatlot3*percentdist[:,2] + uwindtemplatlot4*percentdist[:,3]
vwind = vwindtemplatlot1*percentdist[:,0] + vwindtemplatlot2*percentdist[:,1] + vwindtemplatlot3*percentdist[:,2] + vwindtemplatlot4*percentdist[:,3]

airtemp = np.subtract(airtemp,273.15) # K -> C
airtemp = np.array(airtemp)
skntemp = np.subtract(skntemp,273.15) # K -> C
skntemp = np.array(skntemp)

# convert units
usq = square(uwind)
vsq = square(vwind)
sum  = np.add(usq,vsq)
windidx_prod = squared(sum) # u and v -> total magnitude
wind = np.array(windidx_prod)
qair = np.multiply(q,1000) # kg/kg -> g/kg

es_tempfin_prod = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(airtemp+273.15)))) # air saturation vapor pressure
es_sknfin_prod = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(skntemp+273.15)))) # sea saturation vapor pressure

rat_prh = (62.2/qair+100)*es_tempfin_prod
qskn = 62.2/(rat_prh/es_sknfin_prod-100)

# export file
for_df = np.array([time_sd,eflux,hflux,qair,qskn,airtemp,skntemp,wind])
for_df = for_df.T
for_df = np.ma.masked_where(for_df == 999999986991104, for_df)
for_df = np.ma.compress_rows(for_df)
df = pd.DataFrame(for_df, columns=['time','eflux','hflux','qair','qskn','airtemp','skntemp','wind'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/2017MERRA2interp.csv')

print('...end 2017 interpolation.')
###################################################################################################
# interpolate 2018 data #
print('start 2018 interpolation...')

## MERRA2 VARIABLES ##
time_prod = time_prod2018
lon_prod = lon_prod2018
lat_prod = lat_prod2018
q_prod = q_prod2018
airtemp_prod = airtemp_prod2018
skntemp_prod = skntemp_prod2018
uwind_prod = uwind_prod2018
vwind_prod = vwind_prod2018
eflux_prod = eflux_prod2018
hflux_prod = hflux_prod2018

## SAILDRONE VARIABLES ##
time_sd = sd2018['hours'].values
lon_sd = sd2018['lon'].values
lat_sd = sd2018['lat'].values

location_sd = np.stack((lon_sd,lat_sd),axis=1)

# get MERRA location grid
gridlocation = []
grididx = []
for idx_lon, lon in enumerate(lon_prod):
    for idx_lat,lat in enumerate(lat_prod):
        gridlocation.append([lon,lat])
        grididx.append([idx_lon,idx_lat])
gridlocation = np.asarray(gridlocation)
grididx = np.asarray(grididx)

# find the corresponding times between both MERRA2 and saildrone  files
[idx_prod,idx_sd] = solution(time_prod,time_sd)
idx_prod = np.array(idx_prod)
timeidx_prod = np.copy(time_prod[idx_prod])
efluxidx_prod = np.copy(eflux_prod[idx_prod])
hfluxidx_prod = np.copy(hflux_prod[idx_prod])
airtempidx_prod = np.copy(airtemp_prod[idx_prod])
skntempidx_prod = np.copy(skntemp_prod[idx_prod])
qidx_prod = np.copy(q_prod[idx_prod])
uwindidx_prod = np.copy(uwind_prod[idx_prod])
vwindidx_prod = np.copy(vwind_prod[idx_prod])

# get distance estimations for the MERRA2 model (has everything)
close4 = latLongL2(location_sd,gridlocation)
percentdist = []
percentdistog = []
for grid_idx, point in enumerate(location_sd):
    ind_close4 = close4[grid_idx]
    ind_gridlocations = []
    distancegrid = []
    distancegridog = []
    for ind in ind_close4:
        gridpoint = gridlocation[ind]
        ind_gridlocations.append(gridpoint)
        distancepixog =((point[0] - gridpoint[0])**2 + (point[1] - gridpoint[1])**2)**1/2
        lata, lona, latb, lonb, R = radians(gridpoint[1]),radians(gridpoint[0]), radians(point[1]), radians(point[0]), 6378.0
        lat_diff = latb-lata
        lon_diff = lonb-lona
        a = np.sin(lat_diff/2)**2+np.cos(latb)*np.cos(lata)*np.sin(lon_diff/2)**2
        c = 2*np.arctan2(a**(1/2),(1-a)**(1/2))
        distancepix = R * c
        distancegrid.append(np.divide(1,distancepix))
        distancegridog.append(np.divide(1,distancepixog))
    sumdist = np.sum(distancegrid)
    sumdistog = np.sum(distancegridog)
    percentdist.append(distancegrid/sumdist)
    percentdistog.append(distancegridog/sumdistog)
curiousdist = np.concatenate([curiousdist,percentdist],axis=0)
curiousdistog = np.concatenate([curiousdistog,percentdistog],axis=0)

# get the MERRA flux values for every saildrone time/location
eflux_prod_withsd = []
hflux_prod_withsd = []
for time_s in time_sd:
    for idx_time,time_m in enumerate(timeidx_prod):
        if time_s == time_m:
            eflux_prod_withsd.append(efluxidx_prod[idx_time])
            hflux_prod_withsd.append(hfluxidx_prod[idx_time])
eflux_prod_withsd = np.array(eflux_prod_withsd)
hflux_prod_withsd = np.array(hflux_prod_withsd)

# get the MERRA observation values for every saildrone time/location
airtemp_prod_withsd = []
q_prod_withsd = []
skntemp_prod_withsd = []
uwind_prod_withsd = []
vwind_prod_withsd = []
skntemplatlot = []
uwindtemplatlot = []
vwindtemplatlot = []
for time_s in time_sd:
    for idx_time,time_m in enumerate(timeidx_prod):
        if time_s == time_m:
            airtemp_prod_withsd.append(airtempidx_prod[idx_time])
            q_prod_withsd.append(qidx_prod[idx_time])
            skntemp_prod_withsd.append(skntempidx_prod[idx_time])
            uwind_prod_withsd.append(uwindidx_prod[idx_time])
            vwind_prod_withsd.append(vwindidx_prod[idx_time])
airtemp_prod_withsd = np.array(airtemp_prod_withsd)
q_prod_withsd = np.array(q_prod_withsd)
skntemp_prod_withsd = np.array(skntemp_prod_withsd)
uwind_prod_withsd = np.array(uwind_prod_withsd)
vwind_prod_withsd = np.array(vwind_prod_withsd)

# get longitude and latitude values
timeidx = np.arange(0,len(time_sd),1)
lat1 = []
lat2 = []
lat3 = []
lat4 = []
lon1 = []
lon2 = []
lon3 = []
lon4 = []
for k in close4:
    lat1.append(gridlocation[k[0]][1])
    lon1.append(gridlocation[k[0]][0])
    lat2.append(gridlocation[k[1]][1])
    lon2.append(gridlocation[k[1]][0])
    lat3.append(gridlocation[k[2]][1])
    lon3.append(gridlocation[k[2]][0])
    lat4.append(gridlocation[k[3]][1])
    lon4.append(gridlocation[k[3]][0])
arraylon1 = []
arraylon2 = []
arraylon3 = []
arraylon4 = []
arraylat1 = []
arraylat2 = []
arraylat3 = []
arraylat4 = []
for idx,lon in enumerate(lon1):
    arraylon1.append(find_nearest(lon_prod,lon))
    arraylon2.append(find_nearest(lon_prod,lon2[idx]))
    arraylon3.append(find_nearest(lon_prod,lon3[idx]))
    arraylon4.append(find_nearest(lon_prod,lon4[idx]))
    arraylat1.append(find_nearest(lat_prod,lat1[idx]))
    arraylat2.append(find_nearest(lat_prod,lat2[idx]))
    arraylat3.append(find_nearest(lat_prod,lat3[idx]))
    arraylat4.append(find_nearest(lat_prod,lat4[idx]))  
latlon1 = np.stack((arraylat1,arraylon1),axis=1)
latlon2 = np.stack((arraylat2,arraylon2),axis=1)   
latlon3 = np.stack((arraylat3,arraylon3),axis=1)   
latlon4 = np.stack((arraylat4,arraylon4),axis=1)

# get the closest 4 values
timeidx = np.arange(0,len(time_sd),1)
efluxlatlon1 = []
hfluxlatlon1 = []
efluxlatlon2 = []
hfluxlatlon2 = []
efluxlatlon3 = []
hfluxlatlon3 = []
efluxlatlon4 = []
hfluxlatlon4 = []
airtemplatlon1 = []
airtemplatlon2 = []
airtemplatlon3 = []
airtemplatlon4 = []
qlatlon1 = []
qlatlon2 = []
qlatlon3 = []
qlatlon4 = []
skntemplatlot1 = []
skntemplatlot2 = []
skntemplatlot3 = []
skntemplatlot4 = []
uwindtemplatlot1 = []
uwindtemplatlot2 = []
uwindtemplatlot3 = []
uwindtemplatlot4 = []
vwindtemplatlot1 = []
vwindtemplatlot2 = []
vwindtemplatlot3 = []
vwindtemplatlot4 = []
for j in timeidx:
    efluxlatlon1.append(eflux_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    hfluxlatlon1.append(hflux_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    efluxlatlon2.append(eflux_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    hfluxlatlon2.append(hflux_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    efluxlatlon3.append(eflux_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    hfluxlatlon3.append(hflux_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    efluxlatlon4.append(eflux_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    hfluxlatlon4.append(hflux_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    airtemplatlon1.append(airtemp_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    qlatlon1.append(q_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    skntemplatlot1.append(skntemp_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    uwindtemplatlot1.append(uwind_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    vwindtemplatlot1.append(vwind_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    airtemplatlon2.append(airtemp_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    qlatlon2.append(q_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    skntemplatlot2.append(skntemp_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    uwindtemplatlot2.append(uwind_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    vwindtemplatlot2.append(vwind_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    airtemplatlon3.append(airtemp_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    qlatlon3.append(q_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    skntemplatlot3.append(skntemp_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    uwindtemplatlot3.append(uwind_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    vwindtemplatlot3.append(vwind_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    airtemplatlon4.append(airtemp_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    qlatlon4.append(q_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    skntemplatlot4.append(skntemp_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    uwindtemplatlot4.append(uwind_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    vwindtemplatlot4.append(vwind_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
efluxlatlon1 = np.array(efluxlatlon1)
hfluxlatlon1 = np.array(hfluxlatlon1)
efluxlatlon2 = np.array(efluxlatlon2)
hfluxlatlon2 = np.array(hfluxlatlon2)
efluxlatlon3 = np.array(efluxlatlon3)
hfluxlatlon3 = np.array(hfluxlatlon3)
efluxlatlon4 = np.array(efluxlatlon4)
hfluxlatlon4 = np.array(hfluxlatlon4)
airtemplatlon1 = np.array(airtemplatlon1)
qlatlon1 = np.array(qlatlon1)
skntemplatlot1 = np.array(skntemplatlot1)
uwindtemplatlot1 = np.array(uwindtemplatlot1)
vwindtemplatlot1 = np.array(vwindtemplatlot1)
airtemplatlon2 = np.array(airtemplatlon2)
qlatlon2 = np.array(qlatlon2)
skntemplatlot2 = np.array(skntemplatlot2)
uwindtemplatlot2 = np.array(uwindtemplatlot2)
vwindtemplatlot2 = np.array(vwindtemplatlot2)
airtemplatlon3 = np.array(airtemplatlon3)
qlatlon3 = np.array(qlatlon3)
skntemplatlot3 = np.array(skntemplatlot3)
uwindtemplatlot3 = np.array(uwindtemplatlot3)
vwindtemplatlot3 = np.array(vwindtemplatlot3)
airtemplatlon4 = np.array(airtemplatlon4)
qlatlon4 = np.array(qlatlon4)
skntemplatlot4 = np.array(skntemplatlot4)
uwindtemplatlot4 = np.array(uwindtemplatlot4)
vwindtemplatlot4 = np.array(vwindtemplatlot4)
percentdist = np.array(percentdist)

# get weighted average
eflux = efluxlatlon1*percentdist[:,0] + efluxlatlon2*percentdist[:,1] + efluxlatlon3*percentdist[:,2] + efluxlatlon4*percentdist[:,3]
hflux = hfluxlatlon1*percentdist[:,0] + hfluxlatlon2*percentdist[:,1] + hfluxlatlon3*percentdist[:,2] + hfluxlatlon4*percentdist[:,3]
airtemp = airtemplatlon1*percentdist[:,0] + airtemplatlon2*percentdist[:,1] + airtemplatlon3*percentdist[:,2] + airtemplatlon4*percentdist[:,3]
skntemp = skntemplatlot1*percentdist[:,0] + skntemplatlot2*percentdist[:,1] + skntemplatlot3*percentdist[:,2] + skntemplatlot4*percentdist[:,3]
q = qlatlon1*percentdist[:,0] + qlatlon2*percentdist[:,1] + qlatlon3*percentdist[:,2] + qlatlon4*percentdist[:,3]
uwind = uwindtemplatlot1*percentdist[:,0] + uwindtemplatlot2*percentdist[:,1] + uwindtemplatlot3*percentdist[:,2] + uwindtemplatlot4*percentdist[:,3]
vwind = vwindtemplatlot1*percentdist[:,0] + vwindtemplatlot2*percentdist[:,1] + vwindtemplatlot3*percentdist[:,2] + vwindtemplatlot4*percentdist[:,3]

# convert units
airtemp = np.subtract(airtemp,273.15) # K -> C
airtemp = np.array(airtemp)
skntemp = np.subtract(skntemp,273.15) # K -> C
skntemp = np.array(skntemp)
def square(list):
    return [i ** 2 for i in list]
def squared(list):
    return [i ** 1/2 for i in list]
usq = square(uwind)
vsq = square(vwind)
sum  = np.add(usq,vsq)
windidx_prod = squared(sum) # u and v -> total magnitude
wind = np.array(windidx_prod)
qair = np.multiply(q,1000) # kg/kg -> g/kg

es_tempfin_prod = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(airtemp+273.15)))) # air saturation vapor pressure
es_sknfin_prod = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(skntemp+273.15)))) # sea saturation vapor pressure

rat_prh = (62.2/qair+100)*es_tempfin_prod
qskn = 62.2/(rat_prh/es_sknfin_prod-100)

# export file
for_df = np.array([time_sd,eflux,hflux,qair,qskn,airtemp,skntemp,wind])
for_df = for_df.T
for_df = np.ma.masked_where(for_df == 999999986991104, for_df)
for_df = np.ma.compress_rows(for_df)
df = pd.DataFrame(for_df, columns=['time','eflux','hflux','qair','qskn','airtemp','skntemp','wind'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/2018MERRA2interp.csv')

print('...end 2018 interpolation.')
###################################################################################################
# interpolate 2019 data #
print('start 2019 interpolation...')

## MERRA2 VARIABLES ##
time_prod = time_prod2019
lon_prod = lon_prod2019
lat_prod = lat_prod2019
q_prod = q_prod2019
airtemp_prod = airtemp_prod2019
skntemp_prod = skntemp_prod2019
uwind_prod = uwind_prod2019
vwind_prod = vwind_prod2019
eflux_prod = eflux_prod2019
hflux_prod = hflux_prod2019

## SAILDRONE VARIABLES ##
time_sd = sd2019['hours'].values
lon_sd = sd2019['lon'].values
lat_sd = sd2019['lat'].values

location_sd = np.stack((lon_sd,lat_sd),axis=1)

# get MERRA location grid
gridlocation = []
grididx = []
for idx_lon, lon in enumerate(lon_prod):
    for idx_lat,lat in enumerate(lat_prod):
        gridlocation.append([lon,lat])
        grididx.append([idx_lon,idx_lat])
gridlocation = np.asarray(gridlocation)
grididx = np.asarray(grididx)

# find the corresponding times between both MERRA2 and saildrone  files
[idx_prod,idx_sd] = solution(time_prod,time_sd)
idx_prod = np.array(idx_prod)
timeidx_prod = np.copy(time_prod[idx_prod])
efluxidx_prod = np.copy(eflux_prod[idx_prod])
hfluxidx_prod = np.copy(hflux_prod[idx_prod])
airtempidx_prod = np.copy(airtemp_prod[idx_prod])
skntempidx_prod = np.copy(skntemp_prod[idx_prod])
qidx_prod = np.copy(q_prod[idx_prod])
uwindidx_prod = np.copy(uwind_prod[idx_prod])
vwindidx_prod = np.copy(vwind_prod[idx_prod])

# get distance estimations for the MERRA2 model (has everything)
close4 = latLongL2(location_sd,gridlocation)
percentdist = []
percentdistog = []
for grid_idx, point in enumerate(location_sd):
    ind_close4 = close4[grid_idx]
    ind_gridlocations = []
    distancegrid = []
    distancegridog = []
    for ind in ind_close4:
        gridpoint = gridlocation[ind]
        ind_gridlocations.append(gridpoint)
        distancepixog =((point[0] - gridpoint[0])**2 + (point[1] - gridpoint[1])**2)**1/2
        lata, lona, latb, lonb, R = radians(gridpoint[1]),radians(gridpoint[0]), radians(point[1]), radians(point[0]), 6378.0
        lat_diff = latb-lata
        lon_diff = lonb-lona
        a = np.sin(lat_diff/2)**2+np.cos(latb)*np.cos(lata)*np.sin(lon_diff/2)**2
        c = 2*np.arctan2(a**(1/2),(1-a)**(1/2))
        distancepix = R * c
        distancegrid.append(np.divide(1,distancepix))
        distancegridog.append(np.divide(1,distancepixog))
    sumdist = np.sum(distancegrid)
    sumdistog = np.sum(distancegridog)
    percentdist.append(distancegrid/sumdist)
    percentdistog.append(distancegridog/sumdistog)
curiousdist = np.concatenate([curiousdist,percentdist],axis=0)
curiousdistog = np.concatenate([curiousdistog,percentdistog],axis=0)

# get the MERRA flux values for every saildrone time/location
eflux_prod_withsd = []
hflux_prod_withsd = []
for time_s in time_sd:
    for idx_time,time_m in enumerate(timeidx_prod):
        if time_s == time_m:
            eflux_prod_withsd.append(efluxidx_prod[idx_time])
            hflux_prod_withsd.append(hfluxidx_prod[idx_time])
eflux_prod_withsd = np.array(eflux_prod_withsd)
hflux_prod_withsd = np.array(hflux_prod_withsd)

# get the MERRA observation values for every saildrone time/location
airtemp_prod_withsd = []
q_prod_withsd = []
skntemp_prod_withsd = []
uwind_prod_withsd = []
vwind_prod_withsd = []
skntemplatlot = []
uwindtemplatlot = []
vwindtemplatlot = []
for time_s in time_sd:
    for idx_time,time_m in enumerate(timeidx_prod):
        if time_s == time_m:
            airtemp_prod_withsd.append(airtempidx_prod[idx_time])
            q_prod_withsd.append(qidx_prod[idx_time])
            skntemp_prod_withsd.append(skntempidx_prod[idx_time])
            uwind_prod_withsd.append(uwindidx_prod[idx_time])
            vwind_prod_withsd.append(vwindidx_prod[idx_time])
airtemp_prod_withsd = np.array(airtemp_prod_withsd)
q_prod_withsd = np.array(q_prod_withsd)
skntemp_prod_withsd = np.array(skntemp_prod_withsd)
uwind_prod_withsd = np.array(uwind_prod_withsd)
vwind_prod_withsd = np.array(vwind_prod_withsd)

# get longitude and latitude values
timeidx = np.arange(0,len(time_sd),1)
lat1 = []
lat2 = []
lat3 = []
lat4 = []
lon1 = []
lon2 = []
lon3 = []
lon4 = []
for k in close4:
    lat1.append(gridlocation[k[0]][1])
    lon1.append(gridlocation[k[0]][0])
    lat2.append(gridlocation[k[1]][1])
    lon2.append(gridlocation[k[1]][0])
    lat3.append(gridlocation[k[2]][1])
    lon3.append(gridlocation[k[2]][0])
    lat4.append(gridlocation[k[3]][1])
    lon4.append(gridlocation[k[3]][0])
arraylon1 = []
arraylon2 = []
arraylon3 = []
arraylon4 = []
arraylat1 = []
arraylat2 = []
arraylat3 = []
arraylat4 = []
for idx,lon in enumerate(lon1):
    arraylon1.append(find_nearest(lon_prod,lon))
    arraylon2.append(find_nearest(lon_prod,lon2[idx]))
    arraylon3.append(find_nearest(lon_prod,lon3[idx]))
    arraylon4.append(find_nearest(lon_prod,lon4[idx]))
    arraylat1.append(find_nearest(lat_prod,lat1[idx]))
    arraylat2.append(find_nearest(lat_prod,lat2[idx]))
    arraylat3.append(find_nearest(lat_prod,lat3[idx]))
    arraylat4.append(find_nearest(lat_prod,lat4[idx]))  
latlon1 = np.stack((arraylat1,arraylon1),axis=1)
latlon2 = np.stack((arraylat2,arraylon2),axis=1)   
latlon3 = np.stack((arraylat3,arraylon3),axis=1)   
latlon4 = np.stack((arraylat4,arraylon4),axis=1)

# get the closest 4 values
timeidx = np.arange(0,len(time_sd),1)
efluxlatlon1 = []
hfluxlatlon1 = []
efluxlatlon2 = []
hfluxlatlon2 = []
efluxlatlon3 = []
hfluxlatlon3 = []
efluxlatlon4 = []
hfluxlatlon4 = []
airtemplatlon1 = []
airtemplatlon2 = []
airtemplatlon3 = []
airtemplatlon4 = []
qlatlon1 = []
qlatlon2 = []
qlatlon3 = []
qlatlon4 = []
skntemplatlot1 = []
skntemplatlot2 = []
skntemplatlot3 = []
skntemplatlot4 = []
uwindtemplatlot1 = []
uwindtemplatlot2 = []
uwindtemplatlot3 = []
uwindtemplatlot4 = []
vwindtemplatlot1 = []
vwindtemplatlot2 = []
vwindtemplatlot3 = []
vwindtemplatlot4 = []
for j in timeidx:
    efluxlatlon1.append(eflux_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    hfluxlatlon1.append(hflux_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    efluxlatlon2.append(eflux_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    hfluxlatlon2.append(hflux_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    efluxlatlon3.append(eflux_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    hfluxlatlon3.append(hflux_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    efluxlatlon4.append(eflux_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    hfluxlatlon4.append(hflux_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    airtemplatlon1.append(airtemp_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    qlatlon1.append(q_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    skntemplatlot1.append(skntemp_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    uwindtemplatlot1.append(uwind_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    vwindtemplatlot1.append(vwind_prod_withsd[j,latlon1[j,0],latlon1[j,1]])
    airtemplatlon2.append(airtemp_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    qlatlon2.append(q_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    skntemplatlot2.append(skntemp_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    uwindtemplatlot2.append(uwind_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    vwindtemplatlot2.append(vwind_prod_withsd[j,latlon2[j,0],latlon2[j,1]])
    airtemplatlon3.append(airtemp_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    qlatlon3.append(q_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    skntemplatlot3.append(skntemp_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    uwindtemplatlot3.append(uwind_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    vwindtemplatlot3.append(vwind_prod_withsd[j,latlon3[j,0],latlon3[j,1]])
    airtemplatlon4.append(airtemp_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    qlatlon4.append(q_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    skntemplatlot4.append(skntemp_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    uwindtemplatlot4.append(uwind_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
    vwindtemplatlot4.append(vwind_prod_withsd[j,latlon4[j,0],latlon4[j,1]])
efluxlatlon1 = np.array(efluxlatlon1)
hfluxlatlon1 = np.array(hfluxlatlon1)
efluxlatlon2 = np.array(efluxlatlon2)
hfluxlatlon2 = np.array(hfluxlatlon2)
efluxlatlon3 = np.array(efluxlatlon3)
hfluxlatlon3 = np.array(hfluxlatlon3)
efluxlatlon4 = np.array(efluxlatlon4)
hfluxlatlon4 = np.array(hfluxlatlon4)
airtemplatlon1 = np.array(airtemplatlon1)
qlatlon1 = np.array(qlatlon1)
skntemplatlot1 = np.array(skntemplatlot1)
uwindtemplatlot1 = np.array(uwindtemplatlot1)
vwindtemplatlot1 = np.array(vwindtemplatlot1)
airtemplatlon2 = np.array(airtemplatlon2)
qlatlon2 = np.array(qlatlon2)
skntemplatlot2 = np.array(skntemplatlot2)
uwindtemplatlot2 = np.array(uwindtemplatlot2)
vwindtemplatlot2 = np.array(vwindtemplatlot2)
airtemplatlon3 = np.array(airtemplatlon3)
qlatlon3 = np.array(qlatlon3)
skntemplatlot3 = np.array(skntemplatlot3)
uwindtemplatlot3 = np.array(uwindtemplatlot3)
vwindtemplatlot3 = np.array(vwindtemplatlot3)
airtemplatlon4 = np.array(airtemplatlon4)
qlatlon4 = np.array(qlatlon4)
skntemplatlot4 = np.array(skntemplatlot4)
uwindtemplatlot4 = np.array(uwindtemplatlot4)
vwindtemplatlot4 = np.array(vwindtemplatlot4)
percentdist = np.array(percentdist)

# get weighted average
eflux = efluxlatlon1*percentdist[:,0] + efluxlatlon2*percentdist[:,1] + efluxlatlon3*percentdist[:,2] + efluxlatlon4*percentdist[:,3]
hflux = hfluxlatlon1*percentdist[:,0] + hfluxlatlon2*percentdist[:,1] + hfluxlatlon3*percentdist[:,2] + hfluxlatlon4*percentdist[:,3]
airtemp = airtemplatlon1*percentdist[:,0] + airtemplatlon2*percentdist[:,1] + airtemplatlon3*percentdist[:,2] + airtemplatlon4*percentdist[:,3]
skntemp = skntemplatlot1*percentdist[:,0] + skntemplatlot2*percentdist[:,1] + skntemplatlot3*percentdist[:,2] + skntemplatlot4*percentdist[:,3]
q = qlatlon1*percentdist[:,0] + qlatlon2*percentdist[:,1] + qlatlon3*percentdist[:,2] + qlatlon4*percentdist[:,3]
uwind = uwindtemplatlot1*percentdist[:,0] + uwindtemplatlot2*percentdist[:,1] + uwindtemplatlot3*percentdist[:,2] + uwindtemplatlot4*percentdist[:,3]
vwind = vwindtemplatlot1*percentdist[:,0] + vwindtemplatlot2*percentdist[:,1] + vwindtemplatlot3*percentdist[:,2] + vwindtemplatlot4*percentdist[:,3]

# convert units
airtemp = np.subtract(airtemp,273.15) # K -> C
airtemp = np.array(airtemp)
skntemp = np.subtract(skntemp,273.15) # K -> C
skntemp = np.array(skntemp)
def square(list):
    return [i ** 2 for i in list]
def squared(list):
    return [i ** 1/2 for i in list]
usq = square(uwind)
vsq = square(vwind)
sum  = np.add(usq,vsq)
windidx_prod = squared(sum) # u and v -> total magnitude
wind = np.array(windidx_prod)
qair = np.multiply(q,1000) # kg/kg -> g/kg

es_tempfin_prod = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(airtemp+273.15)))) # air saturation vapor pressure
es_sknfin_prod = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/(skntemp+273.15)))) # sea saturation vapor pressure

rat_prh = (62.2/qair+100)*es_tempfin_prod
qskn = 62.2/(rat_prh/es_sknfin_prod-100)

# export file
for_df = np.array([time_sd,eflux,hflux,qair,qskn,airtemp,skntemp,wind])
for_df = for_df.T
for_df = np.ma.masked_where(for_df == 999999986991104, for_df)
for_df = np.ma.compress_rows(for_df)
df = pd.DataFrame(for_df, columns=['time','eflux','hflux','qair','qskn','airtemp','skntemp','wind'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/2019MERRA2interp.csv')

print('...end 2019 interpolation.')

# curiosities
for_df = np.array([curiousdistog[:,0],curiousdistog[:,1],curiousdistog[:,2],curiousdistog[:,3],curiousdist[:,0],curiousdist[:,1],curiousdist[:,2],curiousdist[:,3]])
df = pd.DataFrame(for_df.T, columns=['pixel1','pixel2','pixel3','pixel4','curve1','curve2','curve3','curve4'])
df.to_csv('/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/distobservations.csv')