# import packages
import numpy as np
import pandas as pd

## export a SD file with ... ##
# time, lon, lat, QL, QS, qs, qa, SST, T, V
sd0 = pd.read_csv('data/sd/avg-at30-2017.csv')

p = sd0['p'] # hPa
T = sd0['T']+273.15 # C to K
SST = sd0['SST']+273.15 # C to K
RH = sd0['RH'] # %

# conversions for sea
# sea saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
es = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/SST))) #hPa
e = es * (RH/100.0) # hPa
qs = 0.622*(e/(p-e))*1000 #g/kg

# conversions for air
es = 6.11*np.exp((2.50*10**6*18.016/(1000*8.3145))*((1/273)-(1/T))) # air saturation vapor pressure : Wallace and Hobbs, Second Edition (pg. 99)
e = es * (RH/100.0) # vapor pressure : Wallace and Hobbs, Second Edition (pg. 82)
qa = 0.622*(e/(p-e))*1000 # specific humidity of atmosphere : Wallace and Hobbs (pg. 80)

t = sd0['datetime']
lon = sd0['longitude']
lat = sd0['latitude']
QL = sd0['QL']
QS = sd0['QS']
V = sd0['V']

sd = np.array([t,lon,lat, QL,qs,qa,QS,SST,T,V,p])
sd = pd.DataFrame(sd.T,columns=['datetime','lon','lat','QL','qs','qa','QS','SST','T','V','p'])
sd.to_csv('data/sd/sd-for-m2-2017.csv', encoding='utf-8', index=False)
