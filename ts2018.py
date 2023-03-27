#  2018 Time Series
# Subhatra Sivam
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# functions
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# paths
sd = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/Saildrone/SD2018avghr.csv'

merra2prod = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/2018MERRA2interp.csv'

filepath = '/Users/subhatrasivam/Documents/Internships/NOAA/Code/MERRA2/plots/2018/'

# graph set up
font = {'family':'serif','size': 30}

xfit = np.linspace(-1000, 1000, 100)
yfit = xfit*0

stdint = [-3,-2,-1,0,1,2,3]

col_ls_range = ['#005F73','#0A9396','#94D2BD','#E9D8A6']
col_ss_range = ['#AE2012','#BB3E03','#CA6702','#EE9B00']

# import SD data
sd = pd.read_csv(sd)

time_sd = sd['hours'].values
lon_sd = sd['lon'].values
lat_sd = sd['lat'].values
eflux_sd = sd['eflux'].values
hflux_sd = sd['hflux'].values
qair_sd = sd['qair'].values
qskn_sd = sd['qskn'].values
airtemp_sd = sd['airtemp'].values
skntemp_sd = sd['skntemp'].values
wind_sd = sd['wind'].values
sp_sd = sd['sp'].values

# import MERRA2 data
merra2prod = pd.read_csv(merra2prod)

timem2 = merra2prod['time']
efluxm2 = merra2prod['eflux'].values
hfluxm2 = merra2prod['hflux'].values
qairm2 = merra2prod['qair'].values
qsknm2 = merra2prod['qskn'].values
airtempm2 = merra2prod['airtemp'].values
skntempm2 = merra2prod['skntemp'].values
windm2 = merra2prod['wind'].values

# extract SD values for MERRA2 values
time_sdm2 = []
lon_sdm2 = []
lat_sdm2 = []
eflux_sdm2 = []
hflux_sdm2 = []
qair_sdm2 = []
qskn_sdm2 = []
airtemp_sdm2 = []
skntemp_sdm2 = []
wind_sdm2 = []
sp_sdm2 = []
for timeprod in timem2:
    k = find_nearest(time_sd,timeprod)
    time_sdm2.append(time_sd[k])
    lon_sdm2.append(lon_sd[k])
    lat_sdm2.append(lat_sd[k])
    eflux_sdm2.append(eflux_sd[k])
    hflux_sdm2.append(hflux_sd[k])
    qair_sdm2.append(qair_sd[k])
    qskn_sdm2.append(qskn_sd[k])
    airtemp_sdm2.append(airtemp_sd[k])
    skntemp_sdm2.append(skntemp_sd[k])
    wind_sdm2.append(wind_sd[k])
    sp_sdm2.append(sp_sd[k])

# difference variables
efluxdiff = efluxm2 - eflux_sdm2
meane = np.mean(efluxdiff)
stde = np.std(efluxdiff)
hfluxdiff = hfluxm2 - hflux_sdm2
meanh = np.mean(hfluxdiff)
stdh = np.std(hfluxdiff)

qairdiff = qairm2 - qair_sdm2
qskndiff = qsknm2 - qskn_sdm2
qdiff = qskndiff - qairdiff

airtempdiff = airtempm2 - airtemp_sdm2
skntempdiff = skntempm2 - skntemp_sdm2
tempdiff = skntempdiff - airtempdiff

winddiff = windm2 - wind_sdm2

# time labels
days_sd = np.divide(time_sdm2,24)
monthnum = np.array([(5,6,7,8,9,10),(31,30,31,31,30,31)])
monthdiff = [0,32,62,93,124,154]
month_sd = []
for days in days_sd:
    if days < 32:
        month_sd.append(5)
    elif days < 62:
        month_sd.append(6)
    elif days < 93:
        month_sd.append(7)
    elif days < 124:
        month_sd.append(8)
    elif days < 154:
        month_sd.append(9)
    else:
        month_sd.append(10)
datestuff = np.array([month_sd,days_sd])
months_sd = []
for num in monthnum[0]:
    k = 0
    for days in datestuff[0]:
        if days == num:
            daydiff = datestuff[1,k] - monthdiff[num-5]
            date = daydiff/monthnum[1,num-5]
            mondate = date+num
            months_sd.append(mondate)   
        k = k + 1
#labeldates = ['6/28','7/8','7/18','7/28','8/7','8/17','8/27','9/6','9/16','9/26','10/6']
labeldates = ['9/26','9/28','9/30','10/2','10/4','10/6']
splitdate = []
monthlabel = []
for d in labeldates:
    sepdate = d.split('/')
    splitdate.append(sepdate)
for k in splitdate:
    monthlabelint = int(k[0])
    for ll in monthnum[0]:
        if ll == monthlabelint:
            dayflt = float(k[1])/monthnum[1,ll-5]
            monthlabel.append(monthlabelint+dayflt)

# saildrone latent heat flux
fig1 = plt.figure(figsize=(21,18))
plt.plot(months_sd,eflux_sdm2,'.',c='k',markersize=8)
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('Saildrone Latent Heat Flux in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(eflux_sdm2)/10)*10
ymaxsen = math.ceil(np.max(eflux_sdm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_L$ (W*m$^{-2}$)')
plt.rc('font', **font)
filesavename = filepath +'SD_eflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False,bbox_inches='tight')

# saildrone sensible heat flux
fig1 = plt.figure(figsize=(21,18))
plt.plot(months_sd,hflux_sdm2,'.',c='k',markersize=8)
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('Saildrone Sensible Heat Flux in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(hflux_sdm2)/10)*10
ymaxsen = math.ceil(np.max(hflux_sdm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_S$ (W*m$^{-2}$)')
plt.rc('font', **font)
filesavename = filepath +'SD_hflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False,bbox_inches='tight')

# MERRA2 latent heat flux
fig1 = plt.figure(figsize=(21,18))
plt.plot(months_sd,efluxm2,'.',c='k',markersize=8)
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('MERRA2 Latent Heat Flux in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(efluxm2)/10)*10
ymaxsen = math.ceil(np.max(efluxm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_L$ (W*m$^{-2}$)')
plt.rc('font', **font)
filesavename = filepath +'M2_eflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False,bbox_inches='tight')

# MERRA2 sensible heat flux
fig1 = plt.figure(figsize=(21,18))
plt.plot(months_sd,hfluxm2,'.',c='k',markersize=8)
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('MERRA2 Sensible Heat Flux in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(hfluxm2)/10)*10
ymaxsen = math.ceil(np.max(hfluxm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_S$ (W*m$^{-2}$)')
plt.rc('font', **font)
filesavename = filepath +'M2_hflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False,bbox_inches='tight')

# MERRA2-SD latent heat flux
fig1 = plt.figure(figsize=(21,18))
plt.plot(xfit,yfit,'--',c='k',lw = 1)
plt.plot(months_sd,efluxdiff,'.',c='k',markersize=8)
avg = yfit + meane
mean_val = 'Mean: '+'{:.2f}'.format(meane) + ' W*m$^{-2}$'
plt.plot(xfit,avg,':',c=col_ls_range[0],lw=3,label=mean_val)
threeplus = np.ma.masked_where(efluxdiff<meane+stde*3,efluxdiff)
threemin = np.ma.masked_where(efluxdiff>meane+stde*-3,efluxdiff)
plt.plot(months_sd,threeplus,'.',c=col_ls_range[3],markersize=8)
plt.plot(months_sd,threemin,'.',c=col_ls_range[3],markersize=8)
twoplus = np.ma.masked_where(np.logical_or(efluxdiff>=meane+stde*3,efluxdiff<meane+stde*2),efluxdiff)
twomin = np.ma.masked_where(np.logical_or(efluxdiff<=meane+stde*-3,efluxdiff>meane+stde*-2),efluxdiff)
plt.plot(months_sd,twoplus,'.',c=col_ls_range[2],markersize=8)
plt.plot(months_sd,twomin,'.',c=col_ls_range[2],markersize=8)
oneplus = np.ma.masked_where(np.logical_or(efluxdiff>=meane+stde*2,efluxdiff<meane+stde),efluxdiff)
onemin = np.ma.masked_where(np.logical_or(efluxdiff<=meane+stde*-2,efluxdiff>meane-stde),efluxdiff)
plt.plot(months_sd,oneplus,'.',c=col_ls_range[1],markersize=8)
plt.plot(months_sd,onemin,'.',c=col_ls_range[1],markersize=8)
for stdi in stdint:
    stdval = meane+stde*stdi
    if abs(stdi) == 3:
        if stdi < 0:
            stdlabel = ''
        else:
            stdlabel = '3$\sigma$'
        stdcolor = col_ls_range[3]
    elif abs(stdi) == 2:
        if stdi < 0:
            stdlabel = ''
        else:
            stdlabel = '2$\sigma$'
        stdcolor = col_ls_range[2]
    elif abs(stdi) == 1:
        stdcolor = col_ls_range[1]
        if stdi < 0:
            stdlabel = ''
        else:
            stdlabel = '1$\sigma$'
    elif abs(stdi) == 0:
        stdcolor = col_ls_range[0]
    plt.axhline(y = stdval,xmin = -1000,xmax = 1000,color = stdcolor,label = stdlabel,linestyle='dotted',lw = 3)
plt.title('MERRA2-Saildrone Latent Heat Flux Difference in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(efluxdiff)/10)*10
ymaxsen = math.ceil(np.max(efluxdiff)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('$\Delta$Q$_L$ (W*m$^{-2}$)')
plt.rc('font', **font)
plt.legend()
filesavename = filepath +'M2diff_eflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False,bbox_inches='tight')

# MERRA2-SD latent heat flux
fig1 = plt.figure(figsize=(21,18))
plt.plot(xfit,yfit,'--',c='k',lw = 1)
plt.plot(months_sd,hfluxdiff,'.',c='k',markersize=8)
avg = yfit + meanh
mean_val = 'Mean: '+'{:.2f}'.format(meanh) + ' W*m$^{-2}$'
plt.plot(xfit,avg,':',c=col_ss_range[0],lw=3,label=mean_val)
threeplus = np.ma.masked_where(hfluxdiff<meanh+stdh*3,hfluxdiff)
threemin = np.ma.masked_where(hfluxdiff>meanh+stdh*-3,hfluxdiff)
plt.plot(months_sd,threeplus,'.',c=col_ss_range[3],markersize=8)
plt.plot(months_sd,threemin,'.',c=col_ss_range[3],markersize=8)
twoplus = np.ma.masked_where(np.logical_or(hfluxdiff>=meanh+stdh*3,hfluxdiff<meanh+stdh*2),hfluxdiff)
twomin = np.ma.masked_where(np.logical_or(hfluxdiff<=meanh+stdh*-3,hfluxdiff>meanh+stdh*-2),hfluxdiff)
plt.plot(months_sd,twoplus,'.',c=col_ss_range[2],markersize=8)
plt.plot(months_sd,twomin,'.',c=col_ss_range[2],markersize=8)
oneplus = np.ma.masked_where(np.logical_or(hfluxdiff>=meanh+stdh*2,hfluxdiff<meanh+stdh),hfluxdiff)
onemin = np.ma.masked_where(np.logical_or(hfluxdiff<=meanh+stdh*-2,hfluxdiff>meanh-stdh),hfluxdiff)
plt.plot(months_sd,oneplus,'.',c=col_ss_range[1],markersize=8)
plt.plot(months_sd,onemin,'.',c=col_ss_range[1],markersize=8)
for stdi in stdint:
    stdval = meanh+stdh*stdi
    if abs(stdi) == 3:
        if stdi < 0:
            stdlabel = ''
        else:
            stdlabel = '3$\sigma$'
        stdcolor = col_ss_range[3]
    elif abs(stdi) == 2:
        if stdi < 0:
            stdlabel = ''
        else:
            stdlabel = '2$\sigma$'
        stdcolor = col_ss_range[2]
    elif abs(stdi) == 1:
        stdcolor = col_ss_range[1]
        if stdi < 0:
            stdlabel = ''
        else:
            stdlabel = '1$\sigma$'
    elif abs(stdi) == 0:
        stdcolor = col_ss_range[0]
    plt.axhline(y = stdval,xmin = -1000,xmax = 1000,color = stdcolor,label = stdlabel,linestyle='dotted',lw = 3)
plt.title('MERRA2-Saildrone Sensible Heat Flux Difference in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(hfluxdiff)/10)*10
ymaxsen = math.ceil(np.max(hfluxdiff)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('$\Delta$Q$_S$ (W*m$^{-2}$)')
plt.rc('font', **font)
plt.legend()
filesavename = filepath +'M2diff_hflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False,bbox_inches='tight')

# Sorted SD latent heat flux
fig1 = plt.figure(figsize=(22,17))
stdmore = []
std3 = []
std2 = []
std1 = []
for idx,diff in enumerate(efluxdiff):
    if diff < meane+stde*-3 or diff > meane+stde*3:
        stdmore.append(idx)
    elif diff < meane+stde*-2 or diff > meane+stde*2:
        std3.append(idx)
    elif diff < meane+stde*-1 or diff > meane+stde*1:
        std2.append(idx)
    else:
        std1.append(idx)
stdmore = np.asarray(stdmore)
std3 = np.asarray(std3)
std2 = np.asarray(std2)
std1 = np.asarray(std1)
monthmore = []
efluxmore = []
month3 = []
eflux3 = []
month2 = []
eflux2 = []
month1 = []
eflux1 = []
for std in stdmore:
    monthmore.append(months_sd[std])
    efluxmore.append(eflux_sdm2[std])
for std in std3:
    month3.append(months_sd[std])
    eflux3.append(eflux_sdm2[std])
for std in std2:
    month2.append(months_sd[std])
    eflux2.append(eflux_sdm2[std])
for std in std1:
    month1.append(months_sd[std])
    eflux1.append(eflux_sdm2[std])
plt.plot(month1,eflux1,'.',c='k',markersize=8,label='<1$\sigma$')
plt.plot(month2,eflux2,'.',c=col_ls_range[1],markersize=8,label='1$\sigma$ to 2$\sigma$')
plt.plot(month3,eflux3,'.',c=col_ls_range[2],markersize=8,label='2$\sigma$ to 3$\sigma$')
plt.plot(monthmore,efluxmore,'.',c=col_ls_range[3],markersize=8,label='>3$\sigma$')
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('Saildrone Latent Heat Flux Sorted by MERRA2-Saildrone $\sigma$ in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(eflux_sdm2)/10)*10
ymaxsen = math.ceil(np.max(eflux_sdm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_L$ (W*m$^{-2}$)')
plt.legend()
filesavename = filepath +'SDsort_eflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False)

# Sorted SD sensible heat flux
fig1 = plt.figure(figsize=(22,17))
stdmore = []
std3 = []
std2 = []
std1 = []
for idx,diff in enumerate(hfluxdiff):
    if diff < meanh+stdh*-3 or diff > meanh+stdh*3:
        stdmore.append(idx)
    elif diff < meanh+stdh*-2 or diff > meanh+stdh*2:
        std3.append(idx)
    elif diff < meanh+stdh*-1 or diff > meanh+stdh*1:
        std2.append(idx)
    else:
        std1.append(idx)
stdmore = np.asarray(stdmore)
std3 = np.asarray(std3)
std2 = np.asarray(std2)
std1 = np.asarray(std1)
monthmore = []
hfluxmore = []
month3 = []
hflux3 = []
month2 = []
hflux2 = []
month1 = []
hflux1 = []
for std in stdmore:
    monthmore.append(months_sd[std])
    hfluxmore.append(hflux_sdm2[std])
for std in std3:
    month3.append(months_sd[std])
    hflux3.append(hflux_sdm2[std])
for std in std2:
    month2.append(months_sd[std])
    hflux2.append(hflux_sdm2[std])
for std in std1:
    month1.append(months_sd[std])
    hflux1.append(hflux_sdm2[std])
plt.plot(month1,hflux1,'.',c='k',markersize=8,label='<1$\sigma$')
plt.plot(month2,hflux2,'.',c=col_ss_range[1],markersize=8,label='1$\sigma$ to 2$\sigma$')
plt.plot(month3,hflux3,'.',c=col_ss_range[2],markersize=8,label='2$\sigma$ to 3$\sigma$')
plt.plot(monthmore,hfluxmore,'.',c=col_ss_range[3],markersize=8,label='>3$\sigma$')
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('Saildrone Sensible Heat Flux Sorted by MERRA2-Saildrone $\sigma$ in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(hflux_sdm2)/10)*10
ymaxsen = math.ceil(np.max(hflux_sdm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_S$ (W*m$^{-2}$)')
plt.legend()
filesavename = filepath +'SDsort_hflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False)

# Sorted MERRA2 latent heat flux
fig1 = plt.figure(figsize=(22,17))
stdmore = []
std3 = []
std2 = []
std1 = []
for idx,diff in enumerate(efluxdiff):
    if diff < meane+stde*-3 or diff > meane+stde*3:
        stdmore.append(idx)
    elif diff < meane+stde*-2 or diff > meane+stde*2:
        std3.append(idx)
    elif diff < meane+stde*-1 or diff > meane+stde*1:
        std2.append(idx)
    else:
        std1.append(idx)
stdmore = np.asarray(stdmore)
std3 = np.asarray(std3)
std2 = np.asarray(std2)
std1 = np.asarray(std1)
monthmore = []
efluxmore = []
month3 = []
eflux3 = []
month2 = []
eflux2 = []
month1 = []
eflux1 = []
for std in stdmore:
    monthmore.append(months_sd[std])
    efluxmore.append(efluxm2[std])
for std in std3:
    month3.append(months_sd[std])
    eflux3.append(efluxm2[std])
for std in std2:
    month2.append(months_sd[std])
    eflux2.append(efluxm2[std])
for std in std1:
    month1.append(months_sd[std])
    eflux1.append(efluxm2[std])
plt.plot(month1,eflux1,'.',c='k',markersize=8,label='<1$\sigma$')
plt.plot(month2,eflux2,'.',c=col_ls_range[1],markersize=8,label='1$\sigma$ to 2$\sigma$')
plt.plot(month3,eflux3,'.',c=col_ls_range[2],markersize=8,label='2$\sigma$ to 3$\sigma$')
plt.plot(monthmore,efluxmore,'.',c=col_ls_range[3],markersize=8,label='>3$\sigma$')
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('MERRA2 Latent Heat Flux Sorted by MERRA2-Saildrone $\sigma$ in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(efluxm2)/10)*10
ymaxsen = math.ceil(np.max(efluxm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_L$ (W*m$^{-2}$)')
plt.legend()
filesavename = filepath +'M2sort_eflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False)

# Sorted MERRA2 sensible heat flux
fig1 = plt.figure(figsize=(22,17))
stdmore = []
std3 = []
std2 = []
std1 = []
for idx,diff in enumerate(hfluxdiff):
    if diff < meanh+stdh*-3 or diff > meanh+stdh*3:
        stdmore.append(idx)
    elif diff < meanh+stdh*-2 or diff > meanh+stdh*2:
        std3.append(idx)
    elif diff < meanh+stdh*-1 or diff > meanh+stdh*1:
        std2.append(idx)
    else:
        std1.append(idx)
stdmore = np.asarray(stdmore)
std3 = np.asarray(std3)
std2 = np.asarray(std2)
std1 = np.asarray(std1)
monthmore = []
hfluxmore = []
month3 = []
hflux3 = []
month2 = []
hflux2 = []
month1 = []
hflux1 = []
for std in stdmore:
    monthmore.append(months_sd[std])
    hfluxmore.append(hfluxm2[std])
for std in std3:
    month3.append(months_sd[std])
    hflux3.append(hfluxm2[std])
for std in std2:
    month2.append(months_sd[std])
    hflux2.append(hfluxm2[std])
for std in std1:
    month1.append(months_sd[std])
    hflux1.append(hfluxm2[std])
plt.plot(month1,hflux1,'.',c='k',markersize=8,label='<1$\sigma$')
plt.plot(month2,hflux2,'.',c=col_ss_range[1],markersize=8,label='1$\sigma$ to 2$\sigma$')
plt.plot(month3,hflux3,'.',c=col_ss_range[2],markersize=8,label='2$\sigma$ to 3$\sigma$')
plt.plot(monthmore,hfluxmore,'.',c=col_ss_range[3],markersize=8,label='>3$\sigma$')
plt.plot(xfit,yfit,':',c='k',lw = 1)
plt.title('MERRA2 Sensible Heat Flux Sorted by MERRA2-Saildrone $\sigma$ in 2018')
plt.rc('font', **font)
plt.xlim(np.min(months_sd)-0.1,np.max(months_sd)+0.1)
plt.xticks(monthlabel,labeldates)
plt.xlabel('Dates in 2018')
yminsen = math.floor(np.min(hfluxm2)/10)*10
ymaxsen = math.ceil(np.max(hfluxm2)/10)*10+10
plt.ylim(yminsen,ymaxsen)
plt.yticks(np.arange(yminsen,ymaxsen,step=10))
plt.ylim(yminsen,ymaxsen)
plt.ylabel('Q$_S$ (W*m$^{-2}$)')
plt.legend()
filesavename = filepath +'M2sort_hflux.png'
fig1.savefig(filesavename,facecolor='white',transparent=False)