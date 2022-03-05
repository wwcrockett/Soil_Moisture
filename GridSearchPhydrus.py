#!/usr/bin/env python
# coding: utf-8
"""  
           Phydrus Bayes MCMC SES 2cm Fit Pipeline:

- Run module 1 to import packages and set matplotlib.pyplot preferences.
- Run module 2 to read in data to pandas dataframe 'data_array' and convert date time strings to datetime objects
- Input desired start and end datetimes in module 3, then run to select time window of data. 
- Module 4 contains the function phydrus that outputs the model for the given parameters
- Module 5 contains the function phydrus_n which returns the LogTheta values for the given parameters
- Module 6 executes the Bayes MCMC, iterating through and using the phydrus_n function.

"""

# In[1]:


# ------ Module 1: import libraries --------

import pandas as pd
import os
import numpy as np
import re
import datetime as dt
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import phydrus as ps
import random
import logging

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[2]:


#---------- Module 2: Data Import -------------

#Setup datetime objects for start and end timestamps- 
#Goes through entire half hour data set, so only needs to be run once.
before = dt.datetime.now()

data_array=pd.read_csv('./read/Ses_halfhourly.csv', parse_dates=True, sep=',', header=0, 
            usecols=['TIMESTAMP_START', 'TIMESTAMP_END', 'P', 'SWC_2cm_gf', 'SWC_12cm_gf', 'SWC_22cm_gf', 'SWC_37cm_gf', 'SWC_52cm_gf'], )


data_array['TIMESTAMP_START']=pd.to_datetime(data_array['TIMESTAMP_START'])
data_array['TIMESTAMP_END']=pd.to_datetime(data_array['TIMESTAMP_END'])
    
after = dt.datetime.now()

seconds = (after-before).seconds
microseconds = (after-before).microseconds
speed = int(seconds) + float(microseconds/10**5)
print(speed)


# In[3]:


#--------- Module 3: Define Time Window ----------

#Defines precipitation times in ATMOSPH.IN file and sets up print times.

#------------INPUT DATETIMES FOR WINDOW--------------------------------------------------------------------
start=dt.datetime(2008,8,1,0,0,0)
end  =dt.datetime(2008,9,1,0,0,0)

#---------------------------------------------------------------------------------------------------------



#Make window of data based on datetime inputs
window_start=data_array[data_array['TIMESTAMP_START']>=start]
window=window_start[window_start['TIMESTAMP_START']<=end]
#Date String:

date_str= (start.strftime('%Y')+'_'+start.strftime('%m')+'_'+start.strftime('%d')+'-'+
        end.strftime('%Y')+'_'+end.strftime('%m')+'_'+end.strftime('%d'))

#Find difference in times between step and starting time
window=window.assign(Tdelta=window['TIMESTAMP_START']-start)
tdelt=[]

#Convert time difference to a float of hours
for x in window['Tdelta']:
    tdelt.append(x.total_seconds()/3600)
window=window.assign(Tdelt=tdelt)
window.reset_index(drop=True, inplace=True)

#Retrieve new precipitation events to make HYDRUS-1D boundary condition input data
P0=0.5
new_precip=[]
for x in window['P']:
    if x==P0:
        new_precip.append(False)
    else:
        new_precip.append(True)
    P0=x
window=window.assign(New_precip=new_precip)
window
PrecipSummary=window[window['New_precip']==True]
(length,width)=PrecipSummary.shape
PrecipSummary.reset_index(drop=True,inplace=True)
    
    #Change precip values from mm/half hour to cm/hour
PrecipSummary['P']=PrecipSummary['P']/20
    
#Move times up one row to fit HYDRUS
for x in range(length-1):
    PrecipSummary.loc[x,'Tdelt']=PrecipSummary.loc[x+1,'Tdelt'] 
print(length)
PrecipSummary=PrecipSummary.drop([length-1])

#Add evaporation and hCritA to dataframe
evap=np.ones(length)*0.001
hcrita=np.ones(length)*100000
PrecipSummary=PrecipSummary.append(window.iloc[-1])
PrecipSummary.reset_index(drop=True,inplace=True)
PrecipSummary=PrecipSummary.assign(Evap=evap)
PrecipSummary=PrecipSummary.assign(hCritA=hcrita)
PrecipSummary=PrecipSummary[['Tdelt', 'P', 'Evap', 'hCritA']]
    

#-----Add info to Phydrus---------------------------------------------------------------------------------------


times = tuple(PrecipSummary['Tdelt'].tolist())
Ps = tuple(PrecipSummary['P'].tolist())
Evaps = tuple(PrecipSummary['Evap'].tolist())

bc = {"tAtm": times, "Prec": Ps, "rSoil": Evaps}
atm = pd.DataFrame(bc, index=times)
    
        #-------------------------------------------------------

points= 3 #Enter number of points for rolling average

#-------------------------------------------------------


data = []
[l,w]=window.shape
for name, values in window.iteritems():
    data.append(list(values))
data1 = copy.deepcopy(data)
printpoints = int((l-l%points)/points)
diff = []
for x in range(3,8):             #Columns in data containing soil moisture at 2cm,12cm,...
    diff.append([])
    for y in range(l-points):
        dum = 0.
        for p in range(points):
            dum += data[x][y+p] 
        data[x][y]=dum/points
        diff[x-3].append(abs(data[x][y]-data1[x][y]))
        

deletables = []
for y in range(l-points,0,-1):
    if not y % points == 0:
        for x in range(len(data)):
            del data[x][y]

data[-2][0] = 0.001
    
#Adding print time info to Phydrus-----------------------------------------------------------

time = data[-2]

# In[5]:


# mat12 = -7.         #boundary between 1 and 2
# mat23 = -17.
# mat34 = -29.5
# mat45 = -44.5




def phydrus_calc(n=2.9,Ks=25,Qs=0.7,Alpha=0.07):
    
# Create Phydrus Model:

# Folder for Hydrus files to be stored
    ws = "SES_Pydrus"
# Path to folder containing hydrus.exe 
    exe = "./read/hydrus"  
# Description
    desc = "Model of Soil Moisture at SES Site"
# Create model
    ml = ps.Model(exe_name=exe, ws_name=ws, name="SES model", description=desc, 
              mass_units="mmol", time_unit="hours", length_unit="cm")
    ml.__init__(exe_name=exe, ws_name=ws, name="SES model", description=desc, 
              mass_units="mmol", time_unit="hours", length_unit="cm")
    ml.basic_info["lFlux"] = True
    ml.basic_info["lShort"] = False
    ml.add_atmospheric_bc(atm, hcrits=3, hcrita=50000)
    ml.add_time_info(tmax=data[-2][-1], print_array=time, dt=0.000001, dtmin= 0.000000001, dtmax=0.05)

# Control the soil profile depth, node count, and material divisions directly (without using HYDRUS interface)

    Depth = -80 #cm depth of column
    Nodes = 161 #Number of nodes in column
    Mat_num = 1

#--- Boundaries ---


#--- Initial Conditions ---

    Qr = min(data[3])
    #Qs = 0.7
    #Alpha = 0.07
    #n = Par[0]
    #Ks = Par[1]
    IC = data[3][0]
    sigma = 0.003
    #Add Profile Information:
    
    ml.add_waterflow(model=0, top_bc=3, bot_bc=4,linitw = True)
    m = ml.get_empty_material_df(n=Mat_num)
    m.loc[[1]] = [[Qr,Qs,Alpha, n, Ks, 0.5]]
    ml.add_material(m)

    # Create Profile
    profile = ps.create_profile(bot=Depth, dx=abs(Depth / (Nodes-1)))
    ml.add_profile(profile)  # Add the profile
    ml.add_obs_nodes([-2,-12,-22,-37,-52])

    ml.write_input()





#Write PROFILE.DAT
    prof_path = Path('./read/PROFILE.DAT')
    prof_str = prof_path.read_text()
    prof_lines=prof_str.split('\n')
    prof_lines[4]= prof_lines[4][0:2]+str(Nodes)+prof_lines[4][5:]
    del prof_lines[5:]
    end_of_str = '    1  0.000000e+000  1.000000e+000  1.000000e+000  1.000000e+000              '

    for N in range(Nodes):
        D = N * Depth/(Nodes-1)
        M = str(1)
        if N == 0:
            new_line = ' '*(5-len(str(N+1))) + str(N+1)+' -'+np.format_float_scientific(D, unique = False, precision = 6, exp_digits=3)                    + ' '+np.format_float_scientific(IC, unique = False, precision = 6, exp_digits=3)+ '    ' + M + end_of_str
        else:
            new_line = ' '*(5-len(str(N+1))) + str(N+1)+' '+np.format_float_scientific(D, unique = False, precision = 6, exp_digits=3)                    + ' '+np.format_float_scientific(IC, unique = False, precision = 6, exp_digits=3)+ '    ' + M + end_of_str
        prof_lines.append(new_line)
    
    prof_lines.append('    0\n')
    prof_str = '\n'.join(prof_lines)
    
    prof = open('./SES_Pydrus/PROFILE.DAT','w')
    prof.write(prof_str)
    prof.close()
    
    
#-----------------Run Model----------------

    ml.simulate()

#---------------Read Output----------------

    #Each output file will be named according to the workspace from which it was generated.    
    infile  = './SES_Pydrus/NOD_INF.OUT'
    outdata = []
    ttdata   = []
    with open(infile, 'r') as f:   # Hydrus data files are a mess.
            raw = f.readlines()

    for line in raw[6:]:
        if "Time:"in line:
            ttdata.append(re.split('\s+', line))
        if len(line) > 111:
            outdata.append(re.split('\s+', line)) # Data are whitespace delimited.

    df = pd.DataFrame(ttdata)
    tdata = df.to_numpy()

    df = pd.DataFrame(outdata)
    data_h = df.to_numpy()
    if df.empty:
        return
    
    nod     = pd.to_numeric(data_h[:,1], errors='coerce')
    dep     = pd.to_numeric(data_h[:,2], errors='coerce') 
    head    = pd.to_numeric(data_h[:,3], errors='coerce') 
    mois    = pd.to_numeric(data_h[:,4], errors='coerce') 
    times   = pd.to_numeric(tdata[:,2], errors='coerce')


#--------- Calculate Log_Theta -----------
    mask = (dep == -2.)
    y = mois[mask].tolist()
    x = []
    dat_len = len(data[3])
    mod_len = len(y)

    for t in times:
        x.append(start+dt.timedelta(hours=t))
    if mod_len-dat_len == 1:
        del y[0]
        del x[0]
    if mod_len-dat_len == 2:
        del y[0:2]
        del x[0:2]
    if dat_len > mod_len:
        return

    Log_Theta =[]
    for i in range(len(y)):
        Log_Theta.append(-0.5*np.log(2*np.pi*sigma**2)-(float(data[3][i])-y[i])**2/((2*sigma**2)))

    return(Log_Theta)

# In[ ]:



before = dt.datetime.now()

NChain = 1
Ns = np.linspace(1,5,NChain)
Kss = np.linspace(5,200,NChain)
Residuals = np.zeros((NChain,NChain))
storageFrame = pd.DataFrame(columns=Ns,index=Kss)
date_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_name = './save/GridSearch_Run_'+date_time
os.mkdir(dir_name)
string_file = open(dir_name+'/sum.txt','w')
string_file.write('parameters = n, Ks \n'+'Grid: 100x100 \n')
string_file.close()

for NN in Ns:
    for KK in Kss: 
        LogTheta = phydrus_calc(n=NN,Ks=KK)        #For first run, Phydrus must be run for theta and theta test
        print(NN)
        print(KK)
        print(LogTheta)
        #print(sumdi)
        storageFrame.loc[KK,NN]=LogTheta
        storageFrame.to_csv(dir_name+'/residuals.csv')

after = dt.datetime.now()

time = after-before
print(time)
RunTime = 'Run Time = %s'%time

#zmin = storageFrame.min().min()
#zmax = storageFrame.max().max()
#levels = np.linspace(zmin,zmax,30)


fig1, ax = plt.subplots()
#ticks = np.linspace(zmin, zmax, 8)
n = storageFrame.columns
ks = storageFrame.index
CS = ax.contourf(n,ks,storageFrame)#, levels=levels)
cbar = fig1.colorbar(CS,label = 'LogTheta Value')
ax.set_xlabel('n')
ax.set_ylabel('Ks')
fig1.tight_layout()
plt.savefig(dir_name+'/GridSearch.png', bbox_inches='tight')
