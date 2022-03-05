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




def phydrus_n(n=2.9,Ks=25,Qs=0.7,Alpha=0.07):
    
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
        return [0,0]

    Log_Theta =0
    for i in range(len(y)):
        Log_Theta += (-np.log(2*np.pi*sigma**2)-(float(data[3][i])-y[i])**2/((2*sigma**2)))

    return(Log_Theta)


# In[ ]:



before = dt.datetime.now()

NChain = 10000   #Iteration number
par_num = 2  #Number of parameters to search

Chain = NChain* [par_num*[0]] #Initialize chain

#print(Chain)
n = 2.5
Ns = NChain * [0]
Likelihoods = NChain *[0]
Ns[0] = n
Kss = NChain * [0]
ks = 50
Kss[0] = ks
Chain[0] = [n, ks]                #Input initial parameter values

Recent = tuple(Chain[0])
Sigma_Theta = [0.2,4]           #Parameter Jump sizes/ initial distribution width
sigma = [3,1]
AcceptCnt = 0
Lower_Bound = [1,1]
Upper_Bound = [3.8,100]

for n in range(1,(NChain)):
    
    print('Iteration: %s' %n)
    Theta = Chain[n-1]
    ThetaTest= list(Recent)
    Prior_Ratio = 1
    
    for p in range(par_num):
        if (Theta[p]-Sigma_Theta[p])<Lower_Bound[p]:
            ThetaTest[p] =  ThetaTest[p] + Sigma_Theta[p]*random.uniform(0,1)
        elif (Theta[p]+Sigma_Theta[p])>Upper_Bound[p]:
            ThetaTest[p] =  ThetaTest[p] + Sigma_Theta[p]*random.uniform(-1,0)
        else:
            ThetaTest[p] =  ThetaTest[p] + Sigma_Theta[p]*random.uniform(-1,1)   #Theta Test takes previous value and randomly changes one parameter
        Prior_Ratio *= Theta[p]/ThetaTest[p]


    if n == 1:
        LogTheta = phydrus_n(n=Theta[0], Ks=Theta[1])        #For first run, Phydrus must be run for theta and theta test
    LogThetaTest = phydrus_n(n=ThetaTest[0], Ks=ThetaTest[1])    #For all runs, Phydrus must be run for theta test
    if len(LogTheta) ==2:
        Chain[n] = Theta
    else:
        Likelihood_Ratio = np.exp(LogThetaTest-LogTheta)
        Alpha = min(1,Prior_Ratio * Likelihood_Ratio)
        Likelihoods[n] = Alpha
        if Alpha > random.random():
            Chain[n] = ThetaTest
            AcceptCnt += 1
            LogTheta = LogThetaTest
        else:
            Chain[n] = Theta
            
    Recent = tuple(Chain[n])
    Ns[n] = Chain[n][0]
    Kss[n] = Chain[n][1]
    
AcceptRate = AcceptCnt/NChain 
AccptRt = 'Acceptance Rate: %s'%AcceptRate
print(AccptRt)

n = np.mean(Ns[50:])
ks = np.mean(Kss[50:])
n_mean = 'n = %s'%n
print(n_mean)
ks_mean = 'ks = %s'%ks
print(ks_mean)

N = np.mean(Ns)
Ks = np.mean(Kss)
#print('n = %s'%N)
#print('Ks = %s'%Ks)
after = dt.datetime.now()

time = after-before
print(time)
RunTime = 'Run Time = %s'%time


date_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_name = './save/BayesFit_Run_'+date_time
os.mkdir(dir_name)
text = AccptRt +'\n' + n_mean +'\n' + ks_mean +'\n' + RunTime +'\n'
text += 'Number of Iterations = %s \n'%NChain+'Initial Conditions = %s'%Chain[0]+'\n Sigmas = %s'%Sigma_Theta
summary_file = open(dir_name+'/bayes_results.txt','w')
summary_file.write(text)
summary_file.close()
string_file = open(dir_name+'/bayes_chain.txt','w')
string_file.write('Chain:\n'+ str(Chain)+'Alphas:\n'+str(Likelihoods))
string_file.close()

fig1, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0, 0].set_title("n Histogram")
axs[0, 0].hist(Ns, bins = 50, color='C0')
axs[0, 0].set_ylabel("Number")


axs[1, 0].set_title('Ks[100:] Histogram')
axs[1, 0].hist(Kss, bins = 50, color='C2')
axs[1, 0].set_ylabel('Number')




axs[0, 1].set_title('n Iterations')
axs[0, 1].plot(Ns, color = 'C0')
axs[0, 1].set_ylabel('n')
axs[0, 1].set_xlabel('Iteration #')

Tdat = data[0]

axs[1, 1].set_title('Ks Iterations')
axs[1, 1].plot(Kss, color = 'C2')
axs[1, 1].set_ylabel('Ks')
axs[1, 1].set_xlabel('Iteration #')


fig1.tight_layout()

plt.savefig(dir_name+'/Par_Sum.png', bbox_inches='tight')
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot()
  
ax3.set_title('Parameter Space')
ax3.set_ylabel('Ks')
ax3.set_xlabel('n')

x = np.linspace(-1,1,NChain)
c = np.tan(x)
im = ax3.scatter(Ns, Kss, c = c, marker = '.')
cbar = plt.colorbar(im)
cbar.set_label('Iteration #')
               
plt.savefig(dir_name+'/Par_Map.png', bbox_inches='tight')
plt.show()

