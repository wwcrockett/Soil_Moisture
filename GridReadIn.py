#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 08:49:57 2022

@author: wcroc
"""

""" 
This file is used to import the residual.csv file from the GridSearchPhydrus.py
Output. From here we can plot certain areas, find the max value, and plot the
 likelihood function.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
def read_in(folder_path):
    file_path = folder_path+'/residuals.csv'
    data = pd.read_csv(file_path, header=0,index_col=0)
    return data

dat = read_in('./save/GridSearch_Run_20220303-173707')
zmin = dat.min().min()
zmax = dat.max().max()
levels = np.linspace(zmin,0,10)
levels = np.append(levels,np.linspace(0,zmax,10)[1:])

fig1, ax = plt.subplots()
n = dat.columns
n = n.astype('float')
ks = dat.index
CS = ax.contourf(n,ks,dat,cmap=cm.coolwarm,levels=levels)
cbar = fig1.colorbar(CS,label = 'LogLikelihood Value')
ax.set_xlabel('n')

ax.set_ylabel('Ks')
fig1.tight_layout()
plt.show()
#plt.savefig(dir_name+'/GridSearch.png', bbox_inches='tight')
dat[dat< 0] = 0
fig2, ax2 = plt.subplots()
zmin = dat.min().min()
zmax = dat.max().max()
levels = np.linspace(zmin,zmax,30)
CS = ax2.contourf(n,ks,dat,cmap=cm.coolwarm,levels=levels)
cbar = fig2.colorbar(CS,label = 'LogLikelihood Value')
ax2.set_xlabel('n')

ax2.set_ylabel('Ks')
fig2.tight_layout()
plt.show()