# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:36:32 2022

@author: Rohini
"""

import numpy as np
from random import random
import pandas as pd
from matplotlib import pyplot as plt
from hmmlearn import hmm
import pandas as pd
from scipy import stats as ss
import seaborn as sns
import matplotlib
import utils
import os

dir=os.getcwd() 

#Static parameters 
nSites=5
nYears=105

#Import LHS Sample
LHS=np.loadtxt('./LHsamples_CO.txt')

for p in range (0,len(LHS)):
    
##################Generate flows for all basins##############################3    
  
    #Import stationary parameters 
    dry_state_means=np.loadtxt('D:/LHS_CO/dry_state_means.txt')
    wet_state_means=np.loadtxt('D:/LHS_CO/wet_state_means.txt')
    covariance_matrix_dry=np.loadtxt('D:/LHS_CO/covariance_matrix_dry.txt')
    covariance_matrix_wet=np.loadtxt('D:/LHS_CO/covariance_matrix_wet.txt')
    transition_matrix=np.loadtxt('D:/LHS_CO/transition_matrix.txt')
   
    
    #Apply mean multipliers 
    dry_state_means_sampled=dry_state_means*LHS[p,0]
    wet_state_means_sampled=wet_state_means*LHS[p,2]
    
    #Apply covariance multipliers 
    covariance_matrix_dry_sampled=covariance_matrix_dry*LHS[p,1]
    
    for j in range(5):
        covariance_matrix_dry_sampled[j,j]=covariance_matrix_dry_sampled[j,j]*LHS[p,1]
        
    covariance_matrix_wet_sampled=covariance_matrix_wet*LHS[p,3]
    
    for j in range(5):
        covariance_matrix_wet_sampled[j,j]=covariance_matrix_wet_sampled[j,j]*LHS[p,3]  
        
    #Apply transition matrix multipliers 
    transition_matrix_sampled=transition_matrix
    transition_matrix_sampled[0,0]=transition_matrix[0,0]+LHS[p,4]
    transition_matrix_sampled[1,1]=transition_matrix[1,1]+LHS[p,5]  
    transition_matrix_sampled[0,1]=1-transition_matrix_sampled[0,0]
    transition_matrix_sampled[1,0]=1-transition_matrix_sampled[1,1]
    
     

    
    # calculate stationary distribution to determine unconditional probabilities 
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(transition_matrix_sampled))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    unconditional_dry=pi[0]
    unconditional_wet=pi[1]
    
    
    logAnnualQ_s=np.zeros([nYears,nSites])
    
    
    states = np.empty([np.shape(logAnnualQ_s)[0]])
    if random() <= unconditional_dry:
        states[0] = 0
        logAnnualQ_s[0,:]=np.random.multivariate_normal(np.reshape(dry_state_means_sampled,-1),covariance_matrix_dry_sampled)
    else:
        states[0] = 1
        logAnnualQ_s[0,:]=np.random.multivariate_normal(np.reshape(wet_state_means_sampled,-1),covariance_matrix_wet_sampled)
        
    # generate remaining state trajectory and log space flows
    for j in range(1,np.shape(logAnnualQ_s)[0]):
        if random() <= transition_matrix_sampled[int(states[j-1]),int(states[j-1])]:
            states[j] = states[j-1]
        else:
            states[j] = 1 - states[j-1]
            
        if states[j] == 0:
            logAnnualQ_s[j,:] = np.random.multivariate_normal(np.reshape(dry_state_means_sampled,-1),covariance_matrix_dry_sampled)
        else:
            logAnnualQ_s[j,:] = np.random.multivariate_normal(np.reshape(wet_state_means_sampled,-1),covariance_matrix_wet_sampled)
    
# convert log-space flows to real-space flows
AnnualQ_s = np.exp(logAnnualQ_s)-1



######################Monthly and Spatial Disaggregation and Irrigation Files#########

    
name='Upper_Colorado'
abbrev='cm'
nSites=208
nIWRSites=379
startXBM=16
startIWR=463
xbm_file='./Data/'+abbrev+'2015_Statemod/StateMod/cm2015x.xbm'
iwr_file='./Data/'+abbrev+'2015_Statemod/StateMod/cm2015B.iwr'
xbm_out='./Data/cm2015_StateMod/NewXBM/'
iwr_out='./Data/cm2015_StateMod/NewIWR/'
abbrev_file_xbm='cm2015x.xbm'
abbrev_file_iwr='cm2015B.iwr'
historical_column=0

#
#
#name='Gunnison'
#abbrev='gm'
#nSites=139
#nIWRSites=541
#startXBM=16
#startIWR=620
#xbm_file='./Data/'+abbrev+'2015_Statemod/StateMod/gm2015x.xbm'
#iwr_file='./Data/'+abbrev+'2015_Statemod/StateMod/gm2015B.iwr'
#xbm_out='./Data/gm2015_StateMod/NewXBM/'
#iwr_out='./Data/gm2015_StateMod/NewIWR/'
#abbrev_file_xbm='gm2015x.xbm'
#abbrev_file_iwr='gm2015B.iwr'
#historical_column=1
#
#
#name='yampa'
#abbrev='ym'
#nSites=94
#nIWRSites=298
#startXBM=16
#startIWR=370
#xbm_file='./Data/'+abbrev+'2015_Statemod/StateMod/ym2015x.xbm'
#iwr_file='./Data/'+abbrev+'2015_Statemod/StateMod/ym2015B.iwr'
#xbm_out='./Data/ym2015_StateMod/NewXBM/'
#iwr_out='./Data/ym2015_StateMod/NewIWR/'
#abbrev_file_xbm='ym2015x.xbm'
#abbrev_file_iwr='ym2015B.iwr'
#historical_column=2
#
#name='White'
#abbrev='wm'
#nSites=43
#nIWRSites=137
#startXBM=16
#startIWR=202
#xbm_file='./Data/'+abbrev+'2015_Statemod/StateMod/wm2015x.xbm'
#iwr_file='./Data/'+abbrev+'2015_Statemod/StateMod/wm2015B.iwr'
#xbm_out='./Data/wm2015_StateMod/NewXBM/'
#iwr_out='./Data/wm2015_StateMod/NewIWR/'
#abbrev_file_xbm='wm2015x.xbm'
#abbrev_file_iwr='wm2015B.iwr'
#historical_column=3
#
#
#name='San_Juan'
#abbrev='sj'
#nSites=165
#nIWRSites=296
#startXBM=16
#startIWR=377
#xbm_file='./Data/'+abbrev+'2015_Statemod/StateMod/sj2015x.xbm'
#iwr_file='./Data/'+abbrev+'2015_Statemod/StateMod/sj2015B.iwr'
#xbm_out='./Data/sj2015_StateMod/NewXBM/'
#iwr_out='./Data/sj2015_StateMod/NewIWR/'
#abbrev_file_xbm='sj2015x.xbm'
#abbrev_file_iwr='sj2015B.iwr'
#historical_column=4


#Make output directories
os.mkdir('./Data/'+abbrev+'2015_StateMod/NewXBM')   
os.mkdir('./Data/'+abbrev+'2015_StateMod/NewIWR')

#Create annual and monthly streamflow dataframes from xbm file 
utils.createXBMDataFrames(xbm_file,startXBM,nSites,abbrev) 

#load annual and monthly flow files
MonthlyQ_h = np.array(pd.read_csv('./Data/'+abbrev+'2015_StateMod/MonthlyQ.csv',header=None))
AnnualQ_h = np.array(pd.read_csv('./Data/'+abbrev+'2015_Statemod/AnnualQ.csv',header=None))

#Create annual and monthly irrigation dataframes from IWR files
utils.createIWRDataFrames(iwr_file,startIWR,nIWRSites,abbrev) 

# load historical (_h) irrigation demand data
AnnualIWR_h = np.loadtxt('./Data/'+abbrev+'2015_StateMod/AnnualIWR.csv',delimiter=',')
MonthlyIWR_h = np.loadtxt('./Data/'+abbrev+'2015_StateMod/MonthlyIWR.csv',delimiter=',')
IWRsums_h = np.sum(AnnualIWR_h,1)
IWRfractions_h = np.zeros(np.shape(AnnualIWR_h))
for i in range(np.shape(AnnualIWR_h)[0]):
    IWRfractions_h[i,:] = AnnualIWR_h[i,:] / IWRsums_h[i]
    
IWRfractions_h = np.mean(IWRfractions_h,0)

# model annual irrigation demand anomaly as function of annual flow anomaly at last node
BetaIWR, muIWR, sigmaIWR = utils.fitIWRmodel(AnnualQ_h, AnnualIWR_h)


# calculate annual IWR anomalies based on annual flow anomalies at last node
TotalAnnualIWRanomalies_s = BetaIWR*(AnnualQ_s[:,historical_column]-np.mean(AnnualQ_s[:,historical_column])) + \
        ss.norm.rvs(muIWR, sigmaIWR,len(AnnualQ_s[:,historical_column]))
TotalAnnualIWR_s = np.mean(IWRsums_h)*LHS[p,6] + TotalAnnualIWRanomalies_s
AnnualIWR_s = np.dot(np.reshape(TotalAnnualIWR_s,[np.size(TotalAnnualIWR_s),1]), \
                                 np.reshape(IWRfractions_h,[1,np.size(IWRfractions_h)]))


#Read in monthly flows at all sites
MonthlyQ_all = utils.organizeMonthly(xbm_file, startXBM, nSites)
MonthlyQ_all_ratios = np.zeros(np.shape(MonthlyQ_all))


#Divide monthly flows at each site by the monthly flow at the last node
for i in range(np.shape(MonthlyQ_all_ratios)[2]):
    MonthlyQ_all_ratios[:,:,i] = MonthlyQ_all[:,:,i]/MonthlyQ_all[:,:,-1]
    
    
#Get historical flow ratios
AnnualQ_h_ratios = np.zeros(np.shape(AnnualQ_h))
for i in range(np.shape(AnnualQ_h_ratios)[0]):
    AnnualQ_h_ratios[i,:] = AnnualQ_h[i,:] / np.sum(AnnualQ_h[i,-1])
    
    
#Get historical flow ratios for last node monthly
last_node_breakdown = np.zeros([105,12])
for i in range(np.shape(last_node_breakdown)[0]):
   last_node_breakdown[i,:] =  MonthlyQ_all[i,:,-1]/ AnnualQ_h[i,-1]
 
    
    
MonthlyQ_s = np.zeros([nYears,nSites,12])
MonthlyIWR_s = np.zeros([nYears,np.shape(MonthlyIWR_h)[1],12])
#disaggregate annual flows and demands at all sites using randomly selected neighbor from k nearest based on flow
dists = np.zeros([nYears,np.shape(AnnualQ_h)[0]])
for j in range(nYears):
   for m in range(np.shape(AnnualQ_h)[0]):
      dists[j,m] = dists[j,m] + (AnnualQ_s[j,0] - AnnualQ_h[m,-1])**2


#Create probabilities for assigning a nearest neighbor for the simulated years
probs = np.zeros([int(np.sqrt(np.shape(AnnualQ_h)[0]))])
for j in range(len(probs)):
    probs[j] = 1/(j+1)       
    probs = probs / np.sum(probs)
    for j in range(len(probs)-1):
        probs[j+1] = probs[j] + probs[j+1]        
probs = np.insert(probs, 0, 0) 


  
for j in range(nYears):
    # select one of k nearest neighbors for each simulated year
    neighbors = np.sort(dists[j,:])[0:int(np.sqrt(np.shape(AnnualQ_h)[0]))]
    indices = np.argsort(dists[j,:])[0:int(np.sqrt(np.shape(AnnualQ_h)[0]))]
    randnum = random()
    for k in range(1,len(probs)):
        if randnum > probs[k-1] and randnum <= probs[k]:
            neighbor_index = indices[k-1]
    #Use selected neighbors to downscale flows and demands each year at last nodw
    MonthlyQ_s[j,-1,:] = last_node_breakdown[neighbor_index,:]*AnnualQ_s[j,0]
        
    #Find monthly flows at all other sites each year
    for k in range(12):
        MonthlyQ_s[j,:,k] = MonthlyQ_all_ratios[neighbor_index,k,:]*MonthlyQ_s[j,-1,k]
        
    for k in range(np.shape(MonthlyIWR_h)[1]):
        if np.sum(MonthlyIWR_h[neighbor_index*12:(neighbor_index+1)*12,k]) > 0:
                proportions = MonthlyIWR_h[neighbor_index*12:(neighbor_index+1)*12,k] / np.sum(MonthlyIWR_h[neighbor_index*12:(neighbor_index+1)*12,k])
        else:
            proportions = np.zeros([12])

        MonthlyIWR_s[j,k,:] = proportions*AnnualIWR_s[j,k]    

# write new flows to file for LHsample i (inputs: filename, firstLine, sampleNo,realization, allMonthlyFlows,output folder)
utils.writeNewStatemodFiles(abbrev_file_xbm, abbrev,startXBM, i, 1, MonthlyQ_s,xbm_out)
        
# write new irrigation demands to file for LHsample i
utils.writeNewStatemodFiles(abbrev_file_iwr, abbrev,startIWR, i, 1, MonthlyIWR_s,iwr_out) 
      


##############################Quick test for spatial correlation################
#test=np.reshape(np.swapaxes(MonthlyQ_s,1,2),[105*12,nSites])
#
#cmap = matplotlib.cm.get_cmap('viridis')
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.matshow(np.corrcoef(np.transpose(test)),cmap=cmap)
#sm = matplotlib.cm.ScalarMappable(cmap=cmap)
#sm.set_array([np.min(np.corrcoef(np.transpose(test))),np.max(np.corrcoef(np.transpose(test)))])
#ax.set_title('Synthetic Spatial Correlation',fontsize=16)
#ax.tick_params(axis='both',labelsize=14)
#ax.set_ylabel('Basin Node',fontsize=16)

 