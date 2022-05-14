# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:54:32 2022

@author: rg727
"""

import numpy as np
import os 

   
name='Upper_Colorado'
abbrev='cm'
res_file='./Data/'+abbrev+'2015_Statemod/StateMod/cm2015B.res'
abbrev_file_res='cm2015B.res'
out_directory='./Data/'+abbrev+'2015_StateMod/NewRES/'   
#Make output directories
os.mkdir('./Data/'+abbrev+'2015_StateMod/NewRES')   



#Import LHS Sample
LHS=np.loadtxt('./LHsamples_CO.txt')

# For RES
# get unsplit data to rewrite everything that's unchanged
with open(res_file,'r') as f:
    all_data_RES = [x for x in f.readlines()]       
f.close() 

# Function for RES files
def writenewRES(lines, k,data,filename,abbrev, sampleNo, realizationNo,out_directory):
    copy_all_data_RES = np.copy(all_data_RES)       
    # Change only the specific lines
    for j in range(len(lines)):
        split_line = all_data_RES[lines[j]].split('.')
        split_line[1] = ' ' + str(int(float(split_line[1])*LHS[k,7]))
        copy_all_data_RES[lines[j]]=".".join(split_line)                
    # write new data to file
    f = open(out_directory+ filename[0:-4] + '_S' + str(sampleNo) + '_R' + str(realizationNo)+ filename[-4::],'w')
    for i in range(len(copy_all_data_RES)):
        f.write(copy_all_data_RES[i])
    f.close()
    
    return None




for k in range(0, len(LHS)):
    writenewRES([395,348,422,290,580,621], k,all_data_RES,abbrev_file_res,abbrev,k,1,out_directory)

