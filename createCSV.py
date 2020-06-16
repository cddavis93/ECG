from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import csv
import wfdb
import scipy as sp
from wfdb import processing
from scipy import signal
import heartpy as hp

name = (100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,
       121,122,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,
       220,221,222,223,228,230,231,232,233,234)

with open('MIT-BIH.csv','w') as t1:
    writer = csv.writer(t1, delimiter=',',lineterminator='\n')
    for file in name:
        sig, fields = wfdb.rdsamp(str(file))
        ann = wfdb.rdann(str(file), 'atr')
        ind_last = 0
        Ch1 = []
        Ch2 = []
        for i in range(0,len(sig)):
            Ch1.append(sig[i][0])
            #Ch2.append(sig[i][1])
        ind_prev = 0
        ind_beg = 0
        ind_end = 0
        bw = hp.filtering.remove_baseline_wander(Ch1,360)

        for i in range(1,len(ann.sample)-1):
            ind_next = ann.sample[i+1]
            ind_curr = ann.sample[i]
            array = []
            
            ind_beg = int((ind_prev+ind_curr)/2)
            ind_end = int((ind_next+ind_curr)/2)
      
            front = signal.resample(bw[ind_beg:ind_curr],100)
            back = signal.resample(bw[ind_curr:ind_next],100)
            write_array = np.concatenate((front,back),axis=None)

            write_array = write_array.tolist()

            if(ann.symbol[i] == 'N'):
                write_array.append(1.0)
            else:
                write_array.append(2.0)
            
            writer.writerow(write_array)
            ind_prev = ind_curr
