import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

print("[INFO] Script Started")
print("[INFO] Loading Data")
apData = np.loadtxt(fname = "MIT-BIH.csv",delimiter = ",",dtype='float')
print(apData.shape)
print("[INFO] Separating Data")
ct1 = 0
ct2 = 0
tct1 = 0
tct2 = 0
Xdata = []
Ytargets = []
Xtest = []
Ytest = []

for i in range(0,len(apData)):
    if(ct1<35000): #of normal beats for training
        if(apData[i][-1] == 1.0):
            Xdata.append(apData[i][:-1])
            Ytargets.append(apData[i][-1])
            ct1 += 1
    elif(tct1<5000):#of normal beats for testing
        if(apData[i][-1] == 1.0):
            Xtest.append(apData[i][:-1])
            Ytest.append(apData[i][-1])
            tct1 += 1
            
    if(ct2<30000):#of abnormal beats for training
        if(apData[i][-1] == 2.0):
            Xdata.append(apData[i][:-1])
            Ytargets.append(apData[i][-1])
            ct2 += 1
    elif(tct2<5000):#of abnormal beats for testing
        if(apData[i][-1] == 2.0):
            Xtest.append(apData[i][:-1])
            Ytest.append(apData[i][-1])
            tct2 += 1
print(len(Xdata))
print(len(Xtest))

print("[INFO] FFT")

#Training
Xfft = np.fft.fft(Xdata)
XfftTemp = Xfft
for i in range(0,len(Xfft)):
    for a in range(0,200):
        XfftTemp[i][a] = Xfft[i][a]**2

Xfft2 = XfftTemp.real

#Testing
tXfft = np.fft.fft(Xtest)
tXfftTemp = tXfft
for i in range(0,len(tXfft)):
    for a in range(0,200):
        tXfftTemp[i][a] = tXfft[i][a]**2
tXfft2 = tXfftTemp.real

C_opt = [0.0001,0.001,0.01,0.1,1.0,2.0,5.0]

print("[INFO] Training Started")

with open('SVM_data.txt','w') as file:
	file.write("Timing and Accuracy Report \n")
	
	for a in C_opt:
		print("[INFO] SVM C-value: " str(a) + "\n")
		file.write("C-value: " + str(a) + "\n")
		clf = svm.SVC(C=a, gamma='auto', kernel='linear')
		train_time_start = time.time()
		clf.fit(Xfft2, Ytargets)
		train_time_end = time.time()
		train_time = train_time_end - train_time_start
		file.write("Train Time: " + str(train_time) + "\n")
		predYdata = clf.predict(tXfft2)

		total = 0
		cPred = 0
		for i in range(0,len(Ytest)):
   	 		if (predYdata[i] == Ytest[i]):
        			cPred += 1
    			total += 1
		accuracy = float(cPred)/float(total)
		file.write("Accuracy: " + str(accuracy) + "\n"
		print("Time: " + str(train_time))
		print("Accuracy: " + str(accuracy))
		