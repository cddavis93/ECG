import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pywt

apData = np.loadtxt(fname = "MIT-BIH_bw_rs.csv",delimiter = ",",dtype='float')

Ytargets = []
Xdata = []

for i in range(0,len(apData)):
    Xdata.append(apData[i][:-1])
    Ytargets.append(apData[i][-1])
trainsize = 80000

#FFT^2
Xtrain = Xdata[0:trainsize]
Ytarg = Ytargets[0:trainsize]

Xfft = np.fft.fft(Xtrain)

XfftTemp = Xfft
for i in range(0,rg):
    for a in range(0,200):
        XfftTemp[i][a] = Xfft[i][a].real*Xfft[i][a].real + Xfft[i][a].imag*Xfft[i][a].imag + 2*Xfft[i][a].real*Xfft[i][a].imag

Xfft2 = XfftTemp.real

clf = svm.SVC(C=1.0, gamma='auto', kernel='linear')
clf.fit(Xfft2, Ytargets)

Xtest = Xdata[trainsize:]

tXfft = np.fft.fft(Xtest)
tXfftTemp = tXfft
for i in range(0,len(tXfft)):
    for a in range(0,200):
        tXfftTemp[i][a] = tXfft[i][a].real*tXfft[i][a].real + tXfft[i][a].imag*tXfft[i][a].imag + 2*tXfft[i][a].real*tXfft[i][a].imag
tXfft2 = tXfftTemp.real
predYdata = clf.predict(tXfft2)

total = 0
cPred = 0
for i in range(0,len(Ytest)):
    if (predYdata[i] == Ytest[i]):
        cPred += 1
    total += 1
accuracy = float(cPred)/float(total)
print (accuracy)