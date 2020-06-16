# Importing libraries
import tensorflow as tf	
from tensorflow import keras	
import matplotlib.pyplot as plt	
import time		
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

print ("[INFO] Script Started")


data = np.loadtxt(fname = "MIT-BIH_bw_rs.csv",delimiter = ",",dtype='float') #load the data from the csv
train_size = 85000 #chosen training size
train_data = []
train_labels = []
test_data = []
test_labels = []

for i in range(0,len(data)): ##loop to separate data into train/test_ data/labels
	if(i<train_size):
		train_data.append(data[i][:-1])
		train_labels.append(data[i][-1])
	else:
		test_data.append(data[i][:-1])
		test_labels.append(data[i][-1])

class_names = ['normal', 'abnormal']
train_data = np.fft.fft(train_data)

temp = train_data
for i in range(0,len(train_data)):
    for a in range(0,200):
        temp[i][a] = train_data[i][a]**2

train_data = temp.real

train_data = keras.utils.normalize(train_data,axis=1)


###double check model, these values may not be correct
model = tf.keras.applications.resnet.ResNet50(
	include_top=True,
	weights=None,
	input_shape=(200,1,1),
	input_tensor=None,
	pooling=None,
	classes=2
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


########## Training our model ##########
print ("[INFO] Training Started")
print ("[INFO] Time Tracking STARTED")
train_time_start = time.time()
model.fit(train_data, train_labels, batch_size=64, epochs=5)
train_time_end = time.time()
print ("[INFO] Time Tracking ENDED")

model.save("MIT_DNN.h5")

train_time_elapsed = train_time_end - train_time_start
print ("Elapsed Training Time: ", round(train_time_elapsed,5), " Seconds")

print ("[INFO] Prediction Starting")

temp = test_data
for i in range(0,len(test_data)):
    for a in range(0,200):
        temp[i][a] = test_data[i][a]**2

test_data = temp.real

test_data = keras.utils.normalize(test_data,axis=1)
prediction = model.predict(test_data)

correct = 0
incorrect = 0

for i in range (test_size):
	actual_label = class_names[int(test_labels[i])]
	prediction_label = class_names[np.argmax(prediction[i])]
	if actual_label == prediction_label:
		correct += 1
	else:
		incorrect += 1

print("Percentage correct: ")
accuracy =  float(correct)/float(rounds) * 100.0
print (accuracy)

print ("[INFO] Script Exiting")









