# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:21:04 2020

@author: Radhika B
"""

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt(r"C:\Users\Vishal\Desktop\pima-indians-diabetes.csv", delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X[0]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #12 neurons or units
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model,
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=10)

# evaluate the keras model
_,accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
import numpy as np    
#MAke predictions
Xnew = np.array([[6,148,72.5,35,0,33.6,0.627,50]])
ynew = model.predict_classes(Xnew)
ynew

Xnew = np.array([[1,89,66,23,94,28.1,0.167,21]])
ynew = model.predict_classes(Xnew)
ynew