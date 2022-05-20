# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:14:39 2022

@author: imran
"""

import os,datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

TRAIN_PATH = os.path.join(os.getcwd(), 'cases_malaysia_train.csv')
TEST_PATH = os.path.join(os.getcwd(), 'cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(), 'Log')

#%% EDA

# Step 1) Data Loading
X_train = pd.read_csv(TRAIN_PATH)
x_test = pd.read_csv(TEST_PATH)
x_test=x_test[1]

# Step 2) Data Intepretation
X_train.info()
X_train.describe()

# Step 3) Data Visualization
plt.figure()
plt.plot()
plt.show()

mms = MinMaxScaler()
x_train_scaled = mms.fit_transform(np.expand_dims(X_train,axis=-1))
x_test_scaled = mms.transform(np.expand_dims(x_test,axis=-1))

window_size = 30 # Window Size 30 Days as mentioned in question
X_train=[]
Y_train=[]
for i in range(window_size,len(X_train)):
    X_train.append(x_train_scaled[i-window_size:i,0])
    Y_train.append(x_train_scaled[i,0])
    
X_train=np.array(X_train)
Y_train=np.array(Y_train)

# Testing Dataset
temp = np.concatenate((x_train_scaled,x_test_scaled))
len_window = window_size+len(x_test_scaled)
temp = temp[-len_window:]

X_test=[]
Y_test=[]
for i in range(window_size,len(temp)):
    X_test.append(temp[i-window_size:i,0])
    Y_test.append(temp[i,0])
    
X_test=np.array(X_test)
Y_test=np.array(Y_test)

X_train=np.expand_dims(X_train,axis=-1)
Y_test=np.expand_dims(X_test,axis=-1)

#%% Model Creation

model = Sequential()
model.add(LSTM(64,
               return_sequences=(True), 
               input_shape=(X_train.shape[1:])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mse', metrics='mse')

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%Y'))
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

hist = model.fit(X_train,
                 Y_train,
                 epochs=5,
                 batch_size=128, 
                 callbacks=[tensorboard_callback,early_stopping_callback])
print(hist.history.keys())

#%% Visualizations

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['vol_loss'])
plt.show()

plt.figure()
plt.plot(hist.history['mse'])
plt.plot(hist.history['vol_mse'])
plt.show()

#%% 

pred = []
for i in X_test:
    pred.append(model.predict(np.expand_dims(i,axis=0)))
    
pred = np.array(pred)

#%% 

plt.figure()
plt.plot(pred.reshape(len(pred),1))
plt.plot(Y_test)
plt.legend(['Predicted','Actual'])
plt.show()

y_true = Y_test
y_pred = pred.reshape(len(pred),1)

print((mean_absolute_error(y_true,y_pred)/sum(abs(y_true)))*100)
