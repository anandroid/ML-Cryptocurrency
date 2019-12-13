#!/usr/bin/env python
# coding: utf-8

# In[7]:


import MySQLdb
import pandas as pd
import numpy as np
import xgboost as xgb

import datetime
import math
import matplotlib.pyplot as plt

conn = MySQLdb.connect(host="remotemysql.com", user="6txKRsiwk3", passwd="nPoqT54q3m", db="6txKRsiwk3")
cursor = conn.cursor()

sql = "select * from fetcherhistory where coin='BNB' and market='BINANCE'"

df = pd.read_sql_query(sql, conn)
# disconnect from server
conn.close()


group = df.groupby('time')
Real_Price = group['sell_for'].mean()
prediction_count = 0
df_train= Real_Price[:len(Real_Price) - prediction_count]
df_test= Real_Price[len(Real_Price) - prediction_count:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1))
test_set = df_test.values
test_set = np.reshape(test_set, (len(test_set), 1))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_data = training_set.reshape(-1,1)
test_data = test_set.reshape(-1,1)
EMA = 0.0
gamma = 0.1
for ti in range(3000):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'time']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

window_size = 100
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(2797,3297),all_mid_data[2797:3297],color='b',label='Predicted')
plt.plot(range(30,N),run_avg_predictions[30:],color='orange', label='True')
plt.xlabel('Date')
plt.ylabel('Sell Price')
plt.legend(fontsize=18)
plt.show()

X11=np.array(all_mid_data[2797:3297])
X22=np.array(run_avg_predictions[2797:3297])
i=0
difference=[]
for i in range(X11.shape[0]):
    difference.append(abs((X11[i]-X22[i])/X22[i]))
sum=0
i=0
for i in range(X11.shape[0]):
    sum=sum+difference[i]
print('accuracy for Exponential Moving Average:')
print(100-(sum/X11.shape[0])*100)

