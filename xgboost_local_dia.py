import pandas
import numpy as np
import xgboost as xgb

import datetime
import math
import matplotlib.pyplot as plt

#from sklearn.xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
old = pandas.read_csv('eth.csv')
old.fillna(0,inplace=True)
new = old.filter(['date', 'CapMrktCurUSD','PriceUSD','ROI30d', 'TxCnt', 'TxTfrValAdjNtv'], axis=1)
#print(new)
msk = np.random.rand(len(new)) < 0.8
m = len(new)
x =0.8*m
y=math.ceil(x)
train = new[:y]
test = new[y: m]
#print(train)

y_train = train.PriceUSD.values.reshape((len(train),1))
dates_train = train.date.values.reshape((len(train)))
x_train = train.drop(['PriceUSD', 'date'], axis=1).values.reshape((len(train), 4))
y_test = test.PriceUSD.values.reshape((len(test),1))
dates_test = test.date.values.reshape((len(test)))
x_test = test.drop(['PriceUSD', 'date'], axis=1).values.reshape((len(test),4))

#print(y_train)
#print(x_train)
#print(x_test)

xgb = xgb.XGBRegressor(colsample_bytree=1, subsample=0.5,
                               learning_rate=0.01, max_depth=3, min_child_weight=1.8, n_estimators=10000,
                               reg_alpha=0.1, reg_lambda=0.01, gamma=0.01,
                               silent=1, random_state =0, nthread = -1)


xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_train = xgb.predict(x_train)

#y_test = xgb.predict(x_train)
#print(xgb_pred)
RMSE = np.sqrt(mean_squared_error(xgb_pred, y_test))
print(RMSE.round(4))
#print(xgb_pred)
y_test1 = y_test.flatten();
#print(y_test)
print(xgb.score(x_test, y_test1))


#date_objects = [datetime.strptime(date, '%m/%d/%Y').date() for date in dates_test]
#print(dates_test)

plt.figure(0)
plt.plot(range(0, 200), xgb_pred[:200], color="b", label="predicted")
plt.plot(range(0, 200), y_test1[:200], color="orange", label="actual")
plt.legend(fontsize=18)
#plt.figure(1)
# plt.plot(dates_train, xgb_train)
# plt.plot(dates_train, y_train)
plt.show()






