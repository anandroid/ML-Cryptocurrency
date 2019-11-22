import MySQLdb
import pandas as pd
import pandas
import numpy as np
import pandas
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


new = df.filter(['buy_for','sell_for', 'volume', 'time'], axis=1)
#print(new)
msk = np.random.rand(len(new)) < 0.8
m = len(new)
x =0.8*m
y=math.ceil(x)
train = new[:y]
test = new[y: m]
#print(train)

y_train = train.sell_for.values.reshape((len(train),1))
dates_train = train.time.values.reshape((len(train)))
x_train = train.drop(['sell_for', 'time'], axis=1).values.reshape((len(train), 2))
y_test = test.sell_for.values.reshape((len(test),1))
dates_test = test.time.values.reshape((len(test)))
x_test = test.drop(['sell_for', 'time'], axis=1).values.reshape((len(test),2))

#print(y_train)
#print(x_train)
#print(x_test)

xgb = xgb.XGBRegressor(colsample_bytree=0.8, subsample=0.5,
                               learning_rate=0.001, max_depth=3, min_child_weight=1.8, n_estimators=2000,
                               reg_alpha=0.1, reg_lambda=0.3, gamma=0.01,
                               silent=1, random_state =7, nthread = -1)


xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_train = xgb.predict(x_train)

#y_test = xgb.predict(x_train)
#print(xgb_pred)
#RMSE = np.sqrt(mean_squared_error(xgb_pred, y_test))
#print(RMSE.round(4))

#date_objects = [datetime.strptime(date, '%m/%d/%Y').date() for date in dates_test]
#print(dates_test)

plt.figure(0)
plt.plot(dates_test[-100:], xgb_pred[-100:])
plt.plot(dates_test[-100:], y_test[-100:])
#plt.figure(1)
# plt.plot(dates_train, xgb_train)
# plt.plot(dates_train, y_train)
plt.show()
