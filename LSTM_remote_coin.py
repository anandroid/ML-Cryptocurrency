import MySQLdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

conn = MySQLdb.connect(host="remotemysql.com", user="6txKRsiwk3", passwd="nPoqT54q3m", db="6txKRsiwk3")
cursor = conn.cursor()

sql = "select * from fetcherhistory where coin='BNB' and market='BINANCE'"

df = pd.read_sql_query(sql, conn)
# disconnect from server
conn.close()

group = df.groupby('time')
Real_Price = group['sell_for'].mean()
prediction_count = 10
df_train = Real_Price[:len(Real_Price) - prediction_count-1000]
df_test = Real_Price[len(Real_Price) - prediction_count:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set) - 1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size=5, epochs=100)
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()
plt.plot(test_set, color='red', label='Real BTC Price')
plt.plot(predicted_BTC_price, color='blue', label='Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=40)
df_test = df_test.reset_index()
x = df_test.index
labels = df_test['time']
plt.xticks(x, labels, rotation='vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()

regressor.save("lstm_remote.h5")

