# Naive Bayes Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix




nvclassifier = GaussianNB()

Compound_X_train = []
Compound_y_train = []

for i in range(1):

    dataset = pd.read_csv('bnb_naive_label.csv')
    X = dataset.iloc[:,:5].values
    y = dataset['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 82)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    Compound_X_train.append(X_train)
    Compound_y_train.append(y_train)



New_X_train = []
New_y_train = []

for x_list in Compound_X_train:
    for x_row in x_list:
        New_X_train.append(x_row)

for y_list in Compound_y_train:
    for y_row in y_list:
        New_y_train.append(y_row)


nvclassifier.fit(New_X_train, New_y_train)

y_pred = nvclassifier.predict(X_test)
print(y_pred)

#lets see the actual and predicted value side by side
y_compare = np.vstack((y_test,y_pred)).T
#actual value on the left side and predicted value on the right hand side
#printing the top 5 values
y_compare[:5,:]


wrong_prediction = 0

for loop_index in range(len(y_test)):
   if y[len(y_train)-1+loop_index] != y_test[loop_index]:
       if y[len(y_train)-1+loop_index] == "SELL":
           wrong_prediction = wrong_prediction+1




print ('\n Loss with Naive Bayes Clasification is: ', (wrong_prediction/len(y_test))*100 )