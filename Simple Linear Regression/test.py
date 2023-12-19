import numpy as np
import pandas as pd
from LinearRegression import LinearRegression

data = pd.read_csv('RealEstate.csv')
#in this data set we have a lot of independent variable, but we are trying to predict the price of the house using the age of the house

#seperate the requited columns
X_data = data['X2 house age '].values.tolist()
Y_data = data['Y house price of unit area'].values.tolist()

#calculate the split point to divide the training and testing data
totalDataSize = int(len(X_data))
trainDataSize = int((len(X_data) * 0.8) // 1)
testDataSize = totalDataSize - trainDataSize 

#get the training and testing data
X_train = X_data[:trainDataSize]
Y_train = Y_data[:trainDataSize]

X_test = X_data[trainDataSize:]
Y_test = Y_data[trainDataSize:]

model = LinearRegression(X_train, Y_train, X_test, Y_test)
model.train(0.001, 10000)
model.check()
