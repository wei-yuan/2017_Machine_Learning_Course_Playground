# --------------------------------------
# Library
# --------------------------------------
# Tools
from pandas import read_csv
import numpy
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# Model
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Validation tool
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

########################
#---- Load dataset ----#
########################
dataset = read_csv('reordered.csv')
print (dataset.head(5))
values  = dataset.values

########################
#------- dataset ------#
########################
n_data = len(dataset.index) # total number of data
train_fields = ['HOUR', 'CALL_TYPE', 'TAXI_ID']
train = read_csv('reordered.csv', usecols=train_fields)
label_fields = ['POLYLINE']
label = read_csv('reordered.csv', usecols=label_fields)
# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
									train, 
									label, 
									test_size=0.4, 
									random_state=0)

########################
#------- Model --------#
########################
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

########################
#------Evaluation------#
########################
# evaluate model
y_pred = regr.predict(x_test)
#diabetes_y_pred = regr.predict(diabetes_X_test)
print ("coefficient of determination R^2 of the prediction: %.2f" % regr.score(x_test, y_test))
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

'''
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
'''