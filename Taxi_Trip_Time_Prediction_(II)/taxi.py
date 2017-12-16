# --------------------------------------
# Library
# --------------------------------------
# Tools
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy
import math
from datetime import datetime
import tzlocal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Model
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Validation tool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

########################
#---- Load dataset ----#
########################
from pandas import read_csv
dataset = read_csv('reordered.csv')
values  = dataset.values

########################
#-Train / Test dataset-#
########################
# split into train and test sets
n_features = 3
n_data = 8
train = values[:n_data, :n_features] # :n_features = 0 : n_train_hours
test = values[n_data:, :]
# split into input and outputs
#n_obs = n_data * n_features

train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

########################
#------- Model --------#
########################
# design network
def linear_regression_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

########################
#------Evaluation------#
########################

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=linear_regression_model, nb_epoch=100, batch_size=200, verbose=0)


kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

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