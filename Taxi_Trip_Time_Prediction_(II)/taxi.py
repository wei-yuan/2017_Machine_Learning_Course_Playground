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

# Model
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Validation tool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error

########################
#---- Load dataset ----#
########################
from pandas import read_csv
dataset = read_csv('sample.csv')
values  = dataset.values
dataset = dataset.astype('float32')

########################
#-- data preprocessing-#
########################
# TAXI_ID
def id_prep(dataset):
	taxi_id = dataset.TAXI_ID
	le = preprocessing.LabelEncoder()
	le.fit(taxi_id)
	label_taxi_id = le.transform(taxi_id)
	max_id = max(label_taxi_id)
	norm_taxi_id = label_taxi_id.astype('float32') / max_id

# TIMESTAMP
def time_prep(dataset):
	unix_timestamp = dataset.TIMESTAMP.astype('float32')

	for index, element in enumerate(unix_timestamp):
		utc_time = datetime.utcfromtimestamp(element)
		print("index, element, utc_time = %s, %s, %s" % (index, element, utc_time))
		#print ",unix_timestamp = %d" %  unix_timestamp[i]
		#print(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))

# POLYLINE
def poly_prep(dataset):
	poly = dataset.POLYLINE

	for index, element in enumerate(poly):



id_prep(dataset)
time_prep(dataset)