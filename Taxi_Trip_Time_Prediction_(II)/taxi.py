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
dataset = read_csv('reordered.csv')
values  = dataset.values

# data concatenation
