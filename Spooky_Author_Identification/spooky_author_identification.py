# Tools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import nltk

# Data preprocessor
from sklearn import preprocessing

# Model
from sklearn import svm

# Validation tool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Read data set
train = pd.read_csv("data_set/train.csv")

# Data cleaning without hand-crafted

