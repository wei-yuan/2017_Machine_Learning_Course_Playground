# Data container and plot tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# Validation tool
from sklearn.cross_validation import train_test_split
from sklearn import metrics as m
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

data = pd.read_csv('bank/bank.csv', sep=';',header='infer')
data = data.drop(['day','poutcome'], axis=1)
print (data.columns.tolist())


def numericalType_(data):    
    #data.deposit.replace(('yes', 'no'), (1, 0), inplace=True)
    data.default.replace(('yes','no','unknown'),(1,0,2),inplace=True)
    data.housing.replace(('yes','no'),(1,0),inplace=True)
    data.loan.replace(('yes','no'),(1,0),inplace=True)
    data.marital.replace(('married','single','divorced'),(1,2,3),inplace=True)
    data.contact.replace(('telephone','cellular','unknown'),(1,2,3),inplace=True)
    data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
    data.education.replace(('primary','secondary','tertiary','unknown'),(1,2,3,4),inplace=True)
    
    return data

data = numericalType_(data)

plt.subplot(5,3,1)
plt.hist((data.age), bins=100)
plt.subplot(5,3,2)
plt.hist((data.job), bins=20)
plt.subplot(5,3,3)
plt.hist((data.marital), bins=10)
plt.subplot(5,3,4)
plt.hist((data.education), bins=10)
plt.subplot(5,3,5)
plt.hist((data.default), bins=3)
plt.subplot(5,3,6)
plt.hist((data.balance), bins=10000)
plt.subplot(5,3,7)
plt.hist((data.housing), bins=2)
plt.subplot(5,3,8)
plt.hist((data.loan), bins=2)
plt.subplot(5,3,9)
plt.hist((data.contact), bins=3)
plt.subplot(5,3,10)
plt.hist((data.month), bins=12)
plt.subplot(5,3,11)
plt.hist((data.duration), bins=100)
plt.subplot(5,3,12)
plt.hist((data.campaign), bins=10)
plt.subplot(5,3,13)
plt.hist((data.pdays), bins=1000)
plt.subplot(5,3,14)
plt.hist((data.previous), bins=100)
plt.subplot(5,3,15)
plt.hist((data.y), bins=2)
plt.show()

