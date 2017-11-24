# Data container and plot tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
# Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# Validation tool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics as m
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# Read data set
data = pd.read_csv('bank/bank.csv', sep=';',header='infer')

def numericalType_(data):    
    # input feature
    data.job.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired',
                    'self-employed','services','student','technician','unemployed','unknown'),
                    (0,1,2,3,4,5,6,7,8,9,10,11),
                    inplace=True)
    data.default.replace(('yes','no','unknown'),(1,0,2),inplace=True)
    data.housing.replace(('yes','no'),(1,0),inplace=True)
    data.loan.replace(('yes','no'),(1,0),inplace=True)
    data.marital.replace(('married','single','divorced'),(1,2,3),inplace=True)
    data.contact.replace(('telephone','cellular','unknown'),(1,2,3),inplace=True)
    data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
    data.education.replace(('primary','secondary','tertiary','unknown'),(1,2,3,4),inplace=True)    
    # output label
    data.y.replace(('yes','no'),(1,0),inplace=True)
    return data

# apply numerical type to original data for non decision tree classifier
data = numericalType_(data)

# prepare X, Y axis term
data_X = data.drop(['day','poutcome','y'], axis=1)
data_Y = data['y']
print ("Original Data Column: %s" % data.columns.tolist(), "After Preprocessed Data Column: %s" % data_X.columns.tolist())

# split data set into train and test
# If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size = 0.2)

# Build classifier, Decision Tree Classifier
clf = DecisionTreeClassifier()
bank_clf = clf.fit(train_X, train_y)

# Predict 
test_y_predicted = bank_clf.predict(test_X)
# print(test_y_predicted)

# Validation Metric
accuracy = m.accuracy_score(test_y, test_y_predicted) 
precision = m.precision_score(test_y,test_y_predicted,average='macro')
recall = m.recall_score(test_y,test_y_predicted,average='macro')
f1_score = m.f1_score(test_y,test_y_predicted,average='macro')
roc_auc = roc_auc_score(test_y, test_y_predicted)

# Print using string formatting
classifiers = {'Decision Tree':DecisionTreeClassifier()}
log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)
log_entry = pd.DataFrame([["Decision Tree",accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)
log = log.append(log_entry)

print(log)

'''
# show data distribution
plt.figure(1)

ax = plt.subplot(4,2,1)
ax.set_title("Age")
plt.hist((data.age), bins=100)

ax = plt.subplot(4,2,2)
ax.set_title("job")
plt.hist((data.job), bins=12)

ax = plt.subplot(4,2,3)
ax.set_title("marital")
plt.hist((data.marital), bins=3)

ax = plt.subplot(4,2,4)
ax.set_title("education")
plt.hist((data.education), bins=10)

ax = plt.subplot(4,2,5)
ax.set_title("default")
plt.hist((data.default), bins=3)

ax = plt.subplot(4,2,6)
ax.set_title("housing")
plt.hist((data.housing), bins=2)

ax = plt.subplot(4,2,7)
ax.set_title("loan")
plt.hist((data.loan), bins=2)

plt.figure(2)
ax.set_title("balance")
plt.hist((data.balance), bins=1000)

plt.figure(3)

ax = plt.subplot(3,2,1)
ax.set_title("contact")
plt.hist((data.contact), bins=3)

ax = plt.subplot(3,2,2)
ax.set_title("month")
plt.hist((data.month), bins=12)

ax = plt.subplot(3,2,3)
ax.set_title("duration")
plt.hist((data.duration), bins=100)

ax = plt.subplot(3,2,4)
ax.set_title("campaign")
plt.hist((data.campaign), bins=10)

ax = plt.subplot(3,2,5)
ax.set_title("pdays")
plt.hist((data.pdays), bins=1000)

ax = plt.subplot(3,2,6)
ax.set_title("previous")
plt.hist((data.previous), bins=100)

plt.show()
'''

