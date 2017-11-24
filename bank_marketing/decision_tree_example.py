from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# Read iris data set
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# split data set into train and test
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

# Build classifier
clf = tree.DecisionTreeClassifier()
iris_clf = clf.fit(train_X, train_y)

# Predict
test_y_predicted = iris_clf.predict(test_X)
print(test_y_predicted)

# Print answer of prediction
print(test_y)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
s = "Accuracy = "
# Print using string formatting
print ('{}{}'.format(s, accuracy)) 
