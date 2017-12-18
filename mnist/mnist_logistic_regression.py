######################
###### Library #######
######################
# dataset
from keras.datasets import mnist
# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# lib
from keras.utils import np_utils

# tool
# %matplotlib inline
import matplotlib.pyplot as plt

# parametter
nb_classes = 10
nb_epoch = 10
batch_size = 256

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
######################
# data preprocessing #
######################
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# convert to float and normalization
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
# change to category
Y_Train = np_utils.to_categorical(y_train, nb_classes)
Y_Test = np_utils.to_categorical(y_test, nb_classes)

# plot
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

######################
####### Model ########
######################
# Logistic regression model
model = Sequential()
model.add(Dense(output_dim=10, input_shape=(784,), init='normal', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# check model
model.summary()

######################
####### Train  #######
######################
train_history = model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2, verbose=1)

######################
##### Evaluation #####
######################
evaluation = model.evaluate(X_test, Y_Test, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')