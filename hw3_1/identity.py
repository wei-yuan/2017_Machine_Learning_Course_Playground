# lib
import numpy as np
import pandas as pd
from keras.utils import np_utils

# model
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense

# tool
# %matplotlib inline
import matplotlib.pyplot as plt

# load mnist dataset
from keras.datasets import mnist
(x_train_image, y_train_label),\
(x_test_image, y_test_label) = mnist.load_data()
print('train data =', len(x_train_image), ',train data dim=', x_train_image.shape)
print('train label=', len(y_train_label), ',train data dim=', y_train_label.shape)
print('test  data =', len(x_test_image), ',test data dim=', x_test_image.shape)

def plot_image(image):
    # set size of image
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()
# show image of mnist
plot_image(x_train_image[0])

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: 
        num=25
    for i in range(0, num):
        # plot sub plot with 5 rows and 5 cols
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label = " + str(labels[idx])
        if len(prediction) > 0:
            title += ",prediction =" + str(prediction[idx])
        ax.set_title(title, fontsize = 10)
        # set no ticks along x and y axis
        ax.set_xticks([])
        ax.set_yticks([])
        # read next data
        idx+=1
    plt.show()

# show what the data set look like
plot_images_labels_prediction(x_train_image, y_train_label, [], 0, 10)

# data reshape from 28 * 28 to 1 * 784 and normalize
x_Train_normalize = x_train_image.reshape(60000, 784).astype('float32') / 255
x_Test_normalize = x_test_image.reshape(10000, 784).astype('float32') / 255

# one-hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

# model
model = Sequential()
# dense: fully connected network
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal', # normal distribution init weight and bias
                activation='linear'))

model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))

print(model.summary())

# model configuration
model.compile(loss='categorical_crossentropy',
             optimizer='adam', metrics=['accuracy'])

# train the model
train_history = model.fit(x=x_Train_normalize, y=y_TrainOneHot,
                          validation_split=0.2,  # 80% for training, 20% for validation
                          epochs=10, 
                          batch_size=200,
                          verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# show training result
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print()
print('accuracy', scores[1])

# GPU memory release
del model