# lib
import numpy as np
import pandas as pd
from keras.utils import np_utils

# model
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense

# load mnist dataset
from keras.datasets import mnist
(x_train_image, y_train_label),\
(x_test_image, y_test_label) = mnist.load_data()
print('train data=', len(x_train_image), ', train data dim=', x_train_image.shape)
print('train label=', len(y_train_label), ', train data dim=', y_train_label.shape)
print('\ntest  data=', len(x_test_image), ',  test data dim=', x_test_image.shape)

# tool
import matplotlib.pyplot as plt

def plot_image(image):
    # set size of image
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

# main function
# input data
X = [[-1, -1], [0., 0.], [1., 1.]]
y = [0, 1] 