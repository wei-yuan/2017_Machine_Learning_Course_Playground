
from keras.utils import np_utils
# mnist dataset
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Activation 

# import data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
######################
# data preprocessing #
######################
input_dim = 784 # 28 * 28 to  784 * 1 
X_train = X_train.reshape(60000, input_dim) 
X_test = X_test.reshape(10000, input_dim) 
# convert to float
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 

X_train /= 255 
X_test /= 255

# change original data to
nb_classes = 10  # 10 digit class
Y_train = np_utils.to_categorical(y_train, nb_classes) 
Y_test = np_utils.to_categorical(y_test, nb_classes)

######################
####### Model ########
######################
output_dim = nb_classes
model = Sequential() 
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
batch_size = 128 
nb_epoch = 20

######################
####### Train  #######
######################
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test)) 

######################
##### Evaluation #####
######################

score = model.evaluate(X_test, Y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])
