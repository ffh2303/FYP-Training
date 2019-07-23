import tensorflow as tf
import numpy as np
import cv2

np.random.seed(1337)
from tensorflow.contrib import lite
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.callbacks import Tensorboard
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from six.moves import cPickle as pickle #deprecated shit, for python 3.x it's now called _pickle

#NAME = "eyes-{}".format(int(time.time()))

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#sess = tf.Session()

pickle_files = ['open_eyes.pickle', 'closed_eyes.pickle']
i = 0
#this block of code below is to read the pickled files and then save them in memory as dataset
#if the pickle files are deleted, the dataset cannot be loaded
#the pickle files are created by the preprocess script

for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            print("here")

            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
            
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save  # hint to help gc free up memory
    i += 1

#print(test_dataset)
# dataset is 24x24, 1 channel since it's greyscale
""" 
Data formats
Data formats refers to the structure of the Tensor passed to a given op. The discussion below is specifically about 4D Tensors representing images. In TensorFlow the parts of the 4D tensor are often referred to by the following letters:

N refers to the number of images in a batch.
H refers to the number of pixels in the vertical (height) dimension.
W refers to the number of pixels in the horizontal (width) dimension.
C refers to the channels. For example, 1 for black and white or grayscale and 3 for RGB. 
NCHW or channels_first
NHWC or channels_last
SOURCE
https://www.tensorflow.org/guide/performance/overview
"""

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 30
nb_classes = 1
epochs = 12

X_train = train_dataset
print(X_train.shape)
# By default the operation before converts the pickle into NHWC format, but the nincompoop who wrote this code converted to NCHW witht the code below
X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
print(X_train.shape)
Y_train = train_labels

X_test = test_dataset
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
Y_test = test_labels

# print shape of data while model is building
print("{1} train samples, {4} channel{0}, {2}x{3}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {4} channel{0}, {2}x{3}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))
print(X_train.shape) # this shows that the data shape is actually NCHW, but we are expecting NHWC for tflite and shit
# input image dimensions
#_,  img_rows, img_cols, img_channels = X_train.shape 
print(X_train.shape)


# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# code block below commented out using ctrl + shift + a

""" model = Sequential()

model.add(Conv2D(32, (3, 3), 
                        input_shape=(img_channels, img_rows, img_cols),data_format='channels_first'))
model.add(Activation('relu'))

model.add(Conv2D(24, (3, 3), data_format='channels_first'),)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', data_format='channels_first'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))


# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

#model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test,  verbose=2)

print('Test score:', score[0])
print('Test accuracy:', score[1])

#import seaborn as sns
#plt.figure(figsize=(8,4))
#sns.countplot(x='label', data=train);

#loss = history.history['loss']
#epochs = range(1, len(loss)+1)
#plt.plot(epochs, loss, color = 'green', label ='validation loss')
#plt.title('Training and Validation Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

#acc = history.history['acc']
#val_acc = history.history['val_acc']
#plt.plot(epochs, acc, color = 'red', label = 'Training Accuracy')
#plt.plot(epochs, val_acc, color='green', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

#from sklearn.metrics import confusion_matrix
#Y_prediction = model.predict(X_test)
# Convert predictions classes to one hot vectors
#Y_pred_classes = np.argmax(Y_prediction,axis = 1)
# Convert validation observations to one hot vectors
#Y_true = np.argmax(Y_test,axis = 1)
# compute the confusion matrix
#confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
#plt.figure(figsize=(10,8))
#sns.heatmap(confusion_mtx, annot=True, fmt="d")
#plt.show()
model.summary()
# Creates a HDF5 file 'my_model.h5'
keras_file = "savedmodel.h5"
keras.models.save_model (model, keras_file)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:             
     json_file.write(model_json) 

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
#converter = lite.TFLiteConverter.from_keras_model_file(keras_file)
#tflite_model = converter.convert()
#open("savedmodel.tflite","wb").write(tflite_model) """
