""" 
This is the rewritten code, improved by removing unnecessary ops 
and changing the data format to NHWC
This works when converting to tflite or protobuf
Code is refactored to Tensorflow 1.13, python 3.73, keras 2.24
Code is tested on a machine with i7 7700HQ and GTX 1060 6GB

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 # Importing opencv
import _pickle as pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Visualizing the graph using Tensorboard
from time import time
from keras.callbacks import TensorBoard

NAME = "CNNvAdamv2" 
# NAME can be changed according to the optimizer used 
# to be shown in TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

""" 
To see this visualization open CMD and type 
[tensorboard --logdir=logs/] 
Then copy the address provided into a browser and open it there

"""
# The model might try to hog VRAM, this is done to avoid that

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# The code block below deals with loading the pickled dataset into memory
pickle_files = ['open_eyes.pickle', 'closed_eyes.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            print("Numpy Concatenate")

            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
            
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save  # hint to help gc free up memory
    i += 1
X_train = train_dataset
Y_train = train_labels
X_test = test_dataset
Y_test = test_labels

print("Data format below, NHWC")
print(X_train.shape)

# The code block below is the model building using the Keras API for Tensorflow

# https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9

batch_size = 32 # internet consensus says this is best for gpu
nb_classes = 1 # nb_classes: number of output classes
epochs = 25 # must not be too large, risk overfitting otherwise

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(24,24,1), data_format='channels_last'))
model.add(Activation('relu'))

model.add(Conv2D(24, (3, 3)),)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
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

#model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test),callbacks=[tensorboard])

score = model.evaluate(X_test, Y_test,  verbose=2)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# The code block above now works fine alhamdulillah

# The code block below saves the model into a file 
# that can be converted into tflite

model.summary()
model.save('adamv2.h5') 
model_json = model.to_json()
with open("adamv2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("adamv2weights.h5")


