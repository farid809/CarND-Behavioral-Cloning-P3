import csv
import cv2
import numpy as np


import platform
import math
import random
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib import image as image

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, SpatialDropout2D, Flatten
from keras.layers import Conv2D, Cropping2D, Input, Conv2D
from keras.optimizers import Adam
from keras.models import load_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.client import device_lib


from keras.models import load_model
import h5py
from keras import __version__ as keras_version


def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load_dataset(path):
    lines = []
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    
    #skip csv header
    iter_lines = iter(lines)
    next(iter_lines)
    
    for line in iter_lines:
    
 
        
        center_image_path = line[0].split('/')[-1]
        left_image_path   = line[1].split('/')[-1]
        right_image_path  = line[2].split('/')[-1]
        
        
        #print(right_image)
        measurement = float(line[3])
        if not math.isclose(float(measurement),0.0):
            measurements.append(measurement)
            center_image = cv2.imread(center_image_path)
            images.append(np.array(cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)))

            
            
            measurements.append(float(measurement)+0.1)
            left_image = cv2.imread(left_image_path)
            images.append(np.array(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)))
            
            
            
            measurements.append(float(measurement)-0.1)
            right_image = cv2.imread(right_image_path)
            images.append(np.array(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)))
        
       
    
    return np.array(images),np.array(measurements)



def augment_data(features, labels):
    augmented_features = []
    augmented_labels = []
    
    close_to_zero_cnt=0
    for image, angle in zip(features, labels):
        if abs(angle) < 1000 :
            augmented_features.append(image)
            augmented_labels.append(angle)
        
            #Augment data set by appending fliped image
            #augmented_features.append(np.fliplr(image))
            augmented_features.append(cv2.flip(image, flipCode=1))
            augmented_labels.append(-1*angle)
            
        else:
            close_to_zero_cnt +=1
            
        
    print("close to zero : {}".format(close_to_zero_cnt))
    return np.array(augmented_features), np.array(augmented_labels)



def nVidia_model():
  



    model = Sequential()

    #Labmbda Layer: Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    
    # set up cropping2D layer
    
    
    model.add(Cropping2D(cropping=((70, 24), (60, 60)))) 
   
    model.add(Conv2D(24, (5, 5), padding="same", strides=(2, 2), activation="relu"))

    model.add(Conv2D(36, (5, 5), padding="same", strides=(2, 2), activation="relu"))
    
    model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu"))

    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))

    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))


    model.add(Flatten())

    model.add(Dense(100))

    #model.add(Dropout(0.1))

    model.add(Dense(50))

    #model.add(Dropout(0.1))

    model.add(Dense(10))

    model.add(Dense(1))

    return model
    

def nVidia_model_v2():
  



    model = Sequential()

    #Labmbda Layer: Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    
    # set up cropping2D layer
    
    #model.add(Cropping2D(cropping=((70,24), (0,0)))) 
    model.add(Cropping2D(cropping=((70, 24), (60, 60)))) 
   
    model.add(Conv2D(24, (5, 5), padding="same", strides=(2, 2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(36, (5, 5), padding="same", strides=(2, 2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())

    model.add(Dense(100))

    model.add(Dropout(0.5))

    model.add(Dense(50))

    model.add(Dropout(0.5))

    model.add(Dense(10))

    model.add(Dense(1))

    return model
    


#GPU Configuration
print("=========================================")
print(">>GPU Configuration")
print("====================")
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

get_available_gpus()



#Loading Training Data
print("=========================================")
print(">>Loading Training Data")
print("====================")
X_train = []
y_train = []

X_train_fwd = []
y_train_fwd = []
X_train_fwd,y_train_fwd = load_dataset('C:/simout/forward-3-laps/')
print("X_train_fwd ="+str(len(X_train_fwd)))
X_train = X_train_fwd
y_train = y_train_fwd


X_train_rev = []
y_train_rev = []
X_train_rev,y_train_rev = load_dataset('C:/simout/reverse-3-laps/')
print("X_train_rev ="+str(len(X_train_rev)))
X_train = np.concatenate( [X_train , X_train_rev])
y_train = np.concatenate ( [y_train , y_train_rev])


X_train_track2_fwd = []
y_train_track2_fwd = []
X_train_track2_fwd,y_train_track2_fwd = load_dataset('C:/simout/track2-fwd/')
print("X_train_track2_fwd ="+str(len(X_train_track2_fwd)))
X_train = np.concatenate( [X_train , X_train_track2_fwd])
y_train = np.concatenate ( [y_train , y_train_track2_fwd])

X_train_corrections = []
y_train_corrections = []
X_train_corrections,y_train_corrections = load_dataset('C:/simout/corrective-actions/')
print("X_train_corrections ="+str(len(X_train_corrections)))
X_train = np.concatenate( [X_train , X_train_corrections])
y_train = np.concatenate ( [y_train , y_train_corrections])


X_train_curves = []
y_train_curves= []
X_train_curves,y_train_curves = load_dataset('C:/simout/curves/')
print("y_train_curves ="+str(len(y_train_curves)))
X_train = np.concatenate( [X_train , X_train_curves])
y_train = np.concatenate ( [y_train , y_train_curves])




#Data Exploration
print("=========================================")
print(">>Data Exploration")
print("====================")

# TODO: Number of training examples
n_train = len(X_train)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print()
print("Number of training examples =", n_train)


print("Image data shape =", image_shape)
print("Number of unique classes =", n_classes)




#Data Augmentation
print("=========================================")
print(">>Data Augmentation")
print("====================")
X_train_aug1=[]
y_train_aug1=[]

#X_train_aug1,y_train_aug1=balance_dataset(X_train,y_train)
X_train_aug1=X_train
y_train_aug1=y_train


X_train_aug=[]
y_train_aug=[]
X_train_aug,y_train_aug= augment_data(X_train_aug1,y_train_aug1)




#Building Model
print("=========================================")
print(">>Building Model")
print("====================")
current_model='model.attempt-test.h5'


if os.path.isfile('./'+current_model):
    model = load_model(current_model)
    print("using saved model")
else:
    model = nVidia_model_v2()
    print("creating new model")
    
print(model.summary())




#Training Model
print("=========================================")
print(">>Training Model")
print("====================")

BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 0.0001


#model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
#let adams optimizer decide
model.compile(optimizer='adam', loss='mse')
history_object=model.fit(X_train_aug, y_train_aug, batch_size=128, epochs=EPOCHS, validation_split=0.20, shuffle=True, verbose=1)


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('./images/model_training_loss_trend.png')

model.save(current_model)


