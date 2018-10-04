# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:41:50 2018

@author: santi
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.keras.api.keras as keras
from sklearn.model_selection import train_test_split

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalized the value to 0-1
X_train = X_train / 255
X_test = X_test / 255

"""
bw = plt.get_cmap('gray')
plt.imshow(X_train[99], cmap = bw)
print(y_train[99])
plt.show()
"""

# Flatten the input data to a 1-d array (vector)
def flatten_input(x_train, x_test):
    
    num_train_samples = x_train.shape[0]
    image_len = x_train.shape[1]
    image_wid = x_train.shape[2]
    
    x_train = x_train.reshape((num_train_samples, image_len*image_wid))
        
    num_test_samples = x_test.shape[0]
    x_test = x_test.reshape((num_test_samples, image_len*image_wid))
    
    return (x_train, x_test)



def baseline_model(input_shape, num_classes):
    # Begin of the model
    model = keras.models.Sequential()
    
    # input layer
    model.add(keras.layers.Dense(input_dim = input_shape,
                                 units = 256, 
                                 activation = 'relu'))
    
    # Output layer
    model.add(keras.layers.Dense(units = num_classes, 
                                 activation = 'softmax'))
    
    model.compile(loss = "categorical_crossentropy", 
                  optimizer = 'adam', 
                  metrics = ['accuracy'])
    
    return (model)

def test_baseline(X_train, X_test, y_train, y_test):
    
    # Flatten input images
    X_train, X_test = flatten_input(X_train, X_test)
    
    # One-hot encoding    
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    # Splitting the training sets into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train, 
                                                  stratify = y_train,
                                                  test_size = 0.2)
    
    input_shape = X_trn.shape[1]
    output_shape = y_trn.shape[1]
    # Get the NN modol
    model = baseline_model(input_shape, output_shape)
    
    # Fit model
    model.fit(x= X_trn, y= y_trn, 
              validation_data = (X_val, y_val), 
              epochs = 10, 
              batch_size = 128)
    
    score = model.evaluate(X_test, y_test)
    
    print("Accuracy: {}%".format(score[1]*100))
    

def cnn_model(input_shape, num_classes):
    
    # Begin of the model
    model = keras.models.Sequential()
    
    # Convolutional layer input
    model.add(keras.layers.Conv2D(input_shape = input_shape, 
                                  filters = 6, # num of filters applied (created)
                                  kernel_size = (5,5), 
                                  activation = 'relu'))
    
    # Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    
    # Dropout to avoid overfitting
    model.add(keras.layers.Dropout(rate = 0.1))
    
    # Falttening
    model.add(keras.layers.Flatten())    
    
    # input layer
    model.add(keras.layers.Dense(units = 256, 
                                 activation = 'relu'))
    
    # Output layer
    model.add(keras.layers.Dense(units = num_classes, 
                                 activation = 'softmax'))
    
    model.compile(loss = "categorical_crossentropy", 
                  optimizer = 'adam', 
                  metrics = ['accuracy'])
    
    return (model)





def test_cnn(X_train, X_test, y_train, y_test):
    
    # Add 1 channel depth
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    
    # One-hot encoding    
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    # Splitting the training sets into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train, 
                                                  stratify = y_train,
                                                  test_size = 0.2)
    
    im_len, im_wid = X_trn.shape[1], X_trn.shape[2]
    im_channels = X_trn.shape[3]
    
    # CNN expect 3D input!!! thus im_channel
    input_shape = (im_len, im_wid, im_channels)
    output_shape = y_trn.shape[1]
    
    # Get the NN modol
    model = cnn_model(input_shape, output_shape)
    
    # Fit model
    model.fit(x= X_trn, y= y_trn, 
              validation_data = (X_val, y_val), 
              epochs = 10, 
              batch_size = 128)
    
    score = model.evaluate(X_test, y_test)
    
    print("Accuracy: {}%".format(score[1]*100))
    
    return model
    

 

def find_errors(model, X_test, y_test):
    idxs = []
    vals = model.predict(X_test)
    
    for i in range(len(vals)): 
        preds = vals[i]
        
        class_pred = np.argmax(preds)
        class_truth = np.argmax(y_test[i])
        
        if class_pred != class_truth:
            idxs.append(i)
            
    errors = X_test[idxs]
    errors = errors.reshape(errors.shape[0], errors.shape[1], errors.shape[2])
    
    return errors


        
model = test_cnn(X_train, X_test, y_train, y_test)

x = np.expand_dims(X_test, -1)
y = keras.utils.to_categorical(y_test)
errors = find_errors(model, x, y)
plt.imshow(errors[3], cmap = plt.get_cmap('gray'))




