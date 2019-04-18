#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import os
import time
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras import optimizers 
from keras import regularizers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras import backend as K


def set_opt(OPT, lr):
    
    if OPT == 'sgd':
        opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.8, 
                             nesterov=True)
    elif OPT == 'adam':
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=1e-6, amsgrad=False)
    elif OPT == 'adamax':
        opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0)

    return opt


def deep_nn(X, y, label, path):

    K.clear_session()

    # Globals
    lr = 0.0001
    epochs = 300
    batch_size = 50
    OPT = 'adamax'

    t = time.time()
    dt = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    name = '_'.join([OPT, str(epochs), str(batch_size), dt])

    # Calculate class weights to improve accuracy 
    class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
    swm = np.array([class_weights[i] for i in y])

    # Convert target to categorical
    y = to_categorical(y, num_classes=len(class_weights))

    # Split up input to train/test/validation
    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

    # Get input and output layer sizes from input data
    in_size = X_train.shape[1]
    # Modify this when increasing artist list target
    out_size = y.shape[1]

    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(in_size,)))

    # Add hidden layers
    model.add(Dense(in_size // 2,
                    activation='relu',
                    # Regularize to reduce overfitting
                    activity_regularizer=regularizers.l1(1e-09),
                    kernel_regularizer=regularizers.l1(1e-07)))
    # Dropout to reduce overfitting
    model.add(Dropout(0.1))
    model.add(Dense(in_size // 4,
                    activation='relu',
                    kernel_regularizer=regularizers.l1(1e-08)))
    model.add(Dropout(0.1))
    model.add(Dense(in_size // 10,
                    activation='relu',
                    kernel_regularizer=regularizers.l1(1e-08)))

    # Add an output layer 
    model.add(Dense(out_size, activation='softmax'))


    opt = set_opt(OPT, lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  # metrics=['accuracy','msle'],
                  metrics=['accuracy'],
                  sample_weight_mode=swm)

    tensorboard = TensorBoard(log_dir=str('./logs/'+label+'/'+name+'.json'),
                              histogram_freq=1,
                              write_graph=True,
                              write_images=False)                 

    print('Training...')    
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
              epochs=epochs, batch_size=batch_size, verbose=1, 
              shuffle=True, callbacks=[tensorboard])

    print('Evaluating...')
    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)

    print('Saving model...')
    # Make directories to save model files
    # path = './model/train/' + dt
    # os.mkdir(path)
    path += ('/' + label)
    os.mkdir(path)
    # Save model structure json
    model_json = model.to_json()
    with open(path + '/model.json', 'w') as file:
        file.write(model_json)
    # Save weights as h5
    model.save_weights(path + '/weights.h5')
    # Save sample weight mode
    np.savetxt(path + '/sample_weights.csv', swm, delimiter=',')
    # Save hyperparams
    with open(path + '/hyperparams.csv', 'w') as file:
        file.write(','.join([str(lr), OPT]))
    print('Model saved to disk')

    return model


# Function to load model from disk
def load_model(path):

    # Get neural net architecture
    with open(path + '/model.json','r') as file:
        structure = file.read()
    model = model_from_json(structure)
    # Get weights
    model.load_weights(path + '/weights.h5')
    # Get sample weights for compiler
    swm = np.genfromtxt(path + '/sample_weights.csv', delimiter=',')
    # Get hyperparameters
    with open(path + '/hyperparams.csv', 'r') as file:
        lr, OPT = file.read().split(',')
    # Create custom optimizer
    opt = set_opt(OPT, float(lr))

    # Compile the loaded model
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy','msle'],
              sample_weight_mode=swm)

    return model





