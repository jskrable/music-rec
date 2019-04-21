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
from keras import initializers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras import backend as K


def set_opt(OPT, lr):
    
    if OPT == 'sgd':
        opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, 
                             nesterov=True)
    elif OPT == 'adam':
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=1e-6, amsgrad=False)
    elif OPT == 'adamax':
        opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0)
    elif OPT == 'adagrad':
        opt = optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    elif OPT == 'adadelta':
        opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    return opt


def deep_nn(X, y, label, path=None):

    K.clear_session()

    # Globals
    lr = 0.01
    epochs = 5000
    batch_size = 52
    OPT = 'adadelta'

    t = time.time()
    dt = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    name = '_'.join([OPT, str(epochs), str(batch_size), dt])

    # Calculate class weights to improve accuracy 
    class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
    swm = np.array([class_weights[i] for i in y])

    # Convert target to categorical
    y = to_categorical(y, num_classes=len(class_weights))

    # Get input and output layer sizes from input data
    in_size = X.shape[1]

    # Hyperparams for tweaking
    hidden_1_size = 1024
    hidden_2_size = 156
    hidden_3_size = 20

    # Modify this when increasing artist list target
    out_size = y.shape[1]

    # Split up input to train/test/validation
    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.70 * len(X)), int(.85 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.70 * len(y)), int(.85 * len(y))])

    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, input_shape=(in_size,), activation='relu',
                    kernel_initializer='orthogonal'))

    # Add hidden 1
    model.add(Dense(in_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.2))

    # Add hidden 1
    model.add(Dense(hidden_1_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.2))

    # Repeat hidden 1
    model.add(Dense(hidden_1_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))

    # Add hidden 2
    model.add(Dense(hidden_2_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))

    # Repeat hidden 2
    model.add(Dense(hidden_2_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.05))

    # Repeat hidden 2
    model.add(Dense(hidden_2_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.05))

    # Add hidden 3
    model.add(Dense(hidden_3_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.05))

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
              metrics=['accuracy'],
              sample_weight_mode=swm)

    return model





