#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling3D
from keras.optimizers import SGD

def simple_nn(x, y):
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(x.shape[1],)))

    # Add hidden layer s
    model.add(Dense(int(x.shape[1]/2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(x.shape[1]/4), activation='relu'))
    model.add(Dropout(0.5))

    # model.add(MaxPooling3D(5, activation='sigmoid'))

    # Add an output layer 
    model.add(Dense(y.shape[1], activation='softmax'))

    # Sotchastic gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
                       
    model.fit(tf.convert_to_tensor(x), tf.convert_to_tensor(y), epochs=20, steps_per_epoch=100, verbose=1)
    # model.fit(x, y, epochs=20, batch_size=10, verbose=1)

    # y_pred = model.predict(X_test)

    # score = model.evaluate(X_test, y_test,verbose=1)

    # print(score)