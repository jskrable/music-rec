#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def simple_nn(x, y):
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(x.shape)))

    # Add one hidden layer 
    model.add(Dense(8, activation='relu'))

    # Add an output layer 
    model.add(Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                       
    model.fit(tf.convert_to_tensor(x), tf.convert_to_tensor(y), epochs=20, batch_size=1, verbose=1)

    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test,verbose=1)

    print(score)