#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling3D
from keras.optimizers import SGD

def simple_nn(X, y):
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(X.shape[1],)))

    # Add hidden layer s
    model.add(Dense(X.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(X.shape[1]/2), activation='relu'))
    # model.add(Dense(int(X.shape[1]/4), activation='relu'))
    # model.add(Dropout(0.5))

    # model.add(MaxPooling3D(5, activation='sigmoid'))

    # Add an output layer 
    model.add(Dense(y.shape[0], activation='softmax'))

    # Sotchastic gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
                       
    model.fit(tf.convert_to_tensor(X), tf.convert_to_tensor(y), epochs=100, steps_per_epoch=128, verbose=1)
    # model.fit(X, y, epochs=20, batch_size=10, verbose=1)

    # y_pred = model.predict(X_test)

    # score = model.evaluate(X_test, y_test,verbose=1)

    # print(score)


def deep_nn(X,y):


    X_valid, X_train = X[:int(X.shape[0]/2)], X[int(X.shape[0]/2):]
    y_valid, y_train = y[:int(y.shape[0]/2)], y[int(y.shape[0]/2):]

    def shuffle_batch(X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    n_inputs = X.shape[1]  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_hidden3 = 50
    n_outputs = y.shape[1]

    learning_rate = 0.001

    n_epochs = 20
    batch_size = 200

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden_layer_1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden_layer_1")
        hidden_layer_2 = tf.layers.dense(hidden_layer_1, n_hidden2, activation=tf.nn.relu, name="hidden_layer_2")
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, n_hidden3, activation=tf.nn.relu, name="hidden_layer_3")
        logits = tf.layers.dense(hidden_layer_3, n_outputs, name="outputs")

        tf.summary.histogram('hidden_layer_1', hidden_layer_1)
        tf.summary.histogram('hidden_layer_2', hidden_layer_2)
        tf.summary.histogram('hidden_layer_3', hidden_layer_3)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    merged_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()

    # means = X_train.mean(axis=0, keepdims=True)
    # stds = X_train.std(axis=0, keepdims=True) + 1e-10
    # X_val_scaled = (X_valid - means) / stds

    train_saver = tf.summary.FileWriter('./model/train', tf.get_default_graph())  # async file saving object
    test_saver = tf.summary.FileWriter('./model/test')  # async file saving object

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                # X_batch_scaled = (X_batch - means) / stds
                summaries, _ = sess.run([merged_summaries, training_op], feed_dict={X: X_batch, y: y_batch})
            train_saver.add_summary(summaries, epoch)
            _, acc_batch = sess.run([merged_summaries, accuracy], feed_dict={X: X_batch, y: y_batch})
            train_summaries, acc_valid = sess.run([merged_summaries, accuracy], feed_dict={X: X_valid, y: y_valid})
            test_saver.add_summary(train_summaries, epoch)
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        train_saver.flush()
