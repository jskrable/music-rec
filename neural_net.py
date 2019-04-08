#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import time
import datetime
import tensorflow as tf
import numpy as np
import keras
from keras import optimizers 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling3D
from keras.callbacks import TensorBoard

def simple_nn(X, y):

    # Globals
    # Lower the learning rate when using adam
    lr = 0.001
    OPT = 'sgd'

    y = keras.utils.to_categorical(y, num_classes=y.shape[0])

    # TODO reshape, stalling @ ~.4 accuracy
    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])
    in_size = X_train.shape[1]
    out_size = y.shape[0]

    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(in_size,)))

    # Add hidden layer s
    # model.add(Dense(in_size, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(in_size // 2, activation='relu'))
    model.add(Dropout(0.1))
    # model.add(Dense(in_size // 10, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(in_size // 4, activation='relu'))
    # model.add(Dropout(0.5))

    # model.add(MaxPooling3D(5, activation='sigmoid'))

    # Add an output layer 
    model.add(Dense(out_size, activation='softmax'))

    if OPT == 'sgd':
        opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.8, nesterov=True)
    elif OPT == 'adam':
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    t = time.time()
    dt = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    tensorboard = TensorBoard(log_dir='./logs/'+dt)                       

    print('Training...')    
    model.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), validation_data=(X_valid, y_valid), epochs=100, steps_per_epoch=168, validation_steps=25, verbose=1, shuffle=True, callbacks=[tensorboard])
    # model.fit(X, y, epochs=20, batch_size=10, verbose=1)

    print('EValuating...')
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test,verbose=1)

    print(score)

    print('Saving model...')

    path = './model/train/'
    model.save(str(path+OPT+'_'+dt+'.h5'))


def deep_nn(X,y):

    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])
    in_size = X_train.shape[1]
    out_size = y.shape[0]

    def shuffle_batch(X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    n_inputs = in_size 
    n_hidden1 = in_size // 4
    n_hidden2 = in_size // 8
    n_hidden3 = 50
    n_outputs = out_size

    learning_rate = 0.001

    n_epochs = 100
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
