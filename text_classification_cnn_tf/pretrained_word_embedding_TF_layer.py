# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
from prepare_text import prepare_data
from data_utils import save_array, load_array

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Function to build a model.
    Accepts standard inputs: x, y and mode (TRAIN, INFER, EVAL. See below).
    Returns a ModelFnOps Object that will be passed to learn.Estimator
    """

    # Input Layer
    input_layer = tf.reshape(features, [-1, MAX_SEQUENCE_LENGTH])

    # embedding layer and look up
    embeddings = tf.get_variable(name="embeddings", shape=embedding_matrix.shape,
                                 initializer=tf.constant_initializer(embedding_matrix), trainable=False)
    embed = tf.nn.embedding_lookup(embeddings, input_layer)

    # ...and here we go with the 3 convolution, pooling and batch normalization layers
    conv1 = tf.layers.conv1d(inputs=embed, filters=128,kernel_size=5, padding="VALID", activation=tf.nn.relu)
    # one could add l2 regularization as:
    # conv1 = tf.layers.conv1d(inputs=embed, filters=128,kernel_size=5, padding="VALID",
    #     activation=tf.nn.relu, activity_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=5, strides=5, padding="VALID")
    bn1 = tf.layers.batch_normalization(pool1)

    conv2 = tf.layers.conv1d(inputs=bn1, filters=128, kernel_size=5, padding="VALID", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=5, strides=5, padding="VALID")
    bn2 = tf.layers.batch_normalization(pool2)

    conv3 = tf.layers.conv1d(inputs=bn2, filters=128, kernel_size=5, padding="VALID", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=35, strides=35, padding="VALID")
    bn3 = tf.layers.batch_normalization(pool3)

    # Dense Layer
    bn3_flat = tf.reshape(bn3, [-1, 128])
    dense = tf.layers.dense(inputs=bn3_flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=20)

    # defining and keep tracking of loss and optimization
    loss = None
    train_op = None

    # Calculate Loss (for TRAIN and EVAL modes): the result object when using
    # learn.Estimator with a ModelFnOps obj has the usual methods; fit,
    # evaluate and predict. When running the mode, the object will "inherit"
    # the ModeKeys, which is "simply" a class withing the model_fn module with
    # 3 attributes: 'TRAIN', 'EVAL' and 'INFER')
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=20)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="Adam")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


if __name__ == '__main__':

    GLOVE_DIR = 'glove.6B/'
    TEXT_DATA_DIR = '20_newsgroup/'
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    # prepare the dataset. Note that I set this experiment so the labels are
    # NOT onehot encoded a priori but within the cnn_model function. This is
    # simply to illustrate this functionality
    x_train, y_train, x_val, y_val, embedding_matrix = prepare_data(GLOVE_DIR,
        TEXT_DATA_DIR, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM,
        VALIDATION_SPLIT, categorical=False)

    # Note: steps and epochs: if you have 20,000 instances and a batch size of
    # 100 then the epoch should contain 20,000 / 100 = 200 steps. Therefore
    # here we have 15998 documents, and a bacth size of 128. This means that
    # one epoch is 125 steps and 10 epochs 1250 steps

    # Create the Estimator
    text_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="./model_layer")

    # Train the model
    text_classifier.fit(x=x_train,y=y_train,batch_size=128,steps=1250)

    # Configure the accuracy metric for evaluation
    metrics = {"accuracy":learn.MetricSpec(
        metric_fn=tf.metrics.accuracy, prediction_key="classes")}

    # Evaluate the model and print results
    eval_results = text_classifier.evaluate(x=x_val, y=y_val, metrics=metrics)
    print(eval_results)
