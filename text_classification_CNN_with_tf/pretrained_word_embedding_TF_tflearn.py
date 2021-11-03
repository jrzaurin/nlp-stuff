# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
from prepare_text import prepare_data
from data_utils import save_array, load_array

import tflearn
import tensorflow as tf
from tflearn.layers import embedding
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

from time import time

if __name__ == '__main__':


    start = time()
    GLOVE_DIR = 'data/glove.6B/'
    TEXT_DATA_DIR = 'data/20_newsgroup/'
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    x_train, y_train, x_val, y_val, embedding_matrix = prepare_data(GLOVE_DIR,
        TEXT_DATA_DIR,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM,VALIDATION_SPLIT)

    # if you would like to save/load this arrays I would recommend use bcolz,
    # which is implemented through the utilities load_array and save_array
    # Example:
    # save_array(data/+'train_data.bc', x_train)
    # x_train = load_array(data/+'train_data.bc')
    # y_train = load_array(data/+'train_labels.bc')
    # x_val = load_array(data/+'val_data.bc')
    # y_val = load_array(data/+'val_labels.bc')
    # embedding_matrix = load_array(data/+'embedding_matrix.bc')

    # The syntax with tflearn is almost identical to keras. Only differences
    # are: there is no need to flatten a tensor before passing it to a
    # fully_connected layer, since the fc layer will take care  of that
    # automatically. Also, the default padding in tflearn is 'same', while in
    # keras is 'valid'.
    net = input_data(shape=[None,MAX_SEQUENCE_LENGTH], name='input')
    net = embedding(net, input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, trainable=False, name="EmbeddingLayer")
    net = conv_1d(net, 128, 5, 1, activation='relu', padding="valid")
    # one could add regularization as:
    # net = conv_1d(net, 128, 5, 1, activation='relu', regularizer="L2", padding="valid")
    net = max_pool_1d(net, 5, padding="valid")
    net = batch_normalization(net)
    net = conv_1d(net, 128, 5, activation='relu', padding="valid")
    net = max_pool_1d(net, 5, padding="valid")
    net = batch_normalization(net)
    net = conv_1d(net, 128, 5, activation='relu', padding="valid")
    net = max_pool_1d(net, 35)
    net = batch_normalization(net)
    net = fully_connected(net, 128, activation='relu')
    net = dropout(net, 0.5)
    net = fully_connected(net, y_train.shape[1], activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
    model.set_weights(embeddingWeights, embedding_matrix)
    model.fit({'input': x_train}, {'target': y_train}, validation_set=({'input': x_val}, {'target': y_val}),
        n_epoch=10, show_metric=True, batch_size = 128, run_id='conv_glove')
    model.save('model_tflearn/model')

    print(time()-start)