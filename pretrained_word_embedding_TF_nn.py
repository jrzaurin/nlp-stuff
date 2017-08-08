# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
from prepare_text import prepare_data
from data_utils import get_batch


def conv1d_layer(inp, filter_shape):
    """This is a 1d conv, so filter_shape = [dim, input_dim, out_dim]"""
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01))
    b = tf.Variable(tf.random_normal(shape=[filter_shape[2]]))
    # or you could initialize it as constant
    # b = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
    x = tf.nn.conv1d(inp,W,stride=1,padding="VALID")
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x


def max_pool1d_layer(inp, ksize, strides):
    """tf.nn does not have max_pool_1d, so we have to expand the incoming layer
    as if we were dealing with a 2D convolution and then squeeze it again.
    Again, since this is a 1D conv, the size of the window (ksize) and the stride
    of the sliding window must have only one dimension (height) != 1
    """
    x = tf.expand_dims(inp, 3)
    x = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="VALID")
    x = tf.squeeze(x, [3])
    return x


def batch_norm_layer(inp):
    """As explained in A. Gerón's book, in the default batch_normalization
    there is no scaling, i.e. gamma is set to 1. This makes sense for layers
    with no activation function or ReLU (like ours), since the next layers
    weights can take care of the scaling. In other circumstances, include
    scaling
    """
    # get the size from input tensor (remember, 1D convolution -> input tensor 3D)
    size = int(inp.shape[2])

    batch_mean, batch_var = tf.nn.moments(inp,[0])
    scale = tf.Variable(tf.ones([size]))
    beta  = tf.Variable(tf.zeros([size]))
    x = tf.nn.batch_normalization(inp,batch_mean,batch_var,beta,scale,
        variance_epsilon=1e-3)
    return x


def dense_layer(inp, n_neurons):
    # input to a fully connected layer -> 2D [batch_size, n_inputs]
    n_inputs = int(inp.shape[1])
    W = tf.Variable(tf.truncated_normal((n_inputs,n_neurons), stddev=0.1))
    b = tf.Variable(tf.random_normal(shape=[n_neurons]))
    # or if you prefer
    # b = tf.Variable(tf.zeros([n_neurons]))
    x = tf.matmul(inp,W) + b
    x = tf.nn.relu(x)
    return x


if __name__ == '__main__':

    GLOVE_DIR = 'glove.6B/'
    TEXT_DATA_DIR = '20_newsgroup/'
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    x_train, y_train, x_val, y_val, embedding_matrix = prepare_data(GLOVE_DIR,
        TEXT_DATA_DIR, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM,
        VALIDATION_SPLIT)

    n_labels = y_train.shape[1]

    batch_size=128
    graph = tf.Graph()
    with graph.as_default():
        l2_loss = tf.constant(0.0)

        # we'll use scopes so they will be saved nicely in corresponding "directories"
        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.int32, shape=[None,MAX_SEQUENCE_LENGTH], name = "x_inp")
            y = tf.placeholder(tf.float32, shape=[None,20], name="y_out")
            is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

            # Note: If you prefer to one_hot encode in "train" pass
            # "categorical=False" to prepare_data() before and and define "y" here as:
            # y = tf.placeholder(tf.int64, shape=[None], name="y_out")
            # Then see the "train" scope below.

        with tf.name_scope("embedding"):
            embeddings = tf.get_variable(name="embeddings", shape=embedding_matrix.shape,
                initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embed = tf.nn.embedding_lookup(embeddings, X)

        with tf.name_scope("conv1"):
            conv1 = conv1d_layer(embed, filter_shape=[5, 100, 128])

        with tf.name_scope("pool1"):
            pool1 = max_pool1d_layer(conv1, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1])

        with tf.name_scope("bn1"):
            bn1 = batch_norm_layer(pool1)

        with tf.name_scope("conv2"):
            conv2 = conv1d_layer(bn1, filter_shape=[5, 128, 128])

        with tf.name_scope("pool2"):
            pool2 = max_pool1d_layer(conv2, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1])

        with tf.name_scope("bn2"):
            bn2 = batch_norm_layer(pool2)

        with tf.name_scope("conv3"):
            conv3 = conv1d_layer(bn2, filter_shape=[5, 128, 128])

        with tf.name_scope("pool3"):
            # global maxpooling
            pool3 = max_pool1d_layer(conv3, ksize=[1, 35, 1, 1], strides=[1, 35, 1, 1])

        with tf.name_scope("bn3"):
            bn3 = batch_norm_layer(pool3)

            # We flatten it in preparation for the dense layer
            bn3_flat = tf.reshape(bn3, shape=[-1, 128])

        with tf.name_scope("fc"):
            fc = dense_layer(bn3_flat, n_neurons=128)

            # Here is the only time I am going to use layers, this is because I
            # believe the the default dropout does not switch off during testing
            fc_drop = tf.layers.dropout(fc, 0.5, training=is_training)

        with tf.name_scope("output"):
            # Here explicitly define the output layer with l2 reg
            W_o = tf.Variable(tf.truncated_normal((128,20), stddev=0.1), name="weights")
            b_o = tf.Variable(tf.zeros([n_labels]), name="biases")
            l2_loss += tf.nn.l2_loss(W_o)
            l2_loss += tf.nn.l2_loss(b_o)
            scores = tf.nn.softmax(tf.matmul(fc_drop,W_o) + b_o)

        with tf.name_scope("train"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y)
            loss = tf.reduce_mean(losses) + 0.001 * l2_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            training_op = optimizer.minimize(loss)

            # Note: if you prefer to one_hot encode here:
            # onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=20)
            # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=scores)
            # optimizer = tf.train.AdamOptimizer()
            # training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            predictions = tf.argmax(scores, 1, name="predictions")
            correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # Note: if you prefer to one_hot encode in "train":
            # probabilities = tf.nn.softmax(scores, name="probabilities")
            # classes = tf.argmax(input=scores, axis=1)
            # accuracy = tf.contrib.metrics.accuracy(y,classes)

        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()


    with tf.Session(graph=graph) as sess:
        init.run()
        batch_iter = get_batch(x_train, y_train)
        for epoch in range(10):
            for iteration in range(x_train.shape[0] // batch_size):
                X_batch, y_batch = batch_iter.next_batch(128)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, is_training: True})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: x_val, y: y_val})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            save_path = saver.save(sess, "model_nn/model")


