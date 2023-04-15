
import tensorflow as tf

import numpy as np


def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32,
                               initializer=tf.constant_initializer(0.01))

    beta = tf.minimum(0.1, tf.maximum(beta, 0.01))

    return tf.maximum(x, beta * x)


def conv2d(input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        
    return conv


def deconv2d(input_, output_shape, kernel=5, stride=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, stride, stride, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

def batch_norm(x, epsilon=1e-5, momentum = 0.9, name="batch_norm", training=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=momentum, 
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=training,
                      scope=name)

def instance_norm(x, name='const_norm'):
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, 1e-9)))
    
def dropout(x, keep_prob=0.5, training=True):
    #prob = tf.cond(training, keep_prob, 1.0)
    return tf.nn.dropout(x, keep_prob=keep_prob)

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def relu(x, name='relu'):
    return tf.nn.relu(x)

def maxpool2d(x, kernel=2, stride=2, name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME')

def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias

def subpixel(X, r, n_out_channel):
    if n_out_channel >= 1:
        assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, 'Invalid Params'
        bsize, a, b, c = X.get_shape().as_list()
        bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
        Xs=tf.split(X,r,3) #b*h*w*r*r
        Xr=tf.concat(Xs,2) #b*h*(r*w)*r
        X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
    else:
        print('Invalid Dim.')
    return X

def dice_coeff(labels, predictions, weights=None, name="diceCoeff"):
    #with tf.variable_scope(name):
    TP = tf.metrics.true_positives(labels=labels,predictions=predictions,weights=weights)
    FN = tf.metrics.false_negatives(labels=labels,predictions=predictions,weights=weights)
    FP = tf.metrics.false_positives(labels=labels,predictions=predictions,weights=weights)
    dice_result = tf.divide(tf.add(TP,TP),tf.add(tf.add(FP,FN),tf.add(TP,TP)))

    return dice_result


def sensitivity(labels, predictions, weights=None, name="sensitivity"):
    #with tf.variable_scope(name):
    TP = tf.metrics.true_positives(labels=labels,predictions=predictions,weights=weights)
    FN = tf.metrics.false_negatives(labels=labels,predictions=predictions,weights=weights)
    sensitivity_result = tf.divide(TP,tf.add(TP, FN))

    return sensitivity_result


def specificity(labels, predictions, weights=None, name="specificity"):
    #with tf.variable_scope(name):
    FP = tf.metrics.false_positives(labels=labels,predictions=predictions,weights=weights)
    TN = tf.metrics.true_negatives(labels=labels,predictions=predictions,weights=weights)
    specificity_result = tf.divide(TN, tf.add(TN, FP))
    return specificity_result


def dilated_conv2d(input_, output_dim, kernel=5, rate=1, stddev=0.02, name="dilated_conv2d", padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.atrous_conv2d(input_, w, rate=rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

    return conv


def dilation_modules(input_, output_dim, rate=2, stddev=0.02, name="dilationMod", padding='SAME'):
    with tf.variable_scope(name):
        # for receptive field to be 240x240, it should be 2^(i+2)-1 where i=0 correspond the following line
        conv = dilated_conv2d(input_,output_dim, kernel=3, rate=1, name="dilatedInception0",padding='SAME')
        conv = batch_norm(conv,name="batchNorm0")
        conv = relu(conv,name="dilationRELU0")
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=1, name="dilatedInception1",padding='SAME')
        conv = batch_norm(conv,name="batchNorm1")
        conv = relu(conv,name="dilationRELU1")
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=2, name="dilatedInception2",padding='SAME')
        conv = batch_norm(conv,name="batchNorm2")
        conv = relu(conv,name="dilationRELU2")
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=4, name="dilatedInception3",padding='SAME')
        conv = batch_norm(conv,name="batchNorm3")
        conv = relu(conv,name="dilationRELU3")
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=8, name="dilatedInception4",padding='SAME')
        conv = batch_norm(conv,name="batchNorm4")
        conv = relu(conv,name="dilationRELU4")
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=16, name="dilatedInception5",padding='SAME')
        conv = batch_norm(conv,name="batchNorm5")
        conv = relu(conv,name="dilationRELU5")
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=32, name="dilatedInception6",padding='SAME')
        conv = batch_norm(conv,name="batchNorm6")
        conv = relu(conv,name="dilationRELU6")
        # following line corresonds to i=6
        conv = dilated_conv2d(conv,output_dim, kernel=3, rate=64, name="dilatedInception7",padding='SAME')
        conv = batch_norm(conv,name="batchNorm7")
        conv = relu(conv,name="dilationRELU7")

    return conv
