#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : models.py
# PythonVersion: python3.6
# Date    : 2021/6/9 10:29
# Software: PyCharm
"""Design keyword spotting models with tf.keras apis."""

import math
import tensorflow as tf


def prepare_model_settings(label_count,
                           sample_rate,
                           clip_duration_ms,
                           window_size_ms,
                           window_stride_ms,
                           dct_coefficient_count):
    """
    Prepare model hyper-parameters for model design and data process
    Args:
        label_count: how many label words
        sample_rate: number of audio samples per second
        clip_duration_ms: length of each audio clip to be analyzed
        window_size_ms: duration of frequency analysis window
        window_stride_ms: how far move in time between frequency windows
        dct_coefficient_count: number of freqency bins to use for analysis

    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)  # 期望sample个数
    window_size_samples = int(sample_rate * window_size_ms / 1000)  # 单window内sample 个数
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)  # window 滑动几个sample
    length_minus_window = (desired_samples - window_size_samples)  # 期望和实际的差
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    # feature dimension
    fingerprint_size = dct_coefficient_count * spectrogram_length

    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
        "spectrogram_length": spectrogram_length
    }


def create_model(model_settings,
                 model_architecture,
                 model_size_info):
    """
    Builds a tf.keras model of the requested architecture compatible with the settings.
    Args:
        model_settings: dictionary of settings of information about the model
        model_architecture: string specifying which kind of model to use
        model_size_info: array with specific information for the chose architecture (convolution parameters, number of layers)

    Returns:
        A tf.keras Model with the requested architecture
    """
    if model_architecture == "dnn":
        return create_dnn_model(model_settings, model_size_info)
    elif model_architecture == "cnn":
        return create_cnn_model2(model_settings, model_size_info)
    elif model_architecture == "ds_cnn":
        return create_ds_cnn_model(model_settings, model_size_info)
    else:
        raise Exception(f"model architecture:{model_architecture} not recognized only support dnn/cnn/ds_cnn")


def create_dnn_model(model_settings, model_size_info):
    """
    Build models with multiple hidden fully-connected layers.
    Ref: https://arxiv.org/abs/1711.07128

    Args:
        model_settings: dictionary of settings of information about the model
        model_size_info: array with specific information for the chose architecture (convolution parameters, number of layers)

    Returns:
        A tf.keras Model with the requested architecture
    """
    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"],), name="input")

    # First fully connected layer
    x = tf.keras.layers.Dense(units=model_size_info[0], activation="relu")(inputs)

    # Hidden layers with Relu activation
    for i in range(1, len(model_size_info)):
        x = tf.keras.layers.Dense(units=model_size_info[i], activation="relu")(x)

    # Output fully connected layer
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    return tf.keras.Model(inputs, output)


def create_cnn_model(model_settings, model_size_info, training=None):
    """
    Build a standard convolution model.

    Args:
        model_settings: dictionary of settings of information about the model
        model_size_info: array with specific information for the chose architecture (convolution parameters, number of layers)
        ref: http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    Returns:
        A tf.keras Model with the requested architecture
    """

    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    # extract cnn hyper-parameters from model_size_info
    first_layer_count = model_size_info[0]
    first_layer_height = model_size_info[1]  # time axis
    first_layer_width = model_size_info[2]  # frequency axis
    first_layer_stride_y = model_size_info[3]  # time axis
    first_layer_stride_x = model_size_info[4]  # frequency axis

    # second layer
    second_layer_count = model_size_info[5]
    second_layer_height = model_size_info[6]  # time axis
    second_layer_width = model_size_info[7]  # frequency axis
    second_layer_stride_y = model_size_info[8]  # time axis
    second_layer_stride_x = model_size_info[9]  # frequency axis

    # third layer
    third_layer_count = model_size_info[10]
    third_layer_height = model_size_info[11]  # time axis
    third_layer_width = model_size_info[12]  # frequency axis
    third_layer_stride_y = model_size_info[13]  # time axis
    third_layer_stride_x = model_size_info[14]  # frequency axis

    linear_layer_size = model_size_info[15]
    fc_size = model_size_info[16]

    # Define layers
    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # For use 2D-convolution
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))
    

    # first conv layer
    x = tf.keras.layers.Conv2D(filters=first_layer_count,
                               kernel_size=(first_layer_height, first_layer_width),
                               strides=(first_layer_stride_y, first_layer_stride_x),
                               padding="SAME")(x)

    x = tf.keras.layers.ReLU()(x)
    # TODO, train or dev
    x = tf.keras.layers.Dropout(rate=0)(x, training=training)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding="SAME")(x)

    # Second convolution
    x = tf.keras.layers.Conv2D(filters=second_layer_count,
                               kernel_size=(second_layer_height, second_layer_width),
                               strides=(second_layer_stride_y, second_layer_stride_x),
                               padding="SAME")(x)

    x = tf.keras.layers.ReLU()(x)
    # TODO, train or dev
    x = tf.keras.layers.Dropout(rate=0.5)(x, training=training)

    # Third cnn layer
    """ 
    x = tf.keras.layers.Conv2D(filters=third_layer_count,
                               kernel_size=(third_layer_height, third_layer_width),
                               strides=(third_layer_stride_y, third_layer_stride_x),
                               padding="SAME")(x)

    # TODO, use bath norm
    #x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)
    # TODO, train or dev
    x = tf.keras.layers.Dropout(rate=0.5)(x, training=training)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding="SAME")(x)
    """
    # Flatten for fully connected layers
    x = tf.keras.layers.Flatten()(x)

    # Fully connected layer with no activation
    x = tf.keras.layers.Dense(units=linear_layer_size)(x)

    # Fully connected layer with Relu activation
    x = tf.keras.layers.Dense(units=fc_size,
            kernel_regularizer="l2")(x)

    # output
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def create_cnn_model2(model_settings, model_size_info, training=None):
    """
    Build a standard convolution model.

    Args:
        model_settings: dictionary of settings of information about the model
        model_size_info: array with specific information for the chose architecture (convolution parameters, number of layers)
        ref: http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    Returns:
        A tf.keras Model with the requested architecture
    """

    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    # extract cnn hyper-parameters from model_size_info
    first_layer_count = model_size_info[0]
    first_layer_height = model_size_info[1]  # time axis
    first_layer_width = model_size_info[2]  # frequency axis
    first_layer_stride_y = model_size_info[3]  # time axis
    first_layer_stride_x = model_size_info[4]  # frequency axis

    # second layer
    second_layer_count = model_size_info[5]
    second_layer_height = model_size_info[6]  # time axis
    second_layer_width = model_size_info[7]  # frequency axis
    second_layer_stride_y = model_size_info[8]  # time axis
    second_layer_stride_x = model_size_info[9]  # frequency axis

    # third layer
    # third_layer_count = model_size_info[10]
    # third_layer_height = model_size_info[11]  # time axis
    # third_layer_width = model_size_info[12]  # frequency axis
    # third_layer_stride_y = model_size_info[13]  # time axis
    # third_layer_stride_x = model_size_info[14]  # frequency axis

    linear_layer_size = model_size_info[15]
    fc_size = model_size_info[16]

    # Define layers
    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # For use 2D-convolution
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # first conv layer
    x = tf.keras.layers.Conv2D(filters=first_layer_count,
                               kernel_size=(first_layer_height, first_layer_width),
                               strides=(first_layer_stride_y, first_layer_stride_x),
                               padding="VALID")(x)
    # TODO, use bath norm
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # TODO, train or dev
    x = tf.keras.layers.Dropout(rate=0.1)(x, training=training)
    #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
    #                              strides=(2, 2),
    #                              padding="SAME")(x)

    # Second convolution
    x = tf.keras.layers.Conv2D(filters=second_layer_count,
                               kernel_size=(second_layer_height, second_layer_width),
                               strides=(second_layer_stride_y, second_layer_stride_x),
                               padding="VALID")(x)

    # TODO, use bath norm
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # TODO, train or dev
    x = tf.keras.layers.Dropout(rate=0.5)(x, training=training)

    #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
    #                              strides=(2, 2),
    #                              padding="SAME")(x)
    # Third cnn layer
    """ 
    x = tf.keras.layers.Conv2D(filters=third_layer_count,
                               kernel_size=(third_layer_height, third_layer_width),
                               strides=(third_layer_stride_y, third_layer_stride_x),
                               padding="SAME")(x)

    # TODO, use bath norm
    #x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)
    # TODO, train or dev
    x = tf.keras.layers.Dropout(rate=0.5)(x, training=training)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding="SAME")(x)
    """
    # Flatten for fully connected layers
    x = tf.keras.layers.Flatten()(x)

    # Fully connected layer with no activation
    x = tf.keras.layers.Dense(units=linear_layer_size)(x)

    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(units=linear_layer_size)(x)

    # Fully connected layer with Relu activation
    x = tf.keras.layers.Dense(units=fc_size,
                              kernel_regularizer="l2")(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x, training=training)

    # output
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def create_ds_cnn_model(model_settings, model_size_info):
    """
    Build a model with convolution and depthwise separable convolutional layers.
    Ref:https://arxiv.org/abs/1711.07128

    Args:
        model_settings: dictionary of settings of information about the model
        model_size_info: Defines number of layers, followed by the DS-Conv layer parameters in
        the order {number of conv features, conv filter height, width and stride in y,x dir.} for each of the layers

    Returns:
        A tf.keras Model with the requested architecture
    """
    label_count = model_settings["label_count"]
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    t_dim = input_time_size
    f_dim = input_time_size

    # Extract model dimensions from model_size_info
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers

    i = 1
    for layer_index in range(0, num_layers):
        conv_feat[layer_index] = model_size_info[i]
        i += 1
        conv_kt[layer_index] = model_size_info[i]
        i += 1
        conv_kf[layer_index] = model_size_info[i]
        i += 1
        conv_st[layer_index] = model_size_info[i]
        i += 1
        conv_sf[layer_index] = model_size_info[i]
        i += 1

    print("Model contains {} layers".format(num_layers))

    inputs = tf.keras.layers.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # reshape the flattened input, 4-d,
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # Depthwise separable convolutions
    layer_index = 0
    for layer_index in range(0, num_layers):
        if layer_index == 0:
            # First layer
            x = tf.keras.layers.Conv2D(filters=conv_feat[0],
                                       kernel_size=(conv_kt[0], conv_kf[0]),
                                       strides=(conv_st[0], conv_sf[0]),
                                       padding="SAME")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        else:
            # Depthwise convolution
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=(conv_kt[layer_index], conv_kf[layer_index]),
                                                strides=(conv_st[layer_index], conv_sf[layer_index]),
                                                padding="SAME")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Pointwise convolution
            x = tf.keras.layers.Conv2D(filters=conv_feat[layer_index],
                                       kernel_size=(1, 1))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        t_dim = math.ceil(t_dim / float(conv_st[layer_index]))
        f_dim = math.ceil(f_dim / float(conv_sf[layer_index]))

    # Global average pool
    x = tf.keras.layers.AveragePooling2D(pool_size=(t_dim, f_dim), strides=1)(x)

    # Squeeze before passing to output fully connected layer
    x = tf.reshape(x, shape=(-1, conv_feat[layer_index]))

    # Output fully connected layer
    output = tf.keras.layers.Dense(units=label_count, activation="softmax")(x)

    return tf.keras.Model(inputs, output)
