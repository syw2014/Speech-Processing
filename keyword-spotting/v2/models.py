#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : models.py
# PythonVersion: python3.6
# Date    : 2021/5/31 10:30
# Software: PyCharm
"""Implement models for keywords spotting with tf.keras(tf2.x)"""
import tensorflow as tf
import time
from absl import flags, app
from prepare_data import DataProcess


FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 10000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 128, 'Batch Size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_string('path', None, 'Path to the data folder')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')


def _next_power_of_two(x):
    """Calculates the smallest enclosing power of two for an input.

    Args:
      x: Positive float or integer number.

    Returns:
      Next largest power of two integer.
    """
    return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


# here we use tf2.3 API
def cnn_model(is_training=False):
    # define hyper-parameters
    input_time_size = 0
    input_frequency_size = 0
    label_count = 5

    # TODO, check the shape of input
    inputs = tf.keras.Input(shape=[None, -1, -1])
    inputs = tf.reshape(inputs, [-1, input_time_size, input_frequency_size, 1])

    # first layer
    x = tf.keras.layers.Conv2D(64, (8, 20), activation='relu', padding='same',
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                               bias_initializer=tf.keras.initializers.Zeros)(inputs)

    if is_training:
        x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='same', data_format=None)(x)

    # second layer
    x = tf.keras.layers.Conv2D(64, (4, 10),
                               activation='relu',
                               padding='same',
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                               bias_initializer=tf.keras.initializers.Zeros)(x)

    if is_training:
        x = tf.keras.layers.Dropout(rate=0.2)(x)
    # reshape
    conv_shape = x.get_shape()
    conv_output_width = conv_shape[2]
    conv_output_height = conv_shape[1]
    conv_output_count = int(conv_output_width * conv_output_height * conv_shape[-1])
    flatten = tf.reshape(x, [-1, conv_output_count])

    # classification
    output = tf.keras.layers.Dense(label_count, activation='softmax')(flatten)
    return tf.keras.Model(inputs=inputs, outputs=output)


class KeywordSpot(object):
    """KWS model for.
    
    Args:
        epochs: number of epochs
        enable_function: If true, train step is decorated with tf.function
        buffer_size: Shuffle buffer size 
        batch_size: batch size
    """
    
    def __init__(self, epochs, enable_function):
        self.epochs = epochs
        self.enable_function = enable_function
        self.label_count = 4
        self.loss_object = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.model = cnn_model()
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model)

    def model_loss(self, logits, targets):
        real_loss = self.loss_object(labels=targets, logits=logits)
        # TODO ,here can add L-Norm for regularization
        return real_loss

    def get_acc(self, logits, targets):
        predicted_indices = tf.argmax(input=logits, axis=1)
        correct_prediction = tf.equal(predicted_indices, targets)
        accuracy_score = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                              tf.float32))
        return accuracy_score

    def train_step(self, inputs, targets):
        """Train model with x epochs"""
        with tf.GradientTape() as tpe:
            logits = self.model(inputs, is_training=True)
            model_loss = self.model_loss(logits, targets)
        model_gradients = tpe.gradients(model_loss,
                                            self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(model_gradients,
                                               self.model.trainable_variables))

        return model_loss

    def train(self, dataset, checkpoint_dir):
        time_list = []
        if self.enable_function:
            self.train_step = tf.function(self.train_step)

        for epoch in range(self.epochs):
            start_time = time.time()
            total_acc = 0
            loss = 0
            for inputs, targets in dataset:
                loss = self.train_step(inputs, targets)
                # calculate accuracy
                acc = self.get_acc(inputs, targets)
                total_acc += acc

            wall_time_sec = time.time() - start_time
            time_list.append(wall_time_sec)

            # saving checnkpoint every 20 epochs
            if (epoch + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=checkpoint_dir)

            print("Epoch {}/{} loss: {} acc:{}".format(epoch, self.epochs, loss, total_acc))


def run_main(argv):
    del argv
    kwargs = kwargs = {'epochs': FLAGS.epochs, 'enable_function': FLAGS.enable_function,
            'path': FLAGS.path, 'buffer_size': FLAGS.buffer_size,
            'batch_size': FLAGS.batch_size}
    main(**kwargs)


def main(epochs, enable_function, path, buffer_size, batch_size):
    kws_object = KeywordSpot(epochs, enable_function)

    # create dataset
    # define parameters
    label_words = ["nihao xiaowen", "xiaowenxiaowen"]
    data_dir = ""
    data_processor = DataProcess(data_dir)
    data_processor.prepare_label_words(label_words)
    data_processor.get_ds_index()

    train_ds, dev_ds, test_ds = data_processor.create_dataset()

    # train model
    print("Training...")
    kws_object.train(train_ds, checkpoint_dir=path)


if __name__ == "__main__":
    app.run(main)
