#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train_gradientTape.py
# PythonVersion: python3.6
# Date    : 2021/6/21 11:09
# Software: PyCharm
"""Implement train steps with gradient tape with tf2.3+ instead of tf.keras.model.compile and fit"""
import argparse
from pathlib import Path
import tensorflow as tf
import time
import numpy as np

import data_process
import models

from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
tf.compat.v1.enable_eager_execution()

#parser = argparse.ArgumentParser()

#FLAGS, _ = parser.parse_known_args()


class KeywordSpotting(object):
    """
    Keyword Spotting.
    """

    def __init__(self, enable_function):
        self.data_dir = FLAGS.data_dir
        self.check_dir = "./result/ck"

        self.model_settings = models.prepare_model_settings(
            len(data_process.prepare_words_list(FLAGS.wanted_words.split(","))),
            FLAGS.sample_rate,
            FLAGS.clip_duration_ms,
            FLAGS.window_size_ms,
            FLAGS.window_strides_ms,
            FLAGS.dct_coefficient_count)

        self.model = models.create_model(self.model_settings,
                                         FLAGS.model_architecture,
                                         FLAGS.model_size_info)
        self.audio_processor = data_process.AudioProcessor(data_dir=self.data_dir,
                                                           silence_percentage=FLAGS.silence_percentage,
                                                           unknown_percentage=FLAGS.unknown_percentage,
                                                           wanted_words=FLAGS.wanted_words.split(","),
                                                           model_settings=self.model_settings)

        # decay learning rate in a constant piecewise way
        training_steps_list = list(map(int, FLAGS.how_many_train_steps.split(",")))
        learning_rates_list = list(map(float, FLAGS.learning_rate.split(",")))
        lr_boundary_list = training_steps_list[:-1]  # only need values at which to change lr
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundary_list,
                                                                           values=learning_rates_list)

        # specify optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              models=self.model)

        # define loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.enable_function = enable_function

        # calculate epochs
        train_max_steps = np.sum(training_steps_list)
        #self.epochs = int(np.ceil(train_max_steps / FLAGS.eval_step_interval))
        self.epochs = 1

    def calcu_loss(self, y_pred, y_true):
        """

        Args:
            logits:
            targets:

        Returns:

        """
        loss = self.loss_object(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        return loss

    def train_step(self, input_x, input_y):
        """

        Args:
            input_x:
            input_y:

        Returns:

        """
        with tf.GradientTape() as tape:
            output = self.model(input_x)

            loss = self.loss_object(input_y, output)
        #
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, output

    def evaluate(self, dataset):
        """

        Args:
            dataset:

        Returns:

        """
        eval_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        loss = 0.0
        cnt = 0
        for step, (input_x, input_y) in enumerate(dataset):
            preds = self.model(input_x)
            loss += self.loss_object(input_y, preds)
            eval_acc.update_state(y_true=input_y, y_pred=preds)
            cnt += 1
        return loss, eval_acc

    def train(self):
        """
        Train process
        Returns:

        """

        # prepare datasets
        train_dataset = self.audio_processor.get_data(self.audio_processor.Modes.training,
                                                 FLAGS.background_frequency,
                                                 FLAGS.background_volume,
                                                 int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000))
        buffer_size = self.audio_processor.get_set_size(self.audio_processor.Modes.training)
        print("train set set:", buffer_size)
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size).repeat().batch(FLAGS.batch_size).prefetch(1)
        steps_per_epoch = buffer_size // FLAGS.batch_size

        val_dataset = self.audio_processor.get_data(self.audio_processor.Modes.validation)
        val_dataset = val_dataset.batch(FLAGS.batch_size).prefetch(1)

        if self.enable_function:
            self.train_step = tf.function(self.train_step)

        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        best_acc = 0

        for epoch in range(self.epochs):
            start_time = time.time()
            loss = 0
            # train on epoch
            for (step, (input_x, input_y)) in enumerate(train_dataset.take(steps_per_epoch)):
                loss, preds = self.train_step(input_x, input_y)
                train_acc.update_state(y_true=input_y, y_pred=preds)
                print(f"Epoch {epoch}/{self.epochs} training loss:{loss:.2f} training accuracy:{train_acc.result().numpy():0.3f}")
            # evaluate
            eval_loss, eval_acc = self.evaluate(val_dataset)
            print(f"Epoch {epoch}/{self.epochs} train loss:{loss:.2f} training accuracy:{train_acc.result().numpy():0.3f}"
                  f" evaluation loss:{eval_loss:0.2f} eval_accuracy:{eval_acc.result().numpy():0.3f}")
            t = float(eval_acc.result().numpy())
            if best_acc < t:
                best_acc = t
                # store
                self.checkpoint.save(self.check_dir)

    def save_mlir(self):
        manager = tf.train.CheckpointManager(self.checkpoint,
                                             directory=self.check_dir,
                                             max_to_keep=1)
        self.checkpoint.restore(manager.latest_checkpoint)

        fff = tf.function(self.model).get_concrete_function(tf.TensorSpec([None, 3920]), tf.float32)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(fff)

        input_tensors = [
            tensor for tensor in frozen_func.inputs
            if tensor.dtype != tf.resource
        ]

        output_tensors = frozen_func.outputs
        graph_def = run_graph_optimizations(
            graph_def,
            input_tensors,
            output_tensors,
            config=get_grappler_config(['pruning', 'function', 'constfold', 'shape', 'remap',
                                        'memory', 'common_subgraph_elimination', 'arithmetic',
                                        'loop', 'dependency', 'debug_stripper']),
            graph=frozen_func.graph)

        tf_mlir_graph = tf.mlir.experimental.convert_graph_def(graph_def)
        outfile = open("./result/kws.mlir", 'wb')
        outfile.write(tf_mlir_graph.encode())
        outfile.close()


def main():
    kws_object = KeywordSpotting(enable_function=True)
    print("Start train...")
    #kws_object.train()
    print("start convert")
    kws_object.save_mlir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str,
                        default="/data1/yw.shi/data/audio/",
                        help="directory of audio wav file")
    parser.add_argument("--background_volume",
                        type=float,
                        default=0.1,
                        help="how loud the background noise should be , between 0~1")
    parser.add_argument("--background_frequency",
                        type=float,
                        default=0.8,
                        help="how many of the training samples have background noise mixed in")

    parser.add_argument("--silence_percentage",
                        type=float,
                        default=10.0,
                        help="how many of the training data should be silence")
    parser.add_argument("--unknown_percentage",
                        type=float,
                        default=10.0,
                        help="how much of the training data should be unknown words")
    parser.add_argument("--time_shift_ms",
                        type=float,
                        default=100.0,
                        help="range to randomly shift the training audio by in time")

    parser.add_argument("--sample_rate",
                        type=int,
                        default=16000,
                        help="expected sample rate of the wavs")
    parser.add_argument("--clip_duration_ms",
                        type=int,
                        default=1000,
                        help="expected duration in milliseconds of the wavs")
    parser.add_argument("--window_size_ms",
                        type=float,
                        default=30.0,
                        help="how long each spectrogram timeslice is")
    parser.add_argument("--window_strides_ms",
                        type=float,
                        default=10.0,
                        help="how long each spectrogram timeslice is")
    parser.add_argument("--dct_coefficient_count",
                        type=int,
                        default=40,
                        help="how many bins to use for the mfcc fingerprint")
    parser.add_argument("--how_many_train_steps",
                        type=str,
                        default="15000, 3000",
                        help="how many training loops to run")
    parser.add_argument("--eval_step_interval",
                        type=int,
                        default=400,
                        help="how often to evaluate the training results")
    parser.add_argument("--learning_rate",
                        type=str,
                        default="0.001, 0.0001",
                        help="how large a learning rate to use when training.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="how many items to train with at once")
    parser.add_argument("--summaries_dir",
                        type=str,
                        default="./result/train_summary",
                        help="summary directory")
    parser.add_argument("--wanted_words",
                        type=str,
                        default="hixiaowen,nihaowenwen",
                        help="keywords list")
    parser.add_argument("--train_dir",
                        type=str,
                        default="./result/models",
                        help="train folder")
    parser.add_argument("--start_checkpoint",
                        type=str,
                        default="",
                        help="restore pre-trained model before any training")
    parser.add_argument("--model_architecture",
                        type=str,
                        default="cnn",
                        help="what model to use dnn/cnn/ds_cnn")
    parser.add_argument("--model_size_info",
                        type=int,
                        nargs="+",
                        default=[64, 8, 20, 1, 1, 64, 4, 20, 1, 1, 512, 512],
                        help="model parameters specified with different model")
    parser.add_argument("--mlir_file",
                        type=str,
                        default="/cnn.mlir",
                        help="model parameters specified with different model")

    #    parser = argparse.ArgumentParser()
    FLAGS, _ = parser.parse_known_args()
    # train model
    kws_object = KeywordSpotting(enable_function=True)
    print("Start train...")
    #kws_object.train()
    kws_object.save_mlir()
