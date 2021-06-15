#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train.py
# PythonVersion: python3.6
# Date    : 2021/6/9 15:51
# Software: PyCharm
"""Train logic for keyword spotting"""

import argparse
from pathlib import Path

#import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

import numpy as np

import data_process
import models

#parser = argparse.ArgumentParser()
#FLAGS,_ = parser.parse_known_args()


def save_mlir(checkpoint, model_func, out_file):
    from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    fff = tf.function(model_func).get_concrete_function(tf.TensorSpec([1, 256, 256, 3]), tf.float32)
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
    outfile = open(out_file, 'wb')
    outfile.write(tf_mlir_graph.encode())
    outfile.close()




def train():
    model_settings = models.prepare_model_settings(len(data_process.prepare_words_list(FLAGS.wanted_words.split(","))),
                                                   FLAGS.sample_rate,
                                                   FLAGS.clip_duration_ms,
                                                   FLAGS.window_size_ms,
                                                   FLAGS.window_strides_ms,
                                                   FLAGS.dct_coefficient_count)

    # Create model
    model = models.create_model(model_settings,
                                FLAGS.model_architecture,
                                FLAGS.model_size_info)

    # Create checkpoint for convert mlir
    #checkpoint = tf.train.Checkpoint(model)
    print("test->", FLAGS.wanted_words)
    # audio processor
    audio_processor = data_process.AudioProcessor(data_dir=FLAGS.data_dir,
                                                  silence_percentage=FLAGS.silence_percentage,
                                                  unknown_percentage=FLAGS.unknown_percentage,
                                                  wanted_words=FLAGS.wanted_words.split(","),
                                                  model_settings=model_settings)

    # decaay learning rate in a constant piecewise way
    training_steps_list = list(map(int, FLAGS.how_many_train_steps.split(",")))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(",")))
    lr_boundary_list = training_steps_list[:-1]     # only need values at which to change lr
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundary_list,
                                                                       values=learning_rates_list)

    # specify optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    # prepare datasets
    train_dataset = audio_processor.get_data(audio_processor.Modes.training,
                                          FLAGS.background_frequency,
                                          FLAGS.background_volume,
                                          int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000))
    buffer_size = audio_processor.get_set_size(audio_processor.Modes.training)
    print("train set set:", buffer_size)
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).repeat().batch(FLAGS.batch_size).prefetch(1)

    val_dataset = audio_processor.get_data(audio_processor.Modes.validation)
    val_dataset = val_dataset.batch(FLAGS.batch_size).prefetch(1)

    # calculate how many epochs because we train for a max number of iterations
    train_max_steps = np.sum(training_steps_list)
    train_max_epochs = int(np.ceil(train_max_steps / FLAGS.eval_step_interval))

    # save models
    train_dir = Path(FLAGS.train_dir) / "best"
    train_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = train_dir / (FLAGS.model_architecture + "_{val_acc:.3f}_ckpt")
    model_checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint(
            filepath=(train_dir / (FLAGS.model_architecture+ "_{val_accuracy:.3f}_ckpt")),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True)

    # Train model
    model.fit(x=train_dataset,
              steps_per_epoch=FLAGS.eval_step_interval,
              validation_data=val_dataset,
              callbacks=[model_checkpoint_call_back],
              verbose=1,
              epochs=int(train_max_epochs/10))

    print("Training model finshed, start to test...")
    # test and save model
    test_dataset = audio_processor.get_data(audio_processor.Modes.testing)
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Final test accuracy:{test_acc:0.2f} loss:{test_loss:0.3f}")

    # =================checkpoint to mlir===================#
    # TODO, convert checkpoint to mlir format
    # checkpoint.restore(checkpoint_dir)
    # save_mlir(checkpoint, model, FLAGS.mlir_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--data_dir",
                        type=str,
                        default="/home/yw.shi/develop/projects/5.asr/data/mobvoi_hotwords_dataset",
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
    FLAGS, _  = parser.parse_known_args()
    # train model
    train()
