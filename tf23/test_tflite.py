#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : models.py
# PythonVersion: python3.6
# Date    : 2021/7/14 15:06
# Software: PyCharm
"""Functions to run inference and test keyword spotting models in tflite format.
    ref:https://github.com/ARM-software/ML-examples/tree/master/tflu-kws-cortex-m
"""

import argparse

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
import numpy as np

import data_process
import models
from test import calculate_accuracy
from tqdm import tqdm
import time
import os

tf.compat.v1.enable_eager_execution()


def create_test_data(feat_dir):
    """
    Load mfcc features from files instead of generated from tensorflow
    Args:
        feat_dir: directory of feature files, each file as a feature of one audio, feature dimension was 3920 the
                same as tensorflow. feature file format <name_label>, name and label was connect by '_'

    Returns:
        np.ndarray
    """
    # check folder was exist
    if not os.path.exists(feat_dir):
        raise ValueError("Required feature folder but found {}".format(feat_dir))

    file_list = os.listdir(feat_dir)
    print("Found {} feature file in folder".format(file_list))
    feature_array = []
    label_array = []
    for file in file_list:
        arr = file.split("_")
        if len(arr) != 2:
            print("Required feature file contain name and label but found:{}".format(file))
        label = arr[1]

        # open file and load features
        with open(feat_dir + "/" + file, 'r', encoding='utf-8') as f:
            feat = []
            for line in f.readlines():
                vals = line.strip().split(' ')
                vals = [float(x) for x in vals]
                feat.extend(vals)
            # check feature dimension was 3920
            if len(feat) != 3920:
                print("feature dimension was not `3920` but found {}".format(len(feat)))
                continue

            # make pair
            feature_array.append(feat)
            label_array.append(label)
    if len(feature_array) != len(label_array):
        raise ValueError("feature number was not the same as label number")
    print("Total found {} features.".format(len(feature_array)))
    # convert to tf.data.dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((feature_array, label_array))
    return test_dataset


def tflite_test(model_settings, audio_processor, tflite_path):
    """Calculate accuracy and confusion matrices on the test set.

    A TFLite model used for doing testing.

    Args:
        model_settings: Dictionary of common model settings.
        audio_processor: Audio processor class object.
        tflite_path: Path to TFLite file to use for inference.
    """
    # test_data = audio_processor.get_data(audio_processor.Modes.testing).batch(1)

    # test model with own feature
    feat_dir = ""
    test_data = create_test_data(feat_dir)

    expected_indices = np.concatenate([y for x, y in test_data])
    predicted_indices = []

    print("Running testing on test set, test size:{}...".format(len(expected_indices)))
    # define interpreter
    interpreter = interpreter_wrapper.Interpreter(tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # t_start = time.time()
    for mfcc, label in tqdm(test_data):
        t_start = time.time()
        print("mfcc", mfcc)
        prediction = tflite_inference(mfcc, interpreter, input_details, output_details)
        predicted_indices.append(np.squeeze(tf.argmax(prediction, axis=1)))
        t_end = time.time()
        print(f"Finish inference total samples{len(expected_indices)} cost time:{t_end - t_start}s")
    test_accuracy = calculate_accuracy(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices,
                                                num_classes=model_settings['label_count'])

    print(confusion_matrix.numpy())
    print(test_accuracy.numpy())
    print(f'Test accuracy = {test_accuracy.numpy() * 100:.2f}%'
          f'(N={audio_processor._set_size["testing"]})')


def tflite_inference(input_data, interpreter, input_details, output_details):
    """Call forwards pass of tflite file and returns the result.

    Args:
        input_data: input data to use on forward pass.
        tflite_path: path to tflite file to run.

    Returns:
        Output from inference.
    """
    # interpreter = interpreter_wrapper.Interpreter(model_path=tflite_path)
    # interpreter.allocate_tensors()

    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def main():
    model_settings = models.prepare_model_settings(len(data_process.prepare_words_list(FLAGS.wanted_words.split(','))),
                                                   FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                                                   FLAGS.window_strides_ms, FLAGS.dct_coefficient_count)

    audio_processor = data_process.AudioProcessor(data_dir=FLAGS.data_dir,
                                                  silence_percentage=FLAGS.silence_percentage,
                                                  unknown_percentage=FLAGS.unknown_percentage,
                                                  wanted_words=FLAGS.wanted_words.split(','),
                                                  augment_percentage=FLAGS.augment_percentage,
                                                  model_settings=model_settings)

    tflite_test(model_settings, audio_processor, FLAGS.tflite_path)


if __name__ == '__main__':
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
                        default=0.0,
                        help="how many of the training samples have background noise mixed in")

    parser.add_argument("--silence_percentage",
                        type=float,
                        default=30.0,
                        help="how many of the training data should be silence")
    parser.add_argument("--unknown_percentage",
                        type=float,
                        default=100.0,
                        help="how much of the training data should be unknown words")
    parser.add_argument("--augment_percentage",
                        default=30.0,
                        help="how much of the training data should be augment")
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
                        default="10000,10000, 10000",
                        help="how many training loops to run")
    parser.add_argument("--eval_step_interval",
                        type=int,
                        default=400,
                        help="how often to evaluate the training results")
    parser.add_argument("--learning_rate",
                        type=str,
                        default="0.0001, 0.0001, 0.00002",
                        help="how large a learning rate to use when training.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="how many items to train with at once")
    parser.add_argument("--summaries_dir",
                        type=str,
                        default="./result/train_summary",
                        help="summary directory")
    parser.add_argument("--wanted_words",
                        type=str,
                        default="nihaoxiaoshun,xiaoshunxiaoshun",
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
                        default="ds_cnn",
                        help="what model to use dnn/cnn/ds_cnn")
    parser.add_argument("--model_size_info",
                        type=int,
                        nargs="+",
                        default=[6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1,
                                 1, 276, 3, 3, 1, 1],
                        # default=[64, 20, 8, 1, 1, 64, 10, 4, 1, 1, 64, 10, 4, 1, 1, 256, 256],
                        help="model parameters specified with different model")
    parser.add_argument("--tflite_path",
                        type=str,
                        # default="./kwsh5.tflite",
                        default="ds_cnn_quantized.tflite",
                        help="model parameters specified with different model")

    #    parser = argparse.ArgumentParser()
    FLAGS, _ = parser.parse_known_args()
    main()
