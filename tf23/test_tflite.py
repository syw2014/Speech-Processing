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


def tflite_test(model_settings, audio_processor, tflite_path):
    """Calculate accuracy and confusion matrices on the test set.

    A TFLite model used for doing testing.

    Args:
        model_settings: Dictionary of common model settings.
        audio_processor: Audio processor class object.
        tflite_path: Path to TFLite file to use for inference.
    """
    test_data = audio_processor.get_data(audio_processor.Modes.TESTING).batch(1)
    expected_indices = np.concatenate([y for x, y in test_data])
    predicted_indices = []

    print("Running testing on test set...")
    for mfcc, label in test_data:
        prediction = tflite_inference(mfcc, tflite_path)
        predicted_indices.append(np.squeeze(tf.argmax(prediction, axis=1)))

    test_accuracy = calculate_accuracy(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices,
                                                num_classes=model_settings['label_count'])

    print(confusion_matrix.numpy())
    print(f'Test accuracy = {test_accuracy * 100:.2f}%'
          f'(N={audio_processor._set_size(audio_processor.Modes.testing)})')


def tflite_inference(input_data, tflite_path):
    """Call forwards pass of tflite file and returns the result.

    Args:
        input_data: input data to use on forward pass.
        tflite_path: path to tflite file to run.

    Returns:
        Output from inference.
    """
    interpreter = interpreter_wrapper.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def main():
    model_settings = models.prepare_model_settings(len(data_process.prepare_words_list(FLAGS.wanted_words.split(','))),
                                                   FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                                                   FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

    audio_processor = data_process.AudioProcessor(data_dir=FLAGS.data_dir,
                                                  silence_percentage=FLAGS.silence_percentage,
                                                  unknown_percentage=FLAGS.unknown_percentage,
                                                  wanted_words=FLAGS.wanted_words.split(','),
                                                  augment_percentage=FLAGS.augment_percentage,
                                                  model_settings=model_settings)

    tflite_test(model_settings, audio_processor, FLAGS.tflite_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
        Where to download the speech training data to.
        """)
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='',
        help='Path to tflite file to use for testing.')
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be silence.
        """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be unknown words.
        """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )

    FLAGS, _ = parser.parse_known_args()
    main()
