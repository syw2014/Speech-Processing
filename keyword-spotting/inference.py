#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : inference.py
# PythonVersion: python3.6
# Date    : 2021/4/17 15:03
# Software: PyCharm
"""Inference for keyword spotting"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import input_data
import models


def run_inference(wanted_words, sample_rate, clip_duration_ms,
                  window_size_ms, window_stride_ms, feature_bin_count,
                  model_architecture, preprocess):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      feature_bin_count: Number of frequency bins to use for analysis.
      model_architecture: Name of the kind of model to generate.
      preprocess: How the spectrogram is processed to produce features.
    """

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    sess = tf.InteractiveSession()

    # Define model setting config
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(wanted_words.split(','))),
        sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, feature_bin_count, preprocess)

    # Define audio processor
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir)

    label_count = model_settings['label_count']
    fingerprint_size = model_settings['fingerprint_size']

    # Define input placeholder
    fingerprint_input = tf.compat.v1.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    # Create model
    logits = models.create_model(
        fingerprint_input,
        model_settings,
        model_architecture,
        is_training=False)

    ground_truth_input = tf.compat.v1.placeholder(
        tf.int64, [None], name='groundtruth_input')

    predicted_indices = tf.argmax(input=logits, axis=1)
    expected_indices = tf.argmax(input=ground_truth_input, axis=1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)

    # training set
    set_size = audio_processor.set_size('training')
    tf.compat.v1.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'training', sess)
        train_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (train_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.compat.v1.logging.warn('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.compat.v1.logging.warn('Final Training accuracy = %.1f%% (N=%d)' %
                              (total_accuracy * 100, set_size))

    # validation set
    set_size = audio_processor.set_size('validation')
    tf.compat.v1.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess)
        validation_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.compat.v1.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.compat.v1.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                    (total_accuracy * 100, set_size))

    # test set
    set_size = audio_processor.set_size('testing')
    tf.compat.v1.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (test_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.compat.v1.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.compat.v1.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                       set_size))


def main(_):
    # Create the model, load weights from checkpoint and run on train/val/test
    run_inference(FLAGS.wanted_words, FLAGS.sample_rate,
                  FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                  FLAGS.window_stride_ms, FLAGS.feature_bin_count,
                  FLAGS.model_architecture, FLAGS.preprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/yw.shi/projects/5.asr/data/mobvoi_hotwords_dataset',
        help="""\
         Where to download the speech training data to.
         """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
         How loud the background noise should be, between 0 and 1.
         """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
         How many of the training samples have background noise mixed in.
         """)
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
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
         Range to randomly shift the training audio by in time.
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
        help='How long each spectrogram timeslice is.', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectrogram timeslices.',
    )
    parser.add_argument(
        '--feature_bin_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='15000,3000',
        help='How many training loops to run', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/home/yw.shi/projects/5.asr/data/mobvoi_hotwords_dataset/result2/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        # default='你好小问,出门问问',
        default='nihaoxiaowen,chumenwenwen',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/home/yw.shi/projects/5.asr/data/mobvoi_hotwords_dataset/result2/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--quantize',
        type=bool,
        default=False,
        help='Whether to train the model for eight-bit deployment')
    parser.add_argument(
        '--preprocess',
        type=str,
        default='mfcc',
        help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')


    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
          value: A member of tf.logging.
        Raises:
          ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == 'DEBUG':
            return tf.compat.v1.logging.DEBUG
        elif value == 'INFO':
            return tf.compat.v1.logging.INFO
        elif value == 'WARN':
            return tf.compat.v1.logging.WARN
        elif value == 'ERROR':
            return tf.compat.v1.logging.ERROR
        elif value == 'FATAL':
            return tf.compat.v1.logging.FATAL
        else:
            raise argparse.ArgumentTypeError('Not an expected value')


    parser.add_argument(
        '--verbosity',
        type=verbosity_arg,
        default=tf.compat.v1.logging.INFO,
        help='Log verbosity. Can be "DEBUG", "INFO", "WARN", "ERROR", or "FATAL"')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='gradient_descent',
        help='Optimizer (gradient_descent or momentum)')

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
