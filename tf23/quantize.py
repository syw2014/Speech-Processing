#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : models.py
# PythonVersion: python3.6
# Date    : 2021/7/14 10:29
# Software: PyCharm
"""Functions for quantizing a trained keyword spotting model and saving to tflite.
    ref:https://github.com/ARM-software/ML-examples/tree/master/tflu-kws-cortex-m
"""

import argparse

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper

import data_process
import models
from test_tflite import tflite_test

tf.compat.v1.enable_eager_execution()
NUM_REP_DATA_SAMPLES = 100  # How many samples to use for post training quantization.


def quantize(model_settings, audio_processor, checkpoint, tflite_path):
    """Load our trained floating point model and quantize it.

    Post training quantization is performed and the resulting model is saved as a TFLite file.
    We use samples from the validation set to do post training quantization.

    Args:
        model_settings: Dictionary of common model settings.
        audio_processor: Audio processor class object.
        checkpoint: Path to training checkpoint to load.
        tflite_path: Output TFLite file save path.
    """
    model = models.create_model(model_settings, FLAGS.model_architecture, FLAGS.model_size_info)
    model.load_weights(checkpoint).expect_partial()

    val_data = audio_processor.get_data(audio_processor.Modes.validation).batch(1)

    def _rep_dataset():
        """Generator function to produce representative dataset."""
        i = 0
        for mfcc, label in val_data:
            if i > NUM_REP_DATA_SAMPLES:
                break
            i += 1
            yield [mfcc]

    # Quantize model and save to disk.
    tflite_model = post_training_quantize(model, _rep_dataset)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Quantized model saved to {tflite_path}.')


def post_training_quantize(keras_model, rep_dataset):
    """Perform post training quantization and returns the tflite model ready for saving.

    See https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization for
    more details.

    Args:
        keras_model: The trained tf Keras model used for post training quantization.
        rep_dataset: Function to use as a representative dataset, must be callable.

    Returns:
        Quantized TFLite model ready for saving to disk.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Int8 post training quantization needs representative dataset.
    # converter.representative_dataset = rep_dataset
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.post_training_quantize = True
    tflite_model = converter.convert()

    return tflite_model


def main():
    model_settings = models.prepare_model_settings(
        len(data_process.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_strides_ms, FLAGS.dct_coefficient_count)

    audio_processor = data_process.AudioProcessor(data_dir=FLAGS.data_dir,
                                                  silence_percentage=FLAGS.silence_percentage,
                                                  unknown_percentage=FLAGS.unknown_percentage,
                                                  wanted_words=FLAGS.wanted_words.split(','),
                                                  augment_percentage=FLAGS.augment_percentage,
                                                  model_settings=model_settings)

    tflite_path = f'{FLAGS.model_architecture}_quantized.tflite'

    # Load floating point model from checkpoint and quantize it.
    quantize(model_settings, audio_processor, FLAGS.checkpoint, tflite_path)

    # Test the newly quantized model on the test set.
    tflite_test(model_settings, audio_processor, tflite_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # *NOTE*, here we copy the parameter from train directly, there are some parameters may not used
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
    parser.add_argument("--checkpoint",
                        type=str,
                        default="./result/models/best/ds_cnn_0.988_ckpt",
                        help="checkpoint folder")
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
    parser.add_argument("--mlir_file",
                        type=str,
                        default="./cnn.mlir",
                        help="model parameters specified with different model")

    #    parser = argparse.ArgumentParser()
    FLAGS, _ = parser.parse_known_args()
    main()
