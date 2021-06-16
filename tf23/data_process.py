#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : data_process.py
# PythonVersion: python3.6
# Date    : 2021/6/7 14:12
# Software: PyCharm
"""To process audio data for keyword spotting with tf2.3.x
Ref: https://github.com/ARM-software/ML-examples/tree/master/tflu-kws-cortex-m
"""

import os
import math
import random
import json
from pathlib import Path
from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()

# Pre-define parameters
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # 134M
RANDOM_SEED = 59185
BACKGROUND_NOISE_DIR_NAME = "_background_noise_"  # directory name of noise, here we use noise download from internet
SILENCE_LABEL = "_silence_"  # define silence label which means no keywords in audio
SILENCE_INDEX = 0  # the index of label
UNKNOWN_LABEL = "_unknown_"  # unknown label means others words in audio
UNKNOWN_INDEX = 1  # index


def load_wav_file(wav_filename, desired_samples):
    """
    Loads and decodes a given a 16bit PCM wav file to a float tensor
    Args:
        wav_filename:  16bit PCM wav filename of audio
        desired_samples: number of samples from the audio fiel.

    Returns:
        Tuple consisting of the decoded audio and sample rate
    """
    wav_file = tf.io.read_file(wav_filename)  # binary file

    # *Notes*, this api has been changed in tf2.5 as tf.audio.decode_wav, should be test.
    decoded_wav = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=desired_samples)

    return decoded_wav.audio, decoded_wav.sample_rate


def calculate_mfcc(audio_signal, audio_sample_rate, window_size, window_stride, num_mfcc):
    """
    Calculate MFCC(Mel Frequency Cepstral Coefficient) for a given audio signal
    Args:
        audio_signal: Raw audio signal in range [-1, 1]
        audio_sample_rate: sample rate for signal
        window_size: window size in samples for calculating spectrogram
        window_stride: window stride
        num_mfcc: number of mfcc features

    Returns:
        calculated mfcc feature

    """
    spectrogram = audio_ops.audio_spectrogram(input=audio_signal, window_size=window_size, stride=window_stride,
                                              magnitude_squared=True)
    mfcc_features = audio_ops.mfcc(spectrogram, audio_sample_rate, dct_coefficient_count=num_mfcc)

    # *Note*, api has been changed in tf2.x > tf2.3
    # TODO add another implementation

    return mfcc_features


def prepare_words_list(wanted_words):
    """
    Prepare label words list, include silence and unknown words
    Args:
        wanted_words: label list

    Returns:
        word list.
    """
    return [SILENCE_LABEL, UNKNOWN_LABEL] + wanted_words


class AudioProcessor(object):
    """
    Prepare audio training data.
    """

    # define run mode train/valid/test
    class Modes(Enum):
        training = 1
        validation = 2
        testing = 3

    def __init__(self, data_dir,
                 silence_percentage,
                 unknown_percentage,
                 wanted_words,
                 model_settings):
        """
        Data preprocess, 1) load data 2) add background. Define nessary variables for model
        Args:
            data_dir: directory of data
            silence_percentage: how many percent of audio data used as silence
            unknown_percentage: how many percent of audio data used as unknown
            wanted_words: keywords list
            model_settings: parameters for kws model
        """
        self.data_dir = data_dir
        self.model_setting = model_settings
        self.words_list = prepare_words_list(wanted_words)

        self._tf_datasets = {}  # tf.dataset for model input
        self.background_data = {}  # back ground data dict
        self._set_size = {"training": 0, "validation": 0, "testing": 0}  # data size dict

        # process logic
        self._prepare_datasets(silence_percentage, unknown_percentage, wanted_words)
        print("Start process background data...")
        self._prepare_background_data()
        print("End process background data")

    def get_data(self, mode, background_frequency=0.0, background_volume_range=0.0, time_shift=0.0):
        """
        Returns the train,validation, test set fro kws model as a TF dataset.
        Args:
            mode: the set will be run training/validation/testing
            background_frequency: how many of the samples have background noise mixed in
            background_volume_range: how loud the background nose should be, 0.0~1.0
            time_shift: range to randomly shift the training audio by in time.

        Returns:
            TF dataset that will generate tuples containing mfcc feature and label index

        Raises:
            value error: if mode not recognized
        """
        if mode == AudioProcessor.Modes.training:
            dataset = self._tf_datasets["training"]
        elif mode == AudioProcessor.Modes.validation:
            dataset = self._tf_datasets["validation"]
        elif mode == AudioProcessor.Modes.testing:
            dataset = self._tf_datasets["testing"]
        else:
            ValueError(f"mode:{mode} not recognized, only support training/validation/testing")

        use_background = (self.background_data != []) and (mode == AudioProcessor.Modes.training)
        dataset = dataset.map(lambda path, label: self._process_wavfile(path, label,
                                                                        self.model_setting,
                                                                        background_frequency,
                                                                        background_volume_range,
                                                                        time_shift,
                                                                        use_background,
                                                                        self.background_data),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def _assign_files(self):
        """
        Assign file in each train/valid/test index
        Returns:
            data_index:
            unknown_index:
            all_words:
        """
        data_index = {"training": [], "validation": [], "testing": []}
        unkonwn_index = {"training": [], "validation": [], "testing": []}
        all_words = {}

        # Note, here we process mobvoi_hot_dataset, should change with your own data
        resource_files = os.path.join(self.data_dir + "/mobvoi_hotword_dataset_resources", "*.json")
        print("test->", resource_files)
        print("test->", tf.io.gfile.glob(resource_files))
        # TODO, gfile to use system package not tf
        for json_path in tf.io.gfile.glob(resource_files):
            print(json_path)
            _, filename = os.path.split(json_path)
            # choose which set and whether unknown
            set_index = ""
            if filename.find("train") != -1:
                set_index = "training"
            elif filename.find("dev") != -1:
                set_index = "validation"
            elif filename.find("test") != -1:
                set_index = "testing"
            else:
                raise ValueError("Unknown mode found in filename: {}".format(filename))
            unknown = False if filename.startswith("p") else True

            with open(json_path) as f:
                jdata = json.load(f)
                for x in jdata:
                    if x["keyword_id"] == 0:
                        word = "hixiaowen"
                    elif x["keyword_id"] == 1:
                        word = "nihaowenwen"
                    elif x["keyword_id"] == -1:
                        word = UNKNOWN_LABEL
                    wav_path = os.path.join(self.data_dir, "temp/mobvoi_hotword_dataset/" + x["utt_id"] + ".wav")
                    if unknown:
                        unkonwn_index[set_index].append({"label": word, "file": wav_path})
                    else:
                        data_index[set_index].append({"label": word, "file": wav_path})

                    all_words[word] = True
        return data_index, unkonwn_index, all_words

    def _prepare_datasets(self,
                          silence_percentage,
                          unknown_percentage,
                          wanted_words):
        """
        Load audio data and convert to tf.dataset
        Args:
            silence_percentage:
            unknown_percentage:
            wanted_words:

        Returns:

        """
        # Make sure the shuffling adn picking of unknowns is deterministic
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, word in enumerate(wanted_words):
            wanted_words_index[word] = index + 2  # 0: silence, 1: unknown

        data_index, unknown_index, all_words = self._assign_files()

        for index, word in enumerate(wanted_words):
            if word not in all_words:
                raise Exception("Tried to find word:{} in labels but only found: {}".format(word, all_words.keys()))

        # assign index to each label word
        word_to_index = {}
        for word in all_words:
            if word in wanted_words:
                word_to_index[word] = wanted_words_index[word]
            else:
                word_to_index[word] = UNKNOWN_INDEX
        word_to_index[SILENCE_LABEL] = SILENCE_INDEX

        # we need an arbitrary file to load as the input for the silence samples
        # It's multiplied by zero later ,so the content doesn't matter
        silence_wav_path = data_index["training"][0]["file"]
        for set_index in ["training", "validation", "testing"]:
            set_size = len(data_index[set_index])  # get the size of set index
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                data_index[set_index].append(
                    {"label": SILENCE_LABEL, "file": silence_wav_path})

            # pick some unknonwns to add  to each partition of the dataset
            random.shuffle(unknown_index[set_index])
            unknonw_size = int(math.ceil(set_size * unknown_percentage / 100))
            # TODO, here we use all unknonwn
            data_index[set_index].extend(unknown_index[set_index][:unknonw_size])
            # size of set index after adding silence and unknown samples
            self._set_size[set_index] = len(data_index[set_index])

            # shuffle
            random.shuffle(data_index[set_index])

            # Transform into TF Datasets ready for easier processing later
            labels, paths = list(zip(*[d.values() for d in data_index[set_index]]))
            # convert label to label id
            labels = [word_to_index[w] for w in labels]
            self._tf_datasets[set_index] = tf.data.Dataset.from_tensor_slices((list(paths), labels))

    def _prepare_background_data(self):
        """
        Load background audio into memory
        Returns:

        """
        background_data = []
        background_dir = self.data_dir + "/" + BACKGROUND_NOISE_DIR_NAME
        if not os.path.exists(background_dir):
            raise Exception("Noise data directory: {} not exist.".format(background_dir))
            return

        search_path = os.path.join(background_dir + "/", '*.wav')
        for wav_path in tf.io.gfile.glob(search_path):
            wav_data, _ = load_wav_file(wav_path, desired_samples=-1)
            background_data.append(tf.reshape(wav_data, [-1]))

        print("Test->", background_data)
        if not background_data:
            raise Exception("No background wav files were found in " + search_path)

        # ragged tensor as wee cant use lists in tf dataset map functions
        self.background_data = tf.ragged.stack(background_data)

    def get_set_size(self, mode):
        """
        Get set size in different mode
        Args:
            mode: train/valid/test

        Returns:
            Returns the number of samples in the partition
        Raises:
            ValueError: If mode is not recognised
        """
        if mode == AudioProcessor.Modes.training:
            return self._set_size["training"]
        elif mode == AudioProcessor.Modes.validation:
            return self._set_size["validation"]
        elif mode == AudioProcessor.Modes.testing:
            return self._set_size["testing"]
        else:
            ValueError("Not recognised mode: {}".format(mode))

    @staticmethod
    def _process_wavfile(wavpath,
                         label,
                         model_settings,
                         background_frequency,
                         background_volume_range,
                         time_shift_samples,
                         use_background,
                         background_data):
        """
        Load wav file  and calculate mfcc features
        Args:
            wavpath: path of audio file
            label: label index of audio
            model_settings: Dictionary of settings for model being trained
            background_frequency: how many clips will have background noise, 0.0~1.0
            background_volume_range: how loud the background noise will be.
            time_shift_samples: how much to randomly shift the clips by
            use_background: Add in background noise to audio clips or not.
            background_data: Ragged tensor of loaded background noise samples.

        Returns:
            Tuple of calculated flattened mfcc features and it's class label.
        """
        desired_samples = model_settings["desired_samples"]
        audio_tensor, sample_rate = load_wav_file(wavpath, desired_samples=desired_samples)

        # Make out own silence audio data.
        if label == SILENCE_INDEX:
            audio_tensor = tf.multiply(audio_tensor, 0)

        # Shift samples start position and pad any gaps with zeros
        if time_shift_samples > 0:
            time_shift_amount = tf.random.uniform(shape=(),
                                                  minval=-time_shift_samples,
                                                  maxval=time_shift_samples,
                                                  dtype=tf.int32)
        else:
            time_shift_amount = 0

        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]

        padded_foreground = tf.pad(audio_tensor, time_shift_padding, mode="CONSTANT")
        sliced_foreground = tf.slice(padded_foreground, time_shift_offset, [desired_samples, -1])

        # Get a random section of background noise
        if use_background:
            background_index = tf.random.uniform(shape=(),
                                                 maxval=len(background_data),
                                                 dtype=tf.int32)
            print("Test->", background_index)
            print("Test-> ", tf.executing_eagerly())
            background_samples = background_data[background_index]
            background_offset = tf.random.uniform(shape=(),
                                                  maxval=len(background_samples) - desired_samples,
                                                  dtype=tf.int32)
            background_clips = background_samples[background_offset:(background_offset + desired_samples)]
            background_reshape = tf.reshape(background_clips, [desired_samples, 1])
            if tf.random.uniform(shape=(), maxval=1) < background_frequency:
                background_volume = tf.random.uniform(shape=(), maxval=background_volume_range)
            else:
                background_volume = tf.constant(0, dtype=tf.float32)
        else:
            background_reshape = np.zeros([desired_samples, 1], dtype=np.float32)
            background_volume = tf.constant(0, dtype="float32")

        # Mix in background noise
        background_mul = tf.multiply(background_reshape, background_volume)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        # final feature
        mfcc = calculate_mfcc(background_clamp,
                              sample_rate,
                              model_settings["window_size_samples"],
                              model_settings["window_stride_samples"],
                              model_settings["dct_coefficient_count"])
        mfcc = tf.reshape(mfcc, [-1])
        # print("print shape of mfcc feature: {}".format(mfcc.shape().as_list()))
        return mfcc, label
