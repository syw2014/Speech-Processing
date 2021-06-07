#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : prepare_data.py
# PythonVersion: python3.6
# Date    : 2021/6/1 10:52
# Software: PyCharm
"""Use tf2.x(>=2.3) and tf.keras API to process data"""
import tensorflow as tf
import os
import numpy as np
import pathlib
from tensorflow.python.platform import gfile
import json
import math
from tensorflow.python.ops import gen_audio_ops as audio_ops

def which_set(filename):
    print(filename)
    result = ""
    if filename.find("train") != -1:
        result = "train"
    elif filename.find("dev") != -1:
        result = "dev"
    elif filename.find("test") != -1:
        result = "test"
    positive = True if filename.startswith("p") else False
    return result, positive


def decode_audio(wav_file):
    return tf.audio.decode_wav(
                wav_file, desired_channels=1, desired_samples=-1)


class DataProcess(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_path = data_dir + "/label"
        self.silence_label = "_silence_"
        self.silence_index = 0
        self.unknown_word_label = "_unknown_"
        self.unknown_word_index = 1
        self.noise_dir = data_dir + "/noise"
        self.label_words_list = []
        self.label_dict = {"_silence_": 0, "_unknown_": 1}
        self.silence_percentage = 10.0

        self.sample_rate = 16000
        self.clip_duration_ms = 1000    # ms
        self.window_size_ms = 30.0  # ms
        self.window_stride_ms = 10.0    # ms
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)
        self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
        self.window_stride_samples = int(self.sample_rate * self.window_stride_ms / 1000)
        self.background_frequency = 0.8
        self.background_volume_range = 0.1

        # prepare dataset for input dataset
        self.train_ds_index = []
        self.dev_ds_index = []
        self.test_ds_index = []
        self.noise_ds_index = []

    def prepare_label_words(self, label_words):
        self.label_words_list = [self.silence_label, self.unknown_word_label] + label_words
        for idx, word in enumerate(label_words):
            self.label_dict[word] = idx + 2
        print("Completed prepare word list and word dict, total words: {}".format(len(self.label_dict)))

    def get_ds_index(self):
        # prepare wav file path and it's label
        resources_file = os.path.join(self.data_dir + "/mobvoi_hotword_dataset_resources/", "*.json")
        for json_path in gfile.Glob(resources_file):
            _, wav_filename = os.path.split(json_path)
            # parse wav label
            with open(json_path) as f:
                jdata = json.load(f)
                for x in jdata:
                    if x["keyword_id"] == 0:
                        word = "hi,小问"
                    elif x["keyword_id"] == 1:
                        word = "你好问问"
                    elif x["keyword_id"] == -1:
                        word = self.unknown_word_label
                    else:
                        raise ValueError("Not support label word: ", x["keyword_id"])

                    wav_path = os.path.join(self.data_dir, "temp/mobvoi_hotword_dataset/" + x["utt_id"] + ".wav")
                # TODO, note here ,we add all negative samples into dataset for training/testing, also we can choose
                #  part of them.
                if wav_filename.find("train"):
                    self.train_ds_index.append({"label": self.label_dict[word], 'file': wav_path})
                elif wav_filename.find("dev"):
                    self.dev_ds_index.append({"label": self.label_dict[word], 'file': wav_path})
                elif wav_filename.find("test"):
                    self.test_ds_index.append({"label": self.label_dict[word], 'file': wav_path})
        print("Parse dataset completed train_size: {}, dev_size:{}, test_size:{}".format(
            len(self.train_ds_index), len(self.dev_ds_index), len(self.test_ds_index)))

        # load noise wav file
        self.noise_ds_index = os.path.join(self.data_dir+"/_background_noise_", "*.wav")
        print("Parse noise data completed total size: {}".format(len(self.noise_ds_index)))

    def preprocess(self):
        """Wav data preprocess add silence samples"""
        # make silence samples in dataset
        # train
        set_size = len(self.train_ds_index)
        silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
        # random choose from dataset that will be zero when training
        silence_wav_path = self.train_ds_index[0]["file"]
        for _ in range(silence_size):
            self.train_ds_index.append({"label": self.silence_label, "file": silence_wav_path})

        # dev
        set_size = len(self.dev_ds_index)
        silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
        # random choose from dataset that will be zero when training
        silence_wav_path = self.train_ds_index[0]["file"]
        for _ in range(silence_size):
            self.train_ds_index.append({"label": self.silence_label, "file": silence_wav_path})

        # test
        set_size = len(self.train_ds_index)
        silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
        # random choose from dataset that will be zero when training
        silence_wav_path = self.train_ds_index[0]["file"]
        for _ in range(silence_size):
            self.train_ds_index.append({"label": self.silence_label, "file": silence_wav_path})

    @tf.function
    def get_wavform_and_label(self, ds):
        """load wav file add some process"""
        audio_binary = tf.io.read_file(ds["file"])
        waveform = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=self.desired_samples)
        label_id = ds["label"]

        # step 1, Allow the audio sample's volume to be adjusted.
        if label_id == self.silence_index:
            foreground_volume = 0
        else:
            foreground_volume = 1
        scaled_foreground = tf.math.multiply(waveform, foreground_volume)

        # step 2, Shift the sample's start position, and pad any gaps with zeros
        padded_foreground = tf.pad(tensor=scaled_foreground,
                                   mode='CONSTANT',
                                   paddings=[[0, 0], [0, 0]])
        sliced_foreground = tf.slice(padded_foreground,
                                     [[0, 0], [0, 0]],
                                     [self.desired_samples, -1])

        # step 3, Mix in background noise.
        # Choose a section of background noise to mix in.
        background_index = np.random.randint(len(self.noise_ds_index))
        background_samples = self.noise_ds_index[background_index]
        if len(background_samples) <= self.desired_samples:
            raise ValueError(
                'Background sample is too short! Need more than %d'
                ' samples but only %d were found' %
                (self.desired_samples, len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - self.desired_samples)
        background_clipped = background_samples[background_offset:(
                background_offset + self.desired_samples)]
        background_reshaped = background_clipped.reshape([self.desired_samples, 1])
        if ds['label'] == self.silence_index:
            background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < self.background_frequency:
            background_volume = np.random.uniform(0, self.background_volume_range)
        else:
            background_volume = 0
        background_mul = tf.multiply(background_reshaped,
                                     background_volume)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = audio_ops.audio_spectrogram(
            background_clamp,
            window_size=self.window_size_samples,
            stride=self.window_stride_samples,
            magnitude_squared=True)

        output = audio_ops.mfcc(
            spectrogram,
            self.sample_rate,
            dct_coefficient_count=40)

        return output, ds["label"]

    def create_dataset(self):
        """Create dataset for model input"""

        train_ds = tf.data.Dataset.from_tensor_slices(self.train_ds_index)
        dev_ds = tf.data.Dataset.from_tensor_slices(self.dev_ds_index)
        test_ds = tf.data.Dataset.from_tensor_slices(self.test_ds_index)

        train_ds.shuffle(buffer_size=10000)
        train_out_ds = train_ds.map(self.get_wavform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        for vec, label in train_out_ds.take(1):
            label = label.numpy().decode("utf-8")
            print("label: ", label)
            print("vec shape: ", vec.shape)

        dev_ds.shuffle(buffer_size=10000)
        dev_output_ds = train_ds.map(self.get_wavform_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        test_ds.shuffle(buffer_size=10000)
        test_out_ds = train_ds.map(self.get_wavform_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        return train_out_ds, dev_output_ds, test_out_ds

