#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : audio_trim.py
# PythonVersion: python3.6
# Date    : 2021/7/20 11:45
# Software: PyCharm
"""
Trim audio with AudioSegment API in pydub package, here we only keep the keyword audio which time duration was 1000
milliseconds.
"""
import json
from pydub import AudioSegment
import os
from tqdm import tqdm


def trim(wav_path, start_time, end_time):
    """
    Segment audio, only keep the time duration start and end.
    Here we only support wav format audio,because other format like mp3/mp4 may need ffmpeg
    Args:
        wav_path: input audio
        start_time: start time, milliseconds
        end_time: end time , milliseconds

    Returns:
        audio[start_time : end time]

    """
    if not wav_path.endswith('.wav'):
        raise ValueError("Only support wav format audio but found: {}".format(wav_path))
    audio = AudioSegment.from_wav(wav_path)

    duration = len(audio)
    if start_time > duration:
        raise ValueError("Start time {} was bigger than raw audio {} ".format(start_time, duration))

    trim_audio = audio[start_time: end_time]
    return trim_audio


if __name__ == '__main__':
    json_file = "wav_desc.json"
    orig_dir = "./no_trim/"
    dest_dir = "./audios/"
    start_time = 999   # 1000 milliseconds
    end_time = 2010     # 2000 milliseconds
    wav_desc = None
    with open(json_file, "r", encoding="utf-8") as f:
        wav_desc = json.load(f)

    print("Found {} wav data".format(len(wav_desc)))
    for info in tqdm(wav_desc):
        wav_file = orig_dir + info["utt_id"] + ".wav"
        trim_file = dest_dir + info["utt_id"] + ".wav"
        trim_audio = trim(wav_file, start_time, end_time)
        trim_audio.export(trim_file, format="wav")