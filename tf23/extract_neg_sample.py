#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : extract_neg_sample.py
# PythonVersion: python3.6
# Date    : 2021/6/28 11:39
# Software: PyCharm
import json
import os
import shutil


if __name__ == "__main__":
    data_dir = ""
    dest_dir = ""

    for file_name in ["n_train.json", "n_dev.json", "n_test.json"]:
        with open(file_name) as f:
            json_data = json.load(f)
            utt_id = json_data["utt_id"]
            wav_path = data_dir + "/" + utt_id + ".wav"
            shutil.copy2(wav_path, os.path.join(dest_dir, f"audios/{utt_id}.wav"))