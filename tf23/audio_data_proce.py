#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : audio_data_proce.py
# PythonVersion: python3.6
# Date    : 2021/6/16 13:54
# Software: PyCharm
"""Create tools to process audio data from Datatang."""
import os
import json
from path import Path
import glob
import shutil
import hashlib
import copy


def create_ids(input):
    """
    Input string and return md5
    Args:
        input: input string

    Returns:
        md5 string
    """
    return hashlib.md5(input)

class DataProcess(object):
    def __init__(self, origin_dir, dest_dir, val_percentage=0.2, test_percentage=0.3):
        """
        Object to process labelling data into train/dev/test datasets
        Args:
            origin_dir: Origin data directory where audio was
            dest_dir:Destination data directory where to store data
            val_percentage: How many data will be treated as validation, default was 0.2
            test_percentage: How many data will be treated as test, default was 0.3
        """
        self.origin_dir = origin_dir
        self.dest_dir = dest_dir
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

        self.all_wavs = []    # all wav info list
        self.data_index = {"train": [], "valid": [], "test": []}

        # Detail information for an audio
        # utt_id: audio hash id, noise_volume: , age: the age of speaker,
        # keyword_id: keyword int id, 你好小顺(0), 小顺小顺(1)
        # noise_type: 电视剧/动漫/游戏/音乐/直播/说话声/无噪声
        # speaker_id: speaker id
        # record_speed: fast,normal, slow
        # record_equipment: record equipment
        # gender: gender of speaker
        self.wav_desc = {
            "utt_id": "",
            "noise_volume": "00db",
            "age": "00",
            "keyword_id": 0,
            "noise_type": "TV",
            "speaker_id": "",
            "distance": "",
            "record_speed": "",
            "record_equipment": "",
            "gender": ""}

        self.keywords_dict = {"你好小顺": 0, "小顺小顺": 1}

        if not os.path.exists(self.dest_dir):
            os.mkdir(os.path.join(self.dest_dir))
            os.mkdir(os.path.join(self.dest_dir, "resources"))
            os.mkdir(os.path.join(self.dest_dir, "audios"))

    def get_wav_name(self, wav_path):
        """
        Parse wav id and user id
        Args:
            filename: wav file name

        Returns:
            wav_id, speaker_id
        """
        arrs = wav_path.rsplit("\\")
        filename = arrs[-1]
        id_list = filename.split(".")[0]
        print("id_list", id_list)
        wav_speaker = id_list.split("_")
        return wav_speaker[0], wav_speaker[1]

    def load_data(self):
        """Load data from original data directory."""
        if not os.path.exists(self.origin_dir):
            raise ValueError(f"Folder {self.origin_dir} not exists!")

        # loop folders
        listglobs = glob.glob(os.path.join(self.origin_dir)+r"[1-9]*")
        for x in listglobs:

            # step1, get speaker id md5
            user_id = x.rsplit("\\")[-1]
            speaker_id = hashlib.md5(user_id.encode("utf-8")).hexdigest()
            self.wav_desc["speaker_id"] = speaker_id

            for k in ["你好小顺", "小顺小顺"]:
                paths = os.path.join(x, k)

                # step2, parse speaker info
                with open(os.path.join(paths, "spearker_info.txt"), 'r', encoding="utf-8") as f:
                    line = f.readline()
                    arrs = line.split("\\t")
                    if len(arrs) != 3:
                        raise ValueError("Required three field in speaker_info<id>\t<gender>\t<age>")
                    self.wav_desc["gender"] = arrs[1].strip("<").rstrip(">")
                    self.wav_desc["age"] = arrs[-1].strip("<").rstrip(">")

                # step3, parse wav detailed information
                # key: wav_id, value: info_list, [keyword, noise_type, distance, speed,user_id, equipment]
                wav_infos_dict = {}
                with open(os.path.join(paths, "wav_desc.txt"), "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        arrs = line.split("\\t")
                        wav_infos_dict[arrs[0].strip("<").rstrip(">")] = [x.strip("<").rstrip(">") for
                                                                          x in arrs[1:]]
                print(wav_infos_dict)
                print(f"Parse wav info finished find {len(wav_infos_dict)} infos.")

                # Step4, audio with background noise
                audio_lists = glob.glob(os.path.join(paths+"\\back_wav", "*.wav"))
                for xa in audio_lists:
                    # copy data to
                    wav_id, user_id = self.get_wav_name(xa)
                    # print(wav_id, user_id)
                    # create md5 id
                    utt_id = hashlib.md5(wav_id.encode("utf-8")).hexdigest()
                    # speaker_id = hashlib.md5(user_id.encode("utf-8")).hexdigest()
                    # print(utt_id, speaker_id)
                    # collect all info for an audio
                    self.wav_desc["utt_id"] = utt_id
                    infos = wav_infos_dict[wav_id]
                    self.wav_desc["keyword_id"] = self.keywords_dict[infos[0]]
                    self.wav_desc["noise_type"] = infos[1]
                    self.wav_desc["distance"] = infos[2]
                    self.wav_desc["record_speed"] = infos[3]
                    self.wav_desc["speaker_id"] = infos[4]
                    self.wav_desc["record_equipment"] = infos[5]

                    # record
                    print(self.wav_desc)
                    self.all_wavs.append(copy.deepcopy(self.wav_desc))



if __name__ == '__main__':
    origin_dir = "E:/work/Project/Tasks/33 智能聊天助手/10.语音唤醒/3. 交付数据/6.15(10人)/"
    dest_dir = "E:/work/Project/Tasks/33 智能聊天助手/10.语音唤醒/3. 交付数据/datasets"
    processor = DataProcess(origin_dir, dest_dir)
    processor.load_data()