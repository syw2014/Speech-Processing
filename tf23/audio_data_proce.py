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
import glob
import shutil
import hashlib
import copy
import math

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_ids(input):
    """
    Input string and return md5
    Args:
        input: input string

    Returns:
        md5 string
    """
    return hashlib.md5(input)


def get_wav_name(wav_path):
    """
    Parse wav id and user id
    Args:
        filename: wav file name

    Returns:
        wav_id, speaker_id
    """
    filename = wav_path.rsplit("\\")[-1]
    id_list = filename.split(".")[0]
    wav_speaker = id_list.split("_")
    return wav_speaker[0], wav_speaker[1]


def which_set(wav_name, valid_percentage, test_percentage):
    """
    To decide which set the wav file will be train/valid/test/
    Args:
        wav_name: input wav file name(here was wav id)
        valid_percentage: how many portion of data treat as valid
        test_percentage: how many portion of data treat as test

    Returns:
        strings as `train`/`valid`/`test`

    """
    hash_name_hashed = hashlib.sha1(wav_name.encode("utf-8")).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < valid_percentage:
        result = 'valid'
    elif percentage_hash < (test_percentage + valid_percentage):
        result = 'test'
    else:
        result = 'train'
    return result


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

        self.all_wavs = []  # all wav info list
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

    def load_data(self):
        """Load data from original data directory."""
        if not os.path.exists(self.origin_dir):
            raise ValueError(f"Folder {self.origin_dir} not exists!")

        # loop folders
        listglobs = glob.glob(os.path.join(self.origin_dir) + r"[0-9]*")
        count = 0
        temp = []
        for x in listglobs:

            # step1, get speaker id md5
            user_id = x.rsplit("\\")[-1]
            speaker_id = hashlib.md5(user_id.encode("utf-8")).hexdigest()
            self.wav_desc["speaker_id"] = speaker_id
            print("1=>", x)

            for k in ["你好小顺", "小顺小顺"]:
                paths = os.path.join(x, k)
                print("2=>", paths)
                # step2, parse speaker info
                with open(os.path.join(paths, "spearker_info.txt"), 'r', encoding="utf-8") as f:
                    line = f.readline()
                    arrs = line.strip().split("\\t")
                    if len(arrs) != 3:
                        raise ValueError("Required three field in speaker_info<id>\t<gender>\t<age>")
                    self.wav_desc["gender"] = arrs[1].strip("<").rstrip(">")
                    self.wav_desc["age"] = arrs[-1].strip("<").rstrip(">")

                # step3, parse wav detailed information
                # key: wav_id, value: info_list, [keyword, noise_type, distance, speed,user_id, equipment]
                wav_infos_dict = {}
                with open(os.path.join(paths, "wav_desc.txt"), "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        arrs = line.strip().split("\\t")
                        wav_infos_dict[arrs[0].strip("<").rstrip(">")] = [x.strip("<").rstrip(">") for
                                                                          x in arrs[1:]]

                print(f"Parse wav info finished find {len(wav_infos_dict)} infos.")

                # Step4, audio with background noise and without nose, which was back_wav and wav_data folder
                for wav_folder in ["back_wav", "wav_data"]:
                    audio_lists = glob.glob(os.path.join(paths + f"\\{wav_folder}", "*.wav"))
                    for xa in audio_lists:
                        # copy data to
                        wav_id, user_id = get_wav_name(xa)
                        # print(wav_id, user_id)
                        # create md5 id
                        utt_id = hashlib.md5(xa.encode("utf-8")).hexdigest()
                        # speaker_id = hashlib.md5(user_id.encode("utf-8")).hexdigest()
                        # print(utt_id, speaker_id)
                        # collect all info for an audio
                        self.wav_desc["utt_id"] = utt_id
                        infos = wav_infos_dict[wav_id]
                        if len(infos) != 6:
                            print("==>", infos)
                        self.wav_desc["keyword_id"] = self.keywords_dict[infos[0]]
                        self.wav_desc["noise_type"] = infos[1]
                        self.wav_desc["distance"] = infos[2]
                        self.wav_desc["record_speed"] = infos[3]
                        self.wav_desc["speaker_id"] = speaker_id
                        self.wav_desc["record_equipment"] = infos[5]

                        # record wav information
                        t_infos = copy.deepcopy(self.wav_desc)
                        self.all_wavs.append(t_infos)
                        count += 1
                        temp.append(utt_id)

                        # copy data to resource folder
                        dest = shutil.copy2(xa, os.path.join(self.dest_dir, f"audios/{utt_id}.wav"))
                        set_index = which_set(dest, 20, 30)
                        self.data_index[set_index].append(t_infos)

        # write wav information into json file
        with open(os.path.join(self.dest_dir, "resources/wav_desc.json"), "w", encoding="utf-8") as f:
            json.dump(self.all_wavs, f, ensure_ascii=False, indent=True)
        print(f"total wavs:{count}, total ids:{len(temp)}")
        for set_index in self.data_index.keys():
            with open(os.path.join(self.dest_dir, f"resources/p_{set_index}.json"), "w", encoding="utf-8") as f:
                json.dump(self.data_index[set_index], f, ensure_ascii=False, indent=True)
                print(f"Collect {set_index} data total {len(self.data_index[set_index])} samples.")


if __name__ == '__main__':
    origin_dir = "E:/work/Project/Tasks/33 智能聊天助手/10.语音唤醒/3. 交付数据/6.15(10人)/"
    dest_dir = "E:/work/Project/Tasks/33 智能聊天助手/10.语音唤醒/3. 交付数据/datasets"
    processor = DataProcess(origin_dir, dest_dir)
    processor.load_data()
