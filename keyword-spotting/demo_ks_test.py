#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : demo_ks_test.py
# PythonVersion: python3.6
# Date    : 2021/4/8 17:00
# Software: PyCharm
"""A demo detect audio and send to keyword spotting service on windows"""

from pyaudio import PyAudio, paInt16
import numpy as np
from datetime import datetime
import wave


class Recoder:
    chunks = 2000      # pyaudio 内置缓冲器大小
    sampling_rate = 16000    # 取样频率
    level = 500             # 声音保持的阈值
    count_num = 20          # num_samples个取样之内出现count_num个大于level的取样则记录声音
    save_length = 8         # 声音记录的最小长度,save_length * num_samples 个取样
    time_count = 60          # 录音时间，单位s

    voice_string = []

    def save_wav(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sampling_rate)
        wf.writeframes(np.array(self.voice_string).tostring())
        wf.close()

    def recoder(self):
        pa = PyAudio()
        stream = pa.open(format=paInt16,
                         channels=1,
                         rate=self.sampling_rate,
                         input=True,
                         frames_per_buffer=self.chunks)
        save_count = 0
        save_buffer = []
        time_count = self.time_count

        while True:
            time_count -= 1
            # read sample
            string_audio_data = stream.read(self.chunks)
            # convert data to array
            audio_data = np.fromstring(string_audio_data, dtype=np.short)
            # 计算大于level的取样个数
            large_sample_count = np.sum(audio_data > self.level)
            print(np.max(audio_data))
            # 如果个数大于count_num，则至少保持save_length个块
            if large_sample_count > self.count_num:
                save_count = self.save_length
            else:
                save_count -= 1
            if save_count < 0:
                save_count = 0

            if save_count > 0:
                save_buffer.append(string_audio_data)
            else:
                # save data into buffer
                if len(save_buffer) > 0:
                    self.voice_string = save_buffer
                    save_buffer = []
                    print("Recode a piece of voice successfully!")
                    return True
            if time_count == 0:
                if len(save_buffer) > 0:
                    self.voice_string = save_buffer
                    save_buffer = []
                    print("Recode a piece of voice successfully!")
                    return True
                else:
                    return False


if __name__ == '__main__':
    r = Recoder()
    r.recoder()
    wav_file = "../data/audio1.wav"
    r.save_wav(wav_file)