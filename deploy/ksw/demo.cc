/*
 * @Author: your name
 * @Date: 2021-07-28 16:02:56
 * @LastEditTime: 2021-07-29 13:38:06
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\ksw\demo.cc
 */

#include "mfcc.h"
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

#define FREQ                                                                   \
    16000 // You can try 48000 to use 48000Hz wav files, but it's more slow.
#define TRAINSIZE                                                              \
    FREQ *                                                                     \
        1 // 4 secondes of voice for trainning
          // --- you can increase this value to improve the recognition rate
#define RECOGSIZE                                                              \
    FREQ *                                                                     \
        1 // 1 seconde of voice for recognition
          // --- you can increase this value to improve the recognition rate

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;

void CheckWavHeader(char *header) {
    int sr;
    if (header[20] != 0x1)
        std::cout << std::endl
                  << "Input audio file has compression [" << header[20]
                  << "] and not required PCM" << std::endl;

    sr = ((header[24] & 0xFF) | ((header[25] & 0xFF) << 8) |
          ((header[26] & 0xFF) << 16) | ((header[27] & 0xFF) << 24));
    std::cout << " " << (int)header[34] << " bits, " << (int)header[22]
              << " channels, " << sr << " Hz";
}

/**
 * @description: read audio data from file
 * @param {*}
 * @return {*}
 */
size_t ReadWav(const std::string &filePath, short int voiceData[],
               size_t sizeData, size_t seek) {
    std::ifstream inFile(filePath, std::ifstream::in | std::ifstream::binary);
    size_t ret;

    if (!inFile.is_open()) {
        std::cout << std::endl << "Can not open the WAV file !!" << std::endl;
        return -1;
    }

    // 校验wav文件格式
    char waveheader[44];
    inFile.read(waveheader, 44);
    if (seek == 0)
        CheckWavHeader(waveheader);

    if (seek != 0)
        inFile.seekg(seek * sizeof(short int), std::ifstream::cur);

    // 读取真是voice data
    inFile.read(reinterpret_cast<char *>(voiceData),
                sizeof(short int) * sizeData);
    // 计算audio samples个数
    ret = (size_t)inFile.gcount() / sizeof(short int);

    inFile.close();
    return ret;
}

int processSingleFile(MFCC &mfcc, std::string filename, std::string outfile) {
    std::cout << "Start extract audio features from file: " << filename
              << std::endl;
    Clock::time_point TStart, TEnd;
    short int bigVoiceBuffer[TRAINSIZE];
    TStart = Clock::now();
    size_t realSize = ReadWav(filename, bigVoiceBuffer, TRAINSIZE, 0);
    if (realSize < 1)
        return 1;

    //** Mfcc analyse WITH BIG BUFFER
    size_t frameCount = mfcc.FeatureExtarct(bigVoiceBuffer, realSize);
    std::vector<std::vector<double>> melCepData = mfcc.GetMFCCData();
    mfcc.Save(outfile);
    TEnd = Clock::now();
    Milliseconds ms = std::chrono::duration_cast<Milliseconds>(TEnd - TStart);
    std::cout << "Completed audio mfcc feature extraction cost time: "
              << ms.count() << "ms" << std::endl;

    return 0;
}

int processFileList(MFCC &mfcc, std::string wavFolder, std::string outfolder) {
    DIR *pDir;
    struct dirent *ptr;
    std::vector<std::string> files;
    std::vector<std::string> outfiles;

    if (!(pDir = opendir(wavFolder.c_str()))) {
        perror(("Folder " + wavFolder + "doesn't exist!").c_str());
        return 1;
    }
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            // std::cout << ptr->d_name << std::endl;
            files.push_back(wavFolder + "/" + ptr->d_name);
            outfiles.push_back(outfolder + "/" + ptr->d_name + ".mfcc");
        }
    }
    closedir(pDir);

    for (int i = 0; i < files.size(); ++i) {
        processSingleFile(mfcc, files[i], outfiles[i]);
    }

    return 0;
}

int main() {

    int sample_rate = 16000;
    int bit_size = 16;
    int window_stride_ms = 10; // 10ms
    int window_stride_samples = int(sample_rate * window_stride_ms / 1000);
    int filter_num = 40;
    int mfcc_dim = 40; 

    MFCC mfcc(sample_rate, bit_size, window_stride_ms, MFCC::Hamming, filter_num, mfcc_dim);

    std::string wav_file = "/data1/yw.shi/project/1.kws/mfcc/kws";
    std::string mfcc_out_dir = "./output.mfcc";
    std::string wav_dir = "./audios/";
    std::string feat_dir = "./features/";
    
    processSingleFile(mfcc, wav_file, mfcc_out_dir);

    // processFileList(mfcc, wav_dir, mfcc_out_dir);



    return 0;
}
