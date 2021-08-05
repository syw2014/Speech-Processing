/*
 * @Author: your name
 * @Date: 2021-08-05 10:12:17
 * @LastEditTime: 2021-08-05 16:14:42
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\wav_mfcc_extract.h
 */
// This was the class intergrate the wav MFCC features extraction
#ifndef WAV_MFCC_EXTRACT_H
#define WAV_MFCC_EXTRACT_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "mfcc.h"
#include "spectrogram.h"
#include "wavHeader.h"

static const int16_t kint16min = static_cast<int16_t>(~0x7FFF);
static const int16_t kint16max = static_cast<int16_t>(0x7FFF);

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;

inline float Int16SampleToFloat(int16_t data) {
    constexpr float kMultiplier = 1.0f / (1 << 15);
    return data * kMultiplier;
}

inline int16_t FloatToInt16Sample(float data) {
    constexpr float kMultiplier = 1.0f * (1 << 15);
    return std::min<float>(
        std::max<float>(roundf(data * kMultiplier), kint16min), kint16max);
}

// Define a struct to hold all parameters that used in features extraction
struct Params {
    //   public:
    int clip_duration_ms; // default 1000ms,
                          // how long time you want to process audio, this must
                          // be the same as it in training process
    int desired_samples;  // how many samples want to process
    int sample_rate;      // sample rate,default=16000, only support 16khz
    int window_size_ms;   // default 30ms, keep the same as model train
    int window_size_samples;   // 480
    int window_stride_ms;      // 10ms
    int window_stride_samples; // 160

    // set parameters for mfcc
    // Defaults to `20`.The lowest frequency to use when calculating the
    // ceptstrum.
    double lower_frequency_limit; // 20hz
    // 4000hz, Defaults to `4000` The highest frequency to use when calculating
    // the ceptstrum
    double upper_frequency_limit;
    // Defaults to `40`.Resolution of the Mel bank used internally.
    int filterbank_channel_count;
    // Defaults to `13`.How many output channels to produce per time slice.
    int dct_coefficient_count;
};

// Wav feature(mfcc) extraction class
class FeatureExtract {
  public:
    // Construct
    FeatureExtract();
    ~FeatureExtract();

    // Read audio data from wav file like tensorflow cc
    size_t ReadWav(const std::string &filePath, std::vector<double> &data,
                   uint32_t &decoded_sample_count,
                   uint16_t &decoded_channel_count,
                   uint32_t &decoded_sample_rate);

    // Convert audio samples to spectrogram
    size_t GetSpectrogram(std::vector<double> audio_samples,
                          std::vector<std::vector<double>> &spectrogram_output);

    size_t SpectrogramToMfcc(std::vector<std::vector<double>> &spectrogram_output,
                       std::vector<std::vector<double>> &mfcc_features);

    // Extract features, calculate mfcc features
    size_t ExtractFeatures(std::vector<double> audio_samples,
                           std::vector<std::vector<double>> &mfcc_features);
    // Setting parameters and initialize all instance
    size_t Initialize(const Params &params);

    // Setting parameters
    // size_t SetParameters(const Params &params);

    // Get parameters
    // Params GetParameters();

    // Calculate mfcc features for a wav file, can be write features to file
    size_t ProcessSingleWav(std::string filename, std::string outfile,
                            bool write_to_file,
                            std::vector<std::vector<double>> &mfcc_features);

    // Calculate mfcc features for given wav folder
    size_t ProcessWavFileList(
        std::string wav_folder, std::string out_folder, bool write_to_file,
        std::vector<std::vector<std::vector<double>>> &mfcc_feature_list);

    void PrintParams();

    size_t CheckParams();

  private:
    Params params_; // Paramters pointer

    Spectrogram sgram_; // Spectrogram instance
    Mfcc mfcc_;         // mfcc instance
};

#endif // wav_mfcc_extract.h