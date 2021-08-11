/*
 * @Author: your name
 * @Date: 2021-08-02 17:58:26
 * @LastEditTime: 2021-08-05 13:50:22
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\tfversion\demo.cc
 */
#include "stdafx.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <dirent.h>
#include <cstring>

//#include "wav_header.h"
#include "mfcc.h"
#include "spectrogram.h"

/* Steps to calculate MFCC
Step1, load wav file prepare audio data
Step2, Spectrogram sgram;   sgram.Initialize(int window_length, int
step_length);  in spectrogram.cc window_length=window_size=480,
step_length=stride=160(tf.audio_spectrogram); use
ComputeSquaredMagnitudeSpectrogram(input, output), get the final spectrogram
results.
Step3, then use mfcc to compute mfcc features in mfcc.cc
      mfcc.Initialize(int input_length, double input_sample_rate),
input_length=input.size(),

*/
static const int16_t kint16min = static_cast<int16_t>(~0x7FFF);
static const int16_t kint16max = static_cast<int16_t>(0x7FFF);

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;

inline float Int16SampleToFloat(int16_t data) {
    constexpr float kMultiplier = 1.0f / (1 << 15);
    return data * kMultiplier;
}

//inline int16_t FloatToInt16Sample(float data) {
//    constexpr float kMultiplier = 1.0f * (1 << 15);
//    return std::min<float>(
//        std::max<float>(roundf(data * kMultiplier), kint16min), kint16max);
//}

struct WAVHeader {
	/* RIFF Chunk Descriptor */
	uint8_t         RIFF[4];        // RIFF Header Magic header,4字节大端序。文件从此处开始，对于WAV或AVI文件，其值总为“RIFF”
	uint32_t        ChunkSize;      // RIFF Chunk Size,4字节小端序。表示文件总字节数减8，减去的8字节表示,ChunkID与ChunkSize本身所占字节数
	uint8_t         WAVE[4];        // WAVE Header,4字节大端序。对于WAV文件，其值总为“WAVE”
									/* "fmt" sub-chunk */
	uint8_t         fmt[4];         // FMT header, 4字节大端序。其值总为“fmt ”，表示Format Chunk从此处开始
	uint32_t        Subchunk1Size;  // Size of the fmt chunk,4字节小端序。表示Format Chunk的总字节数减8
	uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM,2字节小端序
	uint16_t        NumOfChannels;      // Number of channels 1=Mono 2=Stereo,2字节小端序
	uint32_t        SamplesPerSec;  // Sampling Frequency in Hz,4字节小端序,表示在每个通道上每秒包含多少帧
	uint32_t        bytesPerSec;    // bytes per second,4字节小端序。大小等于SampleRate * BlockAlign，表示每秒共包含多少字节
	uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo,2字节小端序。大小等于NumChannels * BitsPerSample / 8， 表示每帧的多通道总字节数
	uint16_t        bitsPerSample;  // Number of bits per sample,2字节小端序。表示每帧包含多少比特
									/* "data" sub-chunk */
	uint8_t         Subchunk2ID[4]; // "data"  string,4字节大端序。其值总为“data”，表示Data Chunk从此处开始
	uint32_t        Subchunk2Size;  // Sampled data length, 4字节小端序。表示data的总字节数


};

// Read audio data from wav file like tensorflow cc
size_t ReadWav(const std::string &filePath, std::vector<double> &data,
               uint32_t &decoded_sample_count, uint16_t &decoded_channel_count,
               uint32_t &decoded_sample_rate) {
    std::ifstream inFile(filePath, std::ifstream::in | std::ifstream::binary);
    size_t ret = 0;

    // read wav header and check infos
    WAVHeader hdr;
    int headerSize = sizeof(WAVHeader);
    inFile.read((char *)&hdr, headerSize);

    // Check audio format
    if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
        std::cerr << "Unsupported audio format, use 16 bit PCM Wave"
                  << std::endl;
        return 1;
    }
    // Check sampling rate
    decoded_sample_rate = hdr.SamplesPerSec;
    if (hdr.SamplesPerSec != 16000) {
        std::cerr << "Sampling rate mismatch: Found " << hdr.SamplesPerSec
                  << " instead of " << 16000 << std::endl;
        return 1;
    }

    // Check sampling rate:
    decoded_channel_count = hdr.NumOfChannels;
    if (hdr.NumOfChannels != 1) {
        std::cerr << hdr.NumOfChannels
                  << " channel files are unsupported. Use mono." << std::endl;
        return 1;
    }

    if (!inFile.is_open()) {
        std::cout << std::endl << "Can not open the WAV file !!" << std::endl;
        return 1;
    }

    // read real audio data
    // calculate how many samples
    uint32_t expected_bytes = (hdr.bitsPerSample * hdr.NumOfChannels + 7) / 8;
    std::cout << "chunk_size: " << hdr.ChunkSize
              << "\t bytes_per_seconds: " << hdr.bytesPerSec
              << "\texpected bytes: " << expected_bytes
              << "bits_per_samples: " << hdr.bitsPerSample << std::endl;
    decoded_sample_count = hdr.ChunkSize / expected_bytes;
    // calculate how many data in audio
    uint32_t data_count = decoded_sample_count * hdr.NumOfChannels;
    std::vector<float> float_values;
    float_values.resize(data_count);
    std::cout << "Total samples in wav:" << data_count << std::endl;

    uint16_t bufferLength = data_count;
    int16_t *buffer = new int16_t[bufferLength];
    int bufferBPS = (sizeof buffer[0]);

    // Read all data into float_values
    inFile.read((char *)buffer, bufferLength * bufferBPS);
    for (int i = 0; i < bufferLength; i++)
        float_values[i] = Int16SampleToFloat(buffer[i]);
    // data[i] = Int16SampleToFloat(buffer[i]);
    delete[] buffer;

    //--------------------------------------
    inFile.close();

    // Convert from float to double for the output value.
    // TODO, here do audio clip the same as in data_process
    int clip_duration_ms = 1000; // only 1000ms
    int desired_samples = int(decoded_sample_rate * clip_duration_ms / 1000);
    // data.resize(float_values.size());
    data.resize(desired_samples);
    std::cout << "Choose process samples size was: " << desired_samples
              << std::endl;
    for (int i = 0; i < desired_samples; ++i) {
        if(i >= float_values.size()) {
            data[i] = 0.0;  // padding
        } else {
            data[i] = float_values[i];
        }
    }

    return 0;
}

// Convert vector of double to string (for writing MFCC file output)
std::string vector_to_string(std::vector<double> vec,
                             const std::string &delimiter) {
    std::stringstream vecStream;
    for (int i = 0; i < vec.size() - 1; i++) {
        vecStream << vec[i];
        vecStream << delimiter;
    }
    vecStream << vec.back();
    vecStream << "\n";
    return vecStream.str();
}

std::string vector_vector_string(std::vector<std::vector<double>> vec,
                                 const std::string &delimiter) {
    std::string s1 = "";
    std::stringstream vec_stream;
    for (int i = 0; i < vec.size() - 1; ++i) {
        s1 = vector_to_string(vec[i], delimiter);
        vec_stream << s1;
        vec_stream << "\n";
    }
    s1 = vector_to_string(vec.back(), delimiter);
    vec_stream << s1;
    // vec_stream << "\n";
    return vec_stream.str();
}

int ExtractMfccFeature(Mfcc &mfcc, Spectrogram &sgram,
                       std::vector<double> audio_samples,
                       uint32_t &sample_rate,
                       std::vector<std::vector<double>> &mfcc_features) {
    // Step1, convert audio data to spectrogram
    std::vector<std::vector<double>> spectrogram_output; 
    // *NOTE*, fft state
    sgram.Reset();   
    sgram.ComputeSquaredMagnitudeSpectrogram(audio_samples, &spectrogram_output);
    std::cout << "spectrogram size: " << spectrogram_output.size()
              << "\tinternal vector size: " << spectrogram_output[0].size()
              << std::endl;

    // std::string res2 = vector_vector_string(spectrogram_output, " ");
    // std::cout << "Specrogram results: \n" << res2 << std::endl;

    // Step2, calculate mfcc features with spectrogram
    // std::vector<std::vector<double>> output_data;
    // *NOTE*, here we only support 1-channel audio data
    int spectrogram_channels = spectrogram_output[0].size();
    mfcc.Initialize(spectrogram_channels, sample_rate);
    for (int i = 0; i < spectrogram_output.size(); ++i) {
        std::vector<double> mfcc_out;
        mfcc.Compute(spectrogram_output[i], &mfcc_out);
        // assert(mfcc_out.size() == dct_coefficient_count_);
        mfcc_features.push_back(mfcc_out);
    }

    // print results
    std::cout << "mfcc out total frames: " << mfcc_features.size()
              << " frame dimension: " << mfcc_features[0].size() << std::endl;

    // std::string mfcc_str = vector_vector_string(output_data, " ");
    // std::cout << "Mfcc feature results: \n" << mfcc_str << std::endl;

    return 0;   // success
}

/**
 * @description: String split with specific pattern
 * @param str: input string to be split
 * @param vec: split results
 * @param pattern: split delimiter
 * @return 
 */
void SplitWord(const std::string &str, std::vector<std::string>& vec, const std::string& pattern) {
    std::string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = str.find(pattern);
	while (std::string::npos != pos2) {
		vec.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + pattern.size();
		pos2 = str.find(pattern, pos1);
	}
	if (pos1 != str.length()) {
		vec.push_back(str.substr(pos1));
	}
}

int processSingleFile(Mfcc &mfcc, Spectrogram &sgram,std::string filename, std::string outfile) {
    std::cout << "Start extract audio features from file: " << filename
              << std::endl;
    Clock::time_point TStart, TEnd;
    TStart = Clock::now();
    std::vector<double> audio_samples;  // to store audio sample data, norm to -1.0~1.0
    uint32_t decoded_sample_count;  // how many samples in wav file
    uint16_t decoded_channel_count; // how many channels in wav file
    uint32_t decoded_sample_rate;   // the real sample rate of wav file

    size_t ret = ReadWav(filename, audio_samples, decoded_sample_count, decoded_channel_count,
            decoded_sample_rate);
    if (ret != 0) {
        std::cout << "Load audio data error!\n";
        return 1;
    }

    // Get Spectrogram and mfcc features
    std::vector<std::vector<double>> mfcc_features;
    ret = ExtractMfccFeature(mfcc, sgram,audio_samples, decoded_sample_rate,mfcc_features);

    TEnd = Clock::now();
    Milliseconds ms = std::chrono::duration_cast<Milliseconds>(TEnd - TStart);
    std::cout << "Completed audio mfcc feature extraction cost time: "
              << ms.count() << "ms" << std::endl;

    // Save mfcc features to files
    std::ofstream outfs(outfile);
    if (!outfs.is_open()) {
        std::cout << "Open outfile " << outfile << "error!\n";
        return 1;
    }

    outfs << std::fixed << std::setprecision(8);
    for(int i=0; i < mfcc_features.size(); ++i) {
        for(int j = 0; j < mfcc_features[i].size(); ++j) {
            outfs << mfcc_features[i][j] << " ";
        }
        outfs << std::endl;
    }
    outfs.close();
    return 0;
}

int processFileList(Mfcc &mfcc, Spectrogram &sgram, std::string wavFolder, std::string outfolder) {
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
            // extract label
            // std::vector<std::string> vec;
            // std::string delimiter = "_";
            // SplitWord(ptr->d_name, vec, delimiter);
            // if (vec.size() != 2) {
            //     std::cout << "wav file name not contain label: " << ptr->d_name
            //               << std::endl;
            // }

            // 5338ca0367ec5ef0d43244cdae31dda7.wav_2
            files.push_back(wavFolder + "/" + ptr->d_name);
            //mfcc.5338ca0367ec5ef0d43244cdae31dda7.wav_2
            outfiles.push_back(outfolder + "/" + "mfcc." + ptr->d_name);
        }
    }
    closedir(pDir);

    for (int i = 0; i < files.size(); ++i) {
        processSingleFile(mfcc, sgram, files[i], outfiles[i]);
    }

    return 0;
}


int main() {

    // ----------------Parameters for Spectrogram and MFCC-----------------//
    int sample_rate = 16000;    // sample rate,default=16000, only support 16khz
    int window_size_ms = 30;    // default 30ms, keep the same as model train
    int window_size_samples = int(sample_rate * window_size_ms / 1000); // 480
    int window_stride_ms = 10;                                          // 10ms
    int window_stride_samples =
        int(sample_rate * window_stride_ms / 1000); // 160

    // Define Spetrogram instance 
    Spectrogram sgram;
    sgram.Initialize(window_size_samples, window_stride_samples);

    // Define Mfcc instance
    Mfcc mfcc;
    // set parameters for mfcc
    // Defaults to `20`.The lowest frequency to use when calculating the
    // ceptstrum.
    double lower_frequency_limit_ = 20; // 20hz
    // 4000hz, Defaults to `4000` The highest frequency to use when calculating
    // the ceptstrum
    double upper_frequency_limit_ = 4000;
    // Defaults to `40`.Resolution of the Mel bank used internally.
    int filterbank_channel_count_ = 40;
    // Defaults to `13`.How many output channels to produce per time slice.
    int dct_coefficient_count_ = 40;
    mfcc.set_upper_frequency_limit(upper_frequency_limit_);
    mfcc.set_lower_frequency_limit(lower_frequency_limit_);
    mfcc.set_filterbank_channel_count(filterbank_channel_count_);
    mfcc.set_dct_coefficient_count(dct_coefficient_count_);
    // TODO, this was calculate by sgram.ComputeSquaredMagnitudeSpectrogram
    const int spectrogram_channels = 257; 
    
    // const int spectrogram_samples = spectrogram_output.size();
    // mfcc.Initialize(spectrogram_channels, sample_rate);


    std::string wav_file = "./47850987c8a227b92673f9a88fb7efb8.wav";
    std::string mfcc_out_dir = "./feat.mfcc";

    processSingleFile(mfcc, sgram,wav_file, mfcc_out_dir);

    // process wav list
    std::string wav_dir = "../audios/";
    std::string feat_dir = "../features/";
    processFileList(mfcc, sgram, wav_dir, feat_dir);

    // std::vector<double> vec;
    // uint32_t decoded_sample_count;
    // uint16_t decoded_channel_count;
    // uint32_t decoded_sample_rate;
    // ReadWav(wav_file, vec, decoded_sample_count, decoded_channel_count,
    //         decoded_sample_rate);
    // std::string delimiter = " ";
    // std::string res = vector_to_string(vec, delimiter);
    // std::cout << "Audio data samples size: " << vec.size() << std::endl;
    // std::cout << res << std::endl;

    // Step2, auido data convert to spectrogram



    // std::vector<std::vector<double>> spectrogram_output;
    // sgram.ComputeSquaredMagnitudeSpectrogram(vec, &spectrogram_output);
    // std::cout << "spectrogram size: " << spectrogram_output.size()
    //           << "\tinternal vector size: " << spectrogram_output[0].size()
    //           << std::endl;



    // Step3, create mfcc features, we implementation as the tensorflow in
    // mfcc_op.cc get spectrogram_channels, the dimension of spectrogram
    // spectrogram_samples, how many spectrograms
    // audio_channels, audio channels, default=1

    // const int spectrogram_channels = spectrogram_output[0].size();
    // const int spectrogram_samples = spectrogram_output.size();
    // const int audio_channels = 1;
    // std::cout << "spectrogram_channels: " << spectrogram_channels
    //           << "\tspectrogram_samples: " << spectrogram_samples
    //           << "\taudio_channels: " << audio_channels << std::endl;
    // Mfcc mfcc;
    // // set parameters for mfcc
    // // Defaults to `20`.The lowest frequency to use when calculating the
    // // ceptstrum.
    // double lower_frequency_limit_ = 20; // 20hz
    // // 4000hz, Defaults to `4000` The highest frequency to use when calculating
    // // the ceptstrum
    // double upper_frequency_limit_ = 4000;
    // // Defaults to `40`.Resolution of the Mel bank used internally.
    // int filterbank_channel_count_ = 40;
    // // Defaults to `13`.How many output channels to produce per time slice.
    // int dct_coefficient_count_ = 40;
    // mfcc.set_upper_frequency_limit(upper_frequency_limit_);
    // mfcc.set_lower_frequency_limit(lower_frequency_limit_);
    // mfcc.set_filterbank_channel_count(filterbank_channel_count_);
    // mfcc.set_dct_coefficient_count(dct_coefficient_count_);
    // mfcc.Initialize(spectrogram_channels, sample_rate);

    // // define output for mfcc
    // std::vector<std::vector<double>> output_data;
    // // *NOTE*, here we only support 1-channel audio data
    // for (int i = 0; i < spectrogram_output.size(); ++i) {
    //     std::vector<double> mfcc_out;
    //     mfcc.Compute(spectrogram_output[i], &mfcc_out);
    //     assert(mfcc_out.size() == dct_coefficient_count_);
    //     output_data.push_back(mfcc_out);
    // }




    // std::string wav_file = "/data1/yw.shi/data/audio/xiaoshun/full_data/audios/"
    //                        "344c2757c72360345bebcb71ce5c76d6.wav";
    // std::string mfcc_out_dir = "./output.mfcc";
    // std::string wav_dir = "./audios/";
    // std::string feat_dir = "./features/";

    // // processSingleFile(mfcc, wav_file, mfcc_out_dir);

    // processFileList(mfcc, wav_dir, feat_dir);
    return 0;
}