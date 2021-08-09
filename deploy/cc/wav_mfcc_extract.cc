/*
 * @Author: your name
 * @Date: 2021-08-05 10:13:53
 * @LastEditTime: 2021-08-09 15:07:26
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\wav_mfcc_extract.cc
 */
#include "wav_mfcc_extract.h"

// Constructor
FeatureExtract::FeatureExtract() {}

// De-constructor
FeatureExtract::~FeatureExtract() {}

/**
 * @description: Initialize all module, Paramters,Spectrogram instance, Mfcc
 * instance
 * @param {const Params} &params: Paramters instances which contain all the
 * setting for model
 * @return Error code
 */
size_t FeatureExtract::Initialize(const Params &params) {

    size_t ret;

    // Set parameters
    this->params_ = params;

    // TODO, Check params was correct
    ret = CheckParams();
    if (ret != 0) {
        return ret;
    }

    // calcuate some paramters
    params_.paramters["window_size_samples"] =
        int(params_.paramters["sample_rate"] *
            params_.paramters["window_size_ms"] / 1000); // 480
    params_.paramters["window_stride_samples"] =
        int(params_.paramters["sample_rate"] *
            params_.paramters["window_stride_ms"] / 1000); // 160
    params_.paramters["desired_samples"] =
        int(params_.paramters["sample_rate"] *
            params_.paramters["clip_duration_ms"] / 1000); // 16000
    int length_minus_window = (params_.paramters["desired_samples"] -
                               params_.paramters["window_size_samples"]);
    params_.paramters["feature_length"] =
        params_.paramters["dct_coefficient_count"] *
        (1 + int(length_minus_window /
                 params_.paramters["window_stride_samples"])); // 40*(97+1)=3920

    // Print parameters for debug
    PrintParams();

    // Spectorgram instance initialize
    bool flag = sgram_.Initialize(params_.paramters["window_size_samples"],
                                  params_.paramters["window_stride_samples"]);
    if (!flag) {
        std::cout << "Spectrogram initialize failed!";
        // TODO , error code
        return 1;
    }

    // Mfcc initialize
    mfcc_.set_upper_frequency_limit(params_.paramters["upper_frequency_limit"]);
    mfcc_.set_lower_frequency_limit(params_.paramters["lower_frequency_limit"]);
    mfcc_.set_filterbank_channel_count(
        params_.paramters["filterbank_channel_count"]);
    mfcc_.set_dct_coefficient_count(params_.paramters["dct_coefficient_count"]);

    return 0;
}

/**
 * @description: Check parameters setting was write or not
 * @param {*}
 * @return {*}
 */
size_t FeatureExtract::CheckParams() {
    // Sampling rate
    if (params_.paramters["sample_rate"] != 16000) {
        std::cout << "Setting audio sampling rate was not 16000!\n";
        // TODO, error code
        return 1;
    }

    // window size
    if (params_.paramters["window_size_ms"] != 30) {
        std::cout << "window_size_ms was not <30ms> is different with it when "
                     "training model";
        return 1;
    }

    // window stride
    if (params_.paramters["window_stride_ms"] != 10) {
        std::cout << "window_stride_ms was not<10>ms is different with it when "
                     "training model!\n";
        return 1;
    }

    // audio clip
    if (params_.paramters["clip_duration_ms"] != 1000) {
        std::cout << "clip_duration_ms was not<1000>ms is different with it "
                     "when training model!\n";
        return 1;
    }

    // upper_frequency_limit
    if (params_.paramters["upper_frequency_limit"] != 4000) {
        std::cout << "upper_frequency_limit was not `4000`HZ is different with "
                     "it when training model!\n";
        return 1;
    }

    // lower_frequency_limit
    if (params_.paramters["lower_frequency_limit"] != 20) {
        std::cout << "lower_frequency_limit was not `20`HZ is different with "
                     "it when training model!\n";
        return 1;
    }

    // filterbank_channel_count
    if (params_.paramters["filterbank_channel_count"] != 40) {
        std::cout << "lower_frequency_limit was not `40` is different with it "
                     "when training model!\n";
        return 1;
    }

    // dct_coefficient_count
    if (params_.paramters["dct_coefficient_count"] != 40) {
        std::cout << "dct_coefficient_count was not `40` is different with it "
                     "when training model!\n";
        return 1;
    }

    // feature length
    if (params_.paramters["feature_length"] != 3920) {
        std::cout << "feature_length was not `3920` is different with it "
                     "when training model!\n";
        return 1;
    }

    return 0;
}

void FeatureExtract::PrintParams() {
    std::cout << "\n\tclip_duration_ms: "
              << params_.paramters["clip_duration_ms"]
              << "ms\n\tsample_rate:" << params_.paramters["sample_rate"]
              << "\n\tdesired_samples:" << params_.paramters["desired_samples"]
              << "\n\twindow_size_ms: " << params_.paramters["window_size_ms"]
              << "ms\n\twindow_size_samples"
              << params_.paramters["window_size_samples"]
              << "\n\twindow_stride_ms: "
              << params_.paramters["window_stride_ms"]
              << "ms\n\twindow_stride_samples: "
              << params_.paramters["window_stride_samples"]
              << "\n\tlower_frequency_limit: "
              << params_.paramters["lower_frequency_limit"]
              << "HZ\n\tupper_frequency_limit: "
              << params_.paramters["upper_frequency_limit"]
              << "HZ\n\tfilterbank_channel_count: "
              << params_.paramters["filterbank_channel_count"]
              << "\n\tdct_coefficient_count: "
              << params_.paramters["dct_coefficient_count"]
              << "\n\tfeature_length: " << params_.paramters["feature_length"]
              << std::endl;
}

/**
 * @description: Read audio data from wav file like tensorflow cc
 * @param filePath: wav file
 * @param data: audio sample data
 * @param decoded_sample_count: decoded how many samples in audio file
 * @param decoded_channel_count: real audio channel in audio file, we only
 * support *1-channel*
 * @param decoded_sample_rate: real sample rate in audio file, only support
 * *16khz*
 * @return {*}
 */
size_t FeatureExtract::ReadWav(const std::string &filePath,
                               std::vector<double> &data,
                               uint32_t &decoded_sample_count,
                               uint16_t &decoded_channel_count,
                               uint32_t &decoded_sample_rate) {
    std::ifstream inFile(filePath, std::ifstream::in | std::ifstream::binary);
    size_t ret = 0;

    // read wav header and check infos
    wavHeader hdr;
    int headerSize = sizeof(wavHeader);
    inFile.read((char *)&hdr, headerSize);

    // Check audio format
    if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
        std::cerr << "Unsupported audio format, use 16 bit PCM Wave"
                  << std::endl;
        // TODO, error code
        return 1;
    }
    // Check sampling rate, only support 16khz
    decoded_sample_rate = hdr.SamplesPerSec;
    if (hdr.SamplesPerSec != 16000) {
        std::cerr << "Sampling rate mismatch: Found " << hdr.SamplesPerSec
                  << " instead of " << 16000 << std::endl;
        // TODO, error code
        return 1;
    }

    // Check audio channel, only support 1-channel
    decoded_channel_count = hdr.NumOfChannels;
    if (hdr.NumOfChannels != 1) {
        std::cerr << hdr.NumOfChannels
                  << " channel files are unsupported. Use mono." << std::endl;
        // TODO, add error code
        return 1;
    }

    if (!inFile.is_open()) {
        std::cout << std::endl << "Can not open the WAV file !!" << std::endl;
        // TODO, add error code
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
    // Sample data convert to -1.0~1.0
    inFile.read((char *)buffer, bufferLength * bufferBPS);
    for (int i = 0; i < bufferLength; i++)
        float_values[i] = Int16SampleToFloat(buffer[i]);
    // data[i] = Int16SampleToFloat(buffer[i]);
    delete[] buffer;
    inFile.close();

    // Convert from float to double for the output value.
    // TODO, here do audio clip the same as in data_process
    // int clip_duration_ms = 1000; // only 1000ms
    // int desired_samples = int(decoded_sample_rate * clip_duration_ms / 1000);
    // data.resize(float_values.size());
    data.resize(params_.paramters["desired_samples"]);
    std::cout << "Choose process samples size was: "
              << params_.paramters["desired_samples"] << std::endl;
    for (int i = 0; i < params_.paramters["desired_samples"]; ++i) {
        if (i >= float_values.size()) {
            data[i] = 0.0; // padding for short audio
        } else {
            data[i] = float_values[i];
        }
    }

    return 0;
}

size_t FeatureExtract::AudioDataNorm(std::vector<int16_t> &audio_data,
                                     std::vector<double> &norm_samples) {

    // Convert data to -1.0~1.0
    norm_samples.resize(params_.paramters["desired_samples"]);
    int audio_data_size = audio_data.size();
    for (int i = 0; i < audio_data.size(); ++i) {
        if (i >= audio_data_size) {
            norm_samples[i] = 0.0; // padding for the specific sample length
        } else {
            norm_samples[i] = Int16SampleToFloat(audio_data[i]);
        }
    }

    return 0;
}

/**
 * @description: Input audio samples and calculate spectrogram
 * @param {vector<double>} audio_samples, input audio samples
 * @param {uint32_t} &sample_rate
 * @return {*}
 */
size_t FeatureExtract::GetSpectrogram(
    std::vector<double> audio_samples,
    std::vector<std::vector<double>> &spectrogram_output) {
    // Step1, convert audio data to spectrogram
    // std::vector<std::vector<double>> spectrogram_output;
    // *NOTE*, fft state
    sgram_.Reset();
    sgram_.ComputeSquaredMagnitudeSpectrogram(audio_samples,
                                              &spectrogram_output);
    std::cout << "spectrogram size: " << spectrogram_output.size()
              << "\tinternal vector size: " << spectrogram_output[0].size()
              << std::endl;
}

/**
 * @description: Calculate MFCC features based on spectrogram
 * @param spectrogram_output: input spectrogram vectors
 * @param mfcc_features: mfcc feature vectors
 * @return error code
 */
size_t FeatureExtract::SpectrogramToMfcc(
    std::vector<std::vector<double>> &spectrogram_output,
    std::vector<std::vector<double>> &mfcc_features) {
    int spectrogram_channels = spectrogram_output[0].size();
    mfcc_.Initialize(spectrogram_channels, params_.paramters["sample_rate"]);
    for (int i = 0; i < spectrogram_output.size(); ++i) {
        std::vector<double> mfcc_out;
        mfcc_.Compute(spectrogram_output[i], &mfcc_out);
        // assert(mfcc_out.size() == dct_coefficient_count_);
        mfcc_features.push_back(mfcc_out);
    }

    // print results
    std::cout << "mfcc out total frames: " << mfcc_features.size()
              << " frame dimension: " << mfcc_features[0].size() << std::endl;
    return 0;
}

// Extract features, calculate mfcc features
size_t FeatureExtract::ExtractFeatures(
    std::vector<double> audio_samples,
    std::vector<std::vector<double>> &mfcc_features) {
    size_t ret = 0;

    Clock::time_point TStart, TEnd;
    TStart = Clock::now();
    // Step1, calculate spectrogram
    std::vector<std::vector<double>> spectrogram_output;
    ret = GetSpectrogram(audio_samples, spectrogram_output);

    // Step2, calculate mfcc
    std::vector<std::vector<double>> mfcc_features;
    ret = SpectrogramToMfcc(spectrogram_output, mfcc_features);

    TEnd = Clock::now();
    Milliseconds ms = std::chrono::duration_cast<Milliseconds>(TEnd - TStart);
    std::cout << "Completed audio mfcc feature extraction cost time: "
              << ms.count() << "ms" << std::endl;
    return 0;
}

// Setting parameters
size_t FeatureExtract::SetParameters(const std::string &param_name,
                                     int &value) {
    // Check parameter name exist or not
    if (params_.paramters.find(param_name) == params_.paramters.end()) {
        std::cout << "[ERROR]: Parameter: " << param_name << "is not exist!\n";
        return 1;
    }

    // Set paramters
    params_.paramters[param_name] = value;
    return 0;
}

// Get parameters
size_t FeatureExtract::GetParameters(const std::string &param_name,
                                     int &value) {
    // Check parameter name exist or not
    if (params_.paramters.find(param_name) == params_.paramters.end()) {
        std::cout << "[ERROR]: Parameter: " << param_name << "is not exist!\n";
        return 1;
    }

    // Set paramters
    value = params_.paramters[param_name];
}

// Calculate mfcc features for a wav file, can be write features to file
size_t FeatureExtract::ProcessSingleWav(
    std::string filename, std::string outfile, bool write_to_file,
    std::vector<std::vector<double>> &mfcc_features) {
    std::cout << "Start extract audio features from file: " << filename
              << std::endl;
    Clock::time_point TStart, TEnd;
    TStart = Clock::now();
    std::vector<double>
        audio_samples; // to store audio sample data, norm to -1.0~1.0
    uint32_t decoded_sample_count;  // how many samples in wav file
    uint16_t decoded_channel_count; // how many channels in wav file
    uint32_t decoded_sample_rate;   // the real sample rate of wav file

    size_t ret = ReadWav(filename, audio_samples, decoded_sample_count,
                         decoded_channel_count, decoded_sample_rate);
    if (ret != 0) {
        std::cout << "Load audio data error!\n";
        return 1;
    }

    // Get Spectrogram and mfcc features
    std::vector<std::vector<double>> mfcc_features;
    ret = ExtractFeatures(audio_samples, mfcc_features);

    TEnd = Clock::now();
    Milliseconds ms = std::chrono::duration_cast<Milliseconds>(TEnd - TStart);
    std::cout << "Load and completed audio mfcc feature extraction cost time: "
              << ms.count() << "ms" << std::endl;

    // Save mfcc features to files
    std::ofstream outfs(outfile);
    if (!outfs.is_open()) {
        std::cout << "Open outfile " << outfile << "error!\n";
        return 1;
    }

    outfs << std::fixed << std::setprecision(8);
    for (int i = 0; i < mfcc_features.size(); ++i) {
        for (int j = 0; j < mfcc_features[i].size(); ++j) {
            outfs << mfcc_features[i][j] << " ";
        }
        outfs << std::endl;
    }
    outfs.close();
    return 0;
}

// Calculate mfcc features for given wav folder
size_t FeatureExtract::ProcessWavFileList(
    std::string wav_folder, std::string out_folder,
    std::vector<std::string> &filenames, bool write_to_file,
    std::vector<std::vector<std::vector<double>>> &mfcc_feature_list) {

    DIR *pDir;
    struct dirent *ptr;
    std::vector<std::string> files;
    std::vector<std::string> outfiles;

    // *NOTE*, wav file name format like 5338ca0367ec5ef0d43244cdae31dda7.wav_2
    if (!(pDir = opendir(wav_folder.c_str()))) {
        perror(("Folder " + wav_folder + "doesn't exist!").c_str());
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
            //     std::cout << "wav file name not contain label: " <<
            //     ptr->d_name
            //               << std::endl;
            // }

            // 5338ca0367ec5ef0d43244cdae31dda7.wav_2
            files.push_back(wav_folder + "/" + ptr->d_name);
            filenames.push_back(ptr->d_name);
            // mfcc.5338ca0367ec5ef0d43244cdae31dda7.wav_2
            outfiles.push_back(out_folder + "/" + "mfcc." + ptr->d_name);
        }
    }
    closedir(pDir);

    std::vector<std::vector<double>> mfcc_features;
    for (int i = 0; i < files.size(); ++i) {
        ProcessSingleWav(files[i], outfiles[i], true, mfcc_features);
        mfcc_feature_list.push_back(mfcc_features);
    }

    return 0;
}

// Mfcc feature flat
void FeatureExtract::MfccFlat(std::vector<std::vector<double>> &mfcc_feature,
                              std::vector<float> &feature) {
    int feat_size = mfcc_feature.size() * mfcc_feature[0].size();
    feature.resize(feat_size);
    for (int i = 0; i < mfcc_feature.size(); ++i) {
        for (int j = 0; j < mfcc_feature[i].size(); ++j) {
            int index = i * mfcc_feature[i].size() + j;
            feature[index] = mfcc_feature[i][j];
        }
    }
}