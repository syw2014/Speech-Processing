/*
 * @Author: your name
 * @Date: 2021-08-05 16:29:27
 * @LastEditTime: 2021-08-09 14:29:25
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\kws.h
 */
// Keyword Spotting model
#ifndef KWS_H
#define KWS_H

#include <iostream>
#include <unordered_map>

#include "c_api.h"
#include "wav_mfcc_extract.h"

// Type to make wav file name and keyword as a pair
typedef std::pair<std::string, std::string> NameKeywordT;
// Type to make score and whether wake or not as a pair
typedef std::pair<float, bool> ScoreWakeT;
// final result of wav file
typedef std::pair<NameKeywordT, ScoreWakeT> KWSResultT;


class KWS {
  public:
    KWS();
    ~KWS();

    // Initialize all state
    size_t ModelInitialize(std::string &model_path, Params &params, float threshold=0.9);

    // Get output tensor from Tflite model
    size_t GetOutputTensorByName(const char *name);

    // Get input tensor from Tflite model
    size_t GetInputTensorByName(const char *name);

    // Model inference
    size_t Inference(const std::vector<float> &features,
                     const int &feature_length, std::vector<std::pair<int, float>> &logits);

    // Extract audio features
    size_t GetFeatures(std::vector<int16_t>& audio_data, std::vector<float>& features);

    // Get keywords
    size_t ParseLogits(std::vector<std::pair<int, float>>& logits, std::string &keyword, int &label_id, float &score);

    // Check result is keyword
    bool IsAwakenedWithAudio(std::vector<int16_t>& audio_samples, std::string &keyword, float &score, float threshold=0.85);

    // Check result is keyword with another
  bool IsAwakenedWithFeature(std::vector<float>& features, std::string &keyword, float &score, float threshold=0.85);

    // Process single wav file
    size_t ProcessWavFile(std::string& wav_file, std::string &keyword, float &score, bool is_wake);

    // Process wav file list
    size_t ProcessWavFileList(std::string& wav_dir, std::vector<std::vector<std::string>>& results);

  private:
    FeatureExtract feature_extractor_;       // mfcc feature extractor
    TfLiteInterpreter *interpreter_; // TFLite interpretere
    TfLiteTensor *input_tensor_;     // model input tensor
    TfLiteTensor *output_tensor_;    // model output tensor

    std::string model_path_; // tflite model path
    std::unordered_map<int, std::string>
        labelId_to_keyword_; // map to convert label_id to keyword
    std::vector<std::string> golden_keywords_;  // real keywords list
    int label_num_;          // label number
    float threshold_;        // threshold to filter predicton results

    std::string input_name_;  // model input tensor name
    std::string output_name_; // model output tensor name
};

#endif // kws.h