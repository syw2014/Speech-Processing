/*
 * @Author: your name
 * @Date: 2021-08-05 10:14:40
 * @LastEditTime: 2021-08-09 14:05:39
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\kws.cc
 */
#include "kws.h"

// Construct
KWS::KWS() {}

// De-Construct
KWS::~KWS() {
    // delete pointer
    if (interpreter_ != NULL) {
        delete interpreter_;
        interpreter_ = NULL;
    }

    if (input_tensor_ != NULL) {
        delete input_tensor_;
        input_tensor_ = NULL;
    }

    if (output_tensor_ != NULL) {
        delete output_tensor_;
        output_tensor_ = NULL;
    }
}

size_t KWS::ModelInitialize(std::string &model_path, Params &params,
                            float threshold) {

    // Define label id map to keyword
    labelId_to_keyword_[0] = "silence";
    labelId_to_keyword_[1] = "unknown";
    labelId_to_keyword_[2] = "nihaoxiaoshun";
    labelId_to_keyword_[3] = "xiaoshunxiaoshun";

    // label count,default 3
    label_num_ = labelId_to_keyword_.size();

    // Golden keywords to check wheather audio was keyword
    golden_keywords_.push_back("nihaoxiaoshun");
    golden_keywords_.push_back("xiaoshunxiaoshun");

    // threshold to filter result
    threshold_ = threshold;

    // Initialize featrue extractor
    feature_extractor_.Initialize(params);

    // Initialize interpreter
    TfLiteModel *model = TfLiteModelCreateFromFile(model_path_.c_str());
    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    interpreter_ = TfLiteInterpreterCreate(model, options);
    if (interpreter_ == nullptr) {
        std::cout << "Create interpreter failed!\n";
        // TODO error code
        return 1;
    }

    // Allocate tensor buffers
    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
        std::cout << "Failed to allocate tensors!\n";
        return 1;
    }

    // get input and output tensor for model input and output
    input_name_ = "input";
    output_name_ = "Identity";
    GetOutputTensorByName(output_name_.c_str());
    GetInputTensorByName(input_name_.c_str());

    return 0;
}

// Get output tensor from Tflite model
size_t KWS::GetOutputTensorByName(const char *name) {
    int count = TfLiteInterpreterGetOutputTensorCount(interpreter_);
    for(int i = 0; i < count; ++i) {
        TfLiteTensor * tensor = (TfLiteTensor*)TfLiteInterpreterGetOutputTensor(interpreter_, i);
        if(!(strcmp(tensor->name, name))) {
            output_tensor_ = tensor;
            return 0;
        }
    }
    return NULL;
}

// Get input tensor from Tflite model
size_t KWS::GetInputTensorByName(const char *name) {
    int count = TfLiteInterpreterGetInputTensorCount(interpreter_);
    for(int i = 0; i < count; ++i) {
        TfLiteTensor *tensor = (TfLiteTensor*)TfLiteInterpreterGetInputTensor(interpreter_,i);
        if(!strcmp(tensor->name, name)) {
            input_tensor_ = tensor;
            return 0;
        }
    }
    return NULL;
}

// Model inference
size_t KWS::Inference(const std::vector<float> &features, const int &feature_length,
                 std::vector<std::pair<int, float>> &logits) {
    // Copy data from vector to tensor
	TfLiteTensorCopyFromBuffer(input_tensor_, features.data(), feature_length * sizeof(float));
	TfLiteInterpreterInvoke(interpreter_);
	float prediction[label_num_];
    logits.resize(label_num_);
    // Copy data from tensor to buffer
	TfLiteTensorCopyToBuffer(output_tensor_, prediction, label_num_ * sizeof(float));
    for(int i = 0; i < label_num_; ++i) {
        logits.push_back(std::make_pair(i, prediction[i]));
    }

	// float maxV = -1;
	// int maxIdx = -1;
	// for (int i = 0; i < 4; ++i) {
	// 	if (logits[i] > maxV) {
	// 		maxV = logits[i];
	// 		maxIdx = i;
	// 	}
	// }
	// cout << u8"类别：" << maxIdx << u8"，概率：" << maxV << endl;
	return 0;
}

// Extract audio features for raw audio get from miscphone
size_t KWS::GetFeatures(std::vector<int16_t> &audio_data,
                   std::vector<float> &features) {
    // TODO, audio pre-process 

    size_t ret = 0;

    // Audio normalize to -1.0~1.0
    std::vector<double> norm_samples;
    ret = feature_extractor_.AudioDataNorm(audio_data, norm_samples);

    // Calculate mfcc features
    std::vector<std::vector<double>> mfcc_features;
    ret = feature_extractor_.ExtractFeatures(norm_samples, mfcc_features);

    // Check feature dimension and Flat feature as 1-dimension vector
    size_t feat_size = mfcc_features.size() * mfcc_features[0].size();
    std::string feat_name = "feature_length";
    int value;
    ret = feature_extractor_.GetParameters(feat_name, value);
    if(feat_size != value) {
        std::cout << "[ERROR]: feature dimension: " << feat_size << "was not the same as setting: "
                << value << std::endl;
    }

    // Flat 2-D feature(vector<vector<double>>) as 1-D feature(vector<float>)
    features.resize(feat_size); 
    for(int i; i < mfcc_features.size(); ++i) {
        for(int j=0; j < mfcc_features[i].size(); ++j) {
            // TODO, here convert double to float which used in tflite inference
            features[i] = mfcc_features[i][j];
        }
    }
    return ret;
}

// Get keywords
size_t KWS::ParseLogits(std::vector<std::pair<int, float>> &logits, std::string &keyword,
                   int &label_id, float &score) {
    // Check empty results
    int logit_size = logits.size();
    if (logit_size == 0 && logit_size != label_num_) {
        std::cout << "[ERROR] label prediction error!\n";
        return 1;
    }
    float max_score = -1.0;
    int max_label_id = -1;
    for(int i = 0; i < logit_size; ++i) {
        if(logits[i].second > max_score) {
            max_score = logits[i].second;
            max_label_id = logits[i].first;
        }
        std::cout << "Prediction label: " << logits[i].first << "\t score: " << logits[i].second << std::endl;
    }

    // Get the keywords
    if(max_label_id != -1 && labelId_to_keyword_.find(max_label_id) != labelId_to_keyword_.end()) {
        keyword = labelId_to_keyword_[max_label_id];
        label_id = max_label_id;
        score = max_score;
    } else {
        keyword = "";
        label_id = -1;
        score = 0.0;
    }
    return 0;
}

// Check result is keyword
// This is main entrance of application, input audio data, return the wake or not
bool KWS::IsAwakened(std::vector<int16_t> &audio_samples,std::string &keyword, float &score, float threshold) {
    
    // Step1, audio data convert to features
    std::vector<float> features;
    GetFeatures(audio_samples, features);

    // Step2, predict
    std::vector<std::pair<int, float>> logits;
    Inference(features, features.size(), logits);

    // Step3, result parse
    std::string keyword;
    int label_id;
    float score;
    ParseLogits(logits, keyword, label_id, score);

    // Step4, check weather wake or not
    bool ret = false;
    if (score >= threshold) {
        std::cout << "Bot was wake, keyword: " << keyword << " score: " << score << std::endl;
        ret = true;
    }

    return ret;
}