/*
 * @Author: your name
 * @Date: 2021-08-05 10:14:40
 * @LastEditTime: 2021-08-09 15:19:41
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\kws.cc
 */
//#include "stdafx.h"
#include "kws.h"

template <typename T>
std::string ToString(T Num)
{
    std::ostringstream oss;
    oss<<Num;
    std::string str(oss.str());
    return str;
}

// Construct
KWS::KWS() {}

// De-Construct
KWS::~KWS() {
    // delete pointer
    if (interpreter_ != nullptr) {
        //delete interpreter_;
        interpreter_ = nullptr;
    }

    if (input_tensor_ != nullptr) {
        //delete input_tensor_;
        input_tensor_ = nullptr;
    }

    if (output_tensor_ != nullptr) {
        //delete output_tensor_;
        output_tensor_ = nullptr;
    }
}

size_t KWS::ModelInitialize(std::string &model_path, Params &params,
                            float threshold) {

    // Define label id map to keyword
    labelId_to_keyword_[0] = "silence";
    labelId_to_keyword_[1] = "unknown";
    labelId_to_keyword_[2] = "nihaoxiaoshun";
    labelId_to_keyword_[3] = "xiaoshunxiaoshun";

    // label count,default 4, there are only two keywords
    label_num_ = labelId_to_keyword_.size();

    // Golden keywords to check wheather audio was keyword
    golden_keywords_.push_back("nihaoxiaoshun");
    golden_keywords_.push_back("xiaoshunxiaoshun");

    // threshold to filter result
    threshold_ = threshold;
	model_path_ = model_path;

    // Initialize featrue extractor
    feature_extractor_.Initialize(params);

    // Initialize interpreter
    TfLiteModel *model = TfLiteModelCreateFromFile(model_path_.c_str());
    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    interpreter_ = TfLiteInterpreterCreate(model, options);
    if (interpreter_ == nullptr) {
        std::cout << " [ERROR]: Create interpreter failed!\n";
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
	//input_tensor_ = nullptr;
	//output_tensor_ = nullptr;
    GetOutputTensorByName(output_name_.c_str());
    GetInputTensorByName(input_name_.c_str());

	//delete model;
	model = nullptr;

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
    return 1;
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
    return 1;
}

// Model inference
size_t KWS::Inference(const std::vector<float> &features, const size_t &feature_length,
                 std::vector<std::pair<int, float>> &logits) {
	logits.clear();
    // Copy data from vector to tensor
	TfLiteTensorCopyFromBuffer(input_tensor_, features.data(), feature_length * sizeof(float));
	TfLiteInterpreterInvoke(interpreter_);
	// TODO, here 4 is the number of label, so you should know the right number before inference
	float prediction[4];
    logits.resize(label_num_);
    // Copy data from tensor to buffer
	TfLiteTensorCopyToBuffer(output_tensor_, prediction, label_num_ * sizeof(float));
    for(int i = 0; i < label_num_; ++i) {
        logits[i] = std::make_pair(i, prediction[i]);
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
	std::cout << "Norm completed!\n";
    // Calculate mfcc features
    std::vector<std::vector<double>> mfcc_features;
    ret = feature_extractor_.ExtractFeatures(norm_samples, mfcc_features);
	std::cout << "feature completed!\n";
    // Check feature dimension and Flat feature as 1-dimension vector
    size_t feat_size = mfcc_features.size() * mfcc_features[0].size();
	std::cout << "feature size: " << feat_size << std::endl;
    std::string feat_name = "feature_length";
    int value;
    ret = feature_extractor_.GetParameters(feat_name, value);
    if(feat_size != value) {
        std::cout << "[ERROR]: feature dimension: " << feat_size << "was not the same as setting: "
                << value << std::endl;
    }

    // Flat 2-D feature(vector<vector<double>>) as 1-D feature(vector<float>)
    features.resize(feat_size); 
    for(int i=0; i < mfcc_features.size(); ++i) {
        for(int j=0; j < mfcc_features[i].size(); ++j) {
            // TODO, here convert double to float which used in tflite inference
            size_t index = i*mfcc_features[i].size() + j;
            features[index] = mfcc_features[i][j];
        }
    }
    return ret;
}

// Get keywords
size_t KWS::ParseLogits(std::vector<std::pair<int, float>> &logits, std::string &keyword,
                   int &label_id, float &score) {
    // Check empty results
    size_t logit_size = logits.size();
    if ((logit_size == 0) && (logit_size != label_num_)) {
        std::cout << "[ERROR] label prediction error!\n";
        return 1;
    }
    float max_score = -1.0;
    int max_label_id = -1;
	std::cout << "[DEBUG] logit size: " << logit_size << "\t label_num:" << label_num_ <<std::endl;
    for(int i = 0; i < logit_size; ++i) {
        if(logits[i].second > max_score) {
            max_score = logits[i].second;
            max_label_id = logits[i].first;
        }
        std::cout << "Prediction label: " << logits[i].first << "\tscore: " << logits[i].second << std::endl;
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


// Check prediction was keyword
bool KWS::IsKeyword(const std::string& word) {
	if (word == "") {
		return false;
	}
	if (std::find(golden_keywords_.begin(), golden_keywords_.end(), word) != golden_keywords_.end()) {
		return true;
	}
	return false;
}

// Check result is keyword
// This is main entrance of application, input audio data, return the wake or not
bool KWS::IsAwakenedWithAudio(std::vector<int16_t> &audio_samples,std::string &keyword, int& label_id, float &score, float threshold) {
    
    // Step1, audio data convert to features
    std::vector<float> features;
    GetFeatures(audio_samples, features);
	std::cout << "normalized finished!\n";

    // Step2, predict
    std::vector<std::pair<int, float>> logits;
    Inference(features, features.size(), logits);

    // Step3, result parse
    //std::string keyword;
    //int label_id;
    //float score;
    ParseLogits(logits, keyword, label_id, score);

    // Step4, check weather wake or not
    bool ret = false;
    if (IsKeyword(keyword) && (score >= threshold)) {
        std::cout << "Bot was wake, keyword: " << keyword << " score: " << score << std::endl;
        ret = true;
    }

    return ret;
}

// Check wakeup or not
// Process PCM format audio data
bool KWS::IsAwakenedWithPCM(const char* pcm_data, int pcm_length, std::string &keyword, int& label_id, float &score, float threshold) {

	bool ret = false;
	std::vector<int16_t> audio_data;
	// TODO convert char* string to int16_t vector
	// parse PCM format data to int16_t vector, to use stringstream
	int16_t t;
	for (int i = 0; i < pcm_length ; i += 2) {
		memcpy(&t, pcm_data + i, 2); // int16_t was 2bytes, PCM char string was 1byte
		audio_data.push_back(t);
	}

	std::cout << "Parsed PCM data completed total size: " << audio_data.size() << std::endl;

	// Prediction
	ret = IsAwakenedWithAudio(audio_data, keyword, label_id, score, threshold);

	return ret;
}

// Check result is keyword
// This is main entrance of application, input audio data, return the wake or not
bool KWS::IsAwakenedWithFeature(std::vector<float>& features,std::string &keyword, int& label_id, float &score, float threshold) {
    
    // Step1, predict
    std::vector<std::pair<int, float>> logits;
    Inference(features, features.size(), logits);

    // Step3, result parse
    //std::string keyword;
    //int label_id;
    //float score;
    ParseLogits(logits, keyword, label_id, score);

    // Step4, check weather wake or not
    bool ret = false;
    if (IsKeyword(keyword) && (score >= threshold)) {
        std::cout << "Bot was wakeup, keyword: " << keyword << " score: " << score << std::endl;
        ret = true;
    }

    return ret;
}


// Check wakup with wav file
bool KWS::IsAwakenedWithFile(std::string& wav_file, std::string &keyword, int& label_id, float &score, bool is_wake) {
	// Load wav file
	std::string outfile = "";
	bool write_to_file = false;
	std::vector<std::vector<double>> mfcc_features;
	feature_extractor_.ProcessSingleWav(wav_file, outfile, false, mfcc_features);

	// mfcc feature flat
	std::vector<float> feature;
	feature_extractor_.MfccFlat(mfcc_features, feature);

	// predict
	//int label_id;
	is_wake = IsAwakenedWithFeature(feature, keyword, label_id, score, 0.85);
	std::cout << "Final result Process file: " << wav_file << " is_wake: " << is_wake <<
		" keyword: " << keyword << " score: " << score << std::endl;

	return is_wake;
}

// Process wav file
size_t KWS::ProcessWavFile(std::string& wav_file, std::string &keyword, float &score, bool is_wake) {
    // Load wav file
    std::string outfile = "";
    bool write_to_file = false;
    std::vector<std::vector<double>> mfcc_features;
    feature_extractor_.ProcessSingleWav(wav_file, outfile, false, mfcc_features);

    // mfcc feature flat
    std::vector<float> feature;
    feature_extractor_.MfccFlat(mfcc_features, feature);

    // predict
	int label_id;
    is_wake = IsAwakenedWithFeature(feature, keyword, label_id, score, 0.85);
    std::cout <<"[DEBUG]Final result Process file: " << wav_file << " is_wake: " << is_wake <<
        " keyword: " << keyword << " score: " << score << std::endl;

    return 0;
}

// Process wav file list
// TODO: Opitimize process logict, read wav and prediction, current was read all wav, calculate all wav mfcc feature, then prediction
size_t KWS::ProcessWavFileList(std::string& wav_dir, std::vector<std::vector<std::string>>& results, std::string& outfile) {
    
    std::string out_folder = "";
    bool write_to_file = false;
    //std::vector<std::vector<std::vector<double>>> mfcc_feature_list;
    std::vector<std::string> filenames;

	// Get wav file list from directory
	DIR *pDir;
	struct dirent *ptr;
	std::vector<std::string> files;
	//std::vector<std::string> outfiles;

	// *NOTE*, wav file name format like 5338ca0367ec5ef0d43244cdae31dda7.wav_2
	if (!(pDir = opendir(wav_dir.c_str()))) {
		perror(("Folder " + wav_dir + "doesn't exist!").c_str());
		return 1;
	}
	while ((ptr = readdir(pDir)) != 0) {
		if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
			// 5338ca0367ec5ef0d43244cdae31dda7.wav_2
			files.push_back(wav_dir + "/" + ptr->d_name);
			filenames.push_back(ptr->d_name);
			// mfcc.5338ca0367ec5ef0d43244cdae31dda7.wav_2
			//outfiles.push_back(out_folder + "/" + "mfcc." + ptr->d_name);
		}
	}
	closedir(pDir);
	pDir = nullptr;

	std::cout << "[DEBUG]Start to prediction with wav list toal file: " << files.size() << std::endl;
    // predict
    //std::vector<float> feature;
    // store final result, order was: is_wake, keyword, score
    std::vector<std::string> prediction; 
    prediction.resize(4);
    bool is_wake = false;
    std::string keyword = "";
    float score = 0.0;
	
	// write final result to file
	std::ofstream outfs(outfile);
	if (!outfs.is_open()) {
		std::cout << "Open outfile " << outfile << "error!\n";
		return 1;
	}
	outfs << std::fixed << std::setprecision(8);
	results.resize(files.size());
	Clock::time_point TStart, TEnd;
	TStart = Clock::now();
	int label_id;
    for(int i = 0; i< files.size(); ++i) {
        //feature.clear();
		keyword = "";
		score = 0.0;
		
        // predict
        is_wake = IsAwakenedWithFile(files[i], keyword, label_id,score, 0.85);

		// write to file
		outfs << filenames[i] << "\t" << ToString(is_wake) << "\t" << keyword << "\t" << label_id
			<< "\t" << ToString(score) << std::endl;
        // Assembly results
        prediction[0] = filenames[i]; // file name
        prediction[1] = ToString(is_wake);    // is_wake
        prediction[2] = keyword;  // keyword
        prediction[3] = ToString(score);  // score
        results[i] = prediction;
		prediction.clear();
		prediction.resize(4);
    }
	TEnd = Clock::now();
	Milliseconds ms = std::chrono::duration_cast<Milliseconds>(TEnd - TStart);
	std::cout << "Wav File list prediction completed cost time: "
		<< ms.count() << "ms" << std::endl;

	outfs.close();
    return 0;
}