/*
* @Author: your name
* @Date: 2021-08-09 14:13:00
* @LastEditTime: 2021-08-09 14:13:38
* @LastEditors: Please set LastEditors
* @Description: In User Settings Edit
* @FilePath: \deploy\cc\kws_unit_test.cc
*/
// Test Keyword Spotting inference with inpu wav file or wav folder, if input was wav folder then
// give the accuracy
#include "kws.h"
#include <unordered_map>


// link tflite static lib
//#pragma comment( lib, "E:/github/ASR/Speech-Processing/deploy/KeywordSpotting/x64/Release/tensorflow-lite.lib" )


// A simple option parser
char* GetCmdOption(char **begin, char **end, const std::string &value) {
	char **iter = std::find(begin, end, value);
	if (iter != end && ++iter != end) {
		return *iter;
	}
	return nullptr;
}

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

std::string GetWavLabelID(const std::string& filename) {
	// filename :003c8e86f8ef9bca4b49c753cb5e9ef5_1.wav,_1 was the real keyword id
	std::vector<std::string> vec;
	SplitWord(filename, vec, ".");
	if (vec.size() == 2) {
		std::vector<std::string> arr;
		SplitWord(vec[0], arr, "_");
		if (arr.size() == 2) {
			return arr[1];
		}
	}
	return "";
}


// Calculate the final result
void ConfusionMatrix(std::vector<int> &predictons, std::vector<int> &targets) {

	// calculate class number
	int num_labels = *max_element(targets.begin(), targets.end()) - *min_element(targets.begin(), targets.end()) + 1;
	// define 4x4 matrix
	std::vector<std::vector<int>> matrix(num_labels, std::vector<int>(num_labels, 0));
	int num_samples = targets.size();

	// initialize all the value as 0
	//for (int i = 0; i < num_labels; ++i) {
	//	for (int j = 0; j < num_labels; ++j) {
	//		matrix[i][j] = 0;
	//	}
	//}

	// Parse true label distrribution
	std::unordered_map<int, std::vector<int>> true_label_maps;
	for (int i = 0; i < num_samples; ++i) {
		if (true_label_maps.find(targets[i]) != true_label_maps.end()) {
			true_label_maps[targets[i]].push_back(i); // insert sample index to the same label
		}
		else {
			std::vector<int> v;
			v.push_back(i);
			true_label_maps[targets[i]] = v;
		}
	}
	//// print true label
	//for (int i = 0; i < true_label_maps.size(); ++i) {
	//	std::cout << "label " << i << "\t:";
	//	for (int j = 0; j < true_label_maps[i].size(); ++j) {
	//		std::cout << true_label_maps[i][j] << ",";
	//	}
	//	std::cout << std::endl;
	//}

	// Filling confusion matrix
	// positive total number, negative total number for calculate accuracy
	int pos_samples_num = 0, neg_samples_num = 0;
	for (int i = 0; i < true_label_maps.size(); ++i) {
		auto v = true_label_maps[i];
		if ((i == 0) || (i == 1)) { // 0: silence, 1:unknown
			neg_samples_num += v.size();
		}
		else if ((i == 2) || (i == 3)) {
			pos_samples_num += v.size();
		}
		for (int j = 0; j < v.size(); ++j) {
			// check prediction same as label
			// Note here `i` the real label

			if (predictons[j] == i) {
				matrix[i][i] += 1; // prediction == target
			}
			else {
				matrix[i][predictons[v[j]]] += 1; // prediction != target
			}
		}
	}

	// print matrix
	std::cout << "----------------------------------------------------\n";
	std::cout << "Confuse Matrix:\n";
	std::cout << "Label 0: silence 1: unknown 2: nihaoxiaoshun 3: xiaoshunxiaoshun\n";
	std::cout << "Label|\t\t0\t1\t2\t3\n";
	std::cout << "----------------------------------------------------\n";
	int acc_cnt = 0, precision_cnt = 0, w_cnt = 0;
	// Here we treat word1 and word2 as positive sample, silence and unknown as negative sample
	// So here we only calculate TP, FP, TN, FN, then we calculate accuracy, precision, recall, F1 score
	int tp = 0, fp = 0, tn = 0, fn = 0;
	double accuracy, precision;
	for (int i = 0; i < num_labels; ++i) {
		std::cout << "       " << i<<"|\t";
		for (int j = 0; j < num_labels; ++j) {
			std::cout << matrix[i][j] << "\t";
			
		}
		std::cout << "\n";
		// tn
		if ((i == 0) || (i == 1)) {
			tn += matrix[i][i];
		}
		// count tp
		if ((i == 2) || (i == 3)) {
			tp += matrix[i][i];
		}
	}

	std::cout << "----------------------------------------------------\n";
	// Calculate accuracy, precision, AUC, wakeup rate
	accuracy = (double)(tp+tn) / targets.size();
	precision = (double)tp / pos_samples_num;
	std::cout << "Accuracy for 4-labels: " << (tp+tn) << "/" << targets.size() << "=" << accuracy << std::endl;
	std::cout << "Precision for word1 and word2: " << (tp) << "/" << pos_samples_num << "=" << precision << std::endl;
	
}


int _main()
{

	// Step1 define all parameters
	std::string model_path = "E:/github/ASR/Speech-Processing/deploy/data/kwsh5.tflite";
	Params params_;
	// Parameter settings
	// 1) sample_rate
	params_.paramters["sample_rate"] = 16000;
	params_.paramters["window_size_ms"] = 30;
	params_.paramters["window_stride_ms"] = 10;
	params_.paramters["lower_frequency_limit"] = 20;
	params_.paramters["upper_frequency_limit"] = 4000;
	params_.paramters["filterbank_channel_count"] = 40;
	params_.paramters["dct_coefficient_count"] = 40;



	float threshold = 0.88;
	// Step2 create KWS module instance
	KWS kws;
	kws.ModelInitialize(model_path, params_, threshold);


	std::string keyword = "";
	float score = 0.0;
	bool is_wake = false;
	// Define variables for modules
	int label_id = -1;

	//------------------------TEST1 Prediction with WAV file--------------------------------//
	std::cout << "//------------------------TEST1 Prediction with WAV file--------------------------------//\n";
	std::string wav_file = "E:/github/ASR/Speech-Processing/deploy/data/47850987c8a227b92673f9a88fb7efb8.wav";
	kws.ProcessWavFile(wav_file, keyword, score, is_wake);


	//------------------------TEST4 prediction from wav file list----------------------------//
	std::cout << "//------------------------TEST prediction from wav file list----------------------------//\n";
	std::string wav_dir = "E:/github/ASR/tensorflow-lite-audio/tensorflow-lite/audios_data";
	std::string outfile = "./wav_list_result.txt";
	std::cout << "Start to inference with input wav folder: " << wav_dir << std::endl;
	// In the predict result, order was : [filenames, is_wake(0/1), keyword, score, pred_keyword_id]
	// id_to_word: 0:silence, 1:unknown, 2:nihaoxiaoshun, 3:xiaoshunxiaoshun
	std::vector<std::vector<std::string>> results;

	kws.ProcessWavFileList(wav_dir, results, outfile, threshold);

	// Calculate accuracy and wakeup rate under threshold
	int num_samples = results.size();
	std::vector<int> pred_ids(num_samples);
	std::vector<int> truth_ids(num_samples);
	std::vector<double> num_wakeup(num_samples);
	std::string real_word_id = "";
	double accuracy = 0.0;
	double precision = 0.0;
	int acc_count = 0;
	int prec_count = 0;
	for (int i = 0; i < results.size(); ++i) {
		pred_ids[i] = std::stoi(results[i][4]);	// predict keyword id
		truth_ids[i] = std::stoi(GetWavLabelID(results[i][0])); // wav file real keyword id
		
		// count wakeup 
		num_wakeup[i] = std::stod(results[i][1]);
		//if (results[i][3] == "1") {
		//	num_wakeup.push_back(results[i][1]);
		//}
	}

	// test confusion matrix
	//std::vector<int> pred_ids{ 0,0,2,2,0,2 };
	//std::vector<int> truth_ids{2,0,2,2,0,1};

	ConfusionMatrix(pred_ids, truth_ids);
	int wake_cnt = 0, real_wake_cnt = 0, wrong_wake_cnt  = 0;
	for (int i = 0; i < num_wakeup.size(); ++i) {
		if (num_wakeup[i] > threshold) { // score big than threshold regard as wakeup
			if (pred_ids[i] == truth_ids[i]) {
				// true keyword
				if ((truth_ids[i] == 2) || (truth_ids[i] == 3)) {
					real_wake_cnt += 1;
				}
				else {
					wrong_wake_cnt += 1; // wrong wake
				}
			}
			else {
				// wrong wake
				wrong_wake_cnt += 1;
			}
		}
	}
	std::cout << "----------------------------------------------------\n";
	std::cout << "True wakeup rate at threshold=" << threshold << "\t rate: " << (double)real_wake_cnt / (real_wake_cnt + wrong_wake_cnt) << std::endl;
	std::cout << "Error wakeup rate at threshold= " << threshold << " is: " << (double)wrong_wake_cnt / (real_wake_cnt + wrong_wake_cnt) << std::endl;
	while(1){}

	return 0;
}