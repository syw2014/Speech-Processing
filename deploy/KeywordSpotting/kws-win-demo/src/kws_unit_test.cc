/*
 * @Author: your name
 * @Date: 2021-08-09 14:13:00
 * @LastEditTime: 2021-08-09 14:13:38
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\kws_unit_test.cc
 */
// Unit test of KWS module and MFCC feature extract module
// how to link static library in visual studio ref:https://www.huaweicloud.com/articles/5cfd52b6cfed50a213315d020e3e25ea.html

#include "kws.h"


// link tflite static lib
#pragma comment( lib, "E:/github/ASR/Speech-Processing/deploy/KeywordSpotting/kws-win-demo/libs/tensorflowlite_c.dll.if.lib" )

int main()
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
	std::string wav_file = "E:/github/ASR/Speech-Processing/deploy/data/47850987c8a227b92673f9a88fb7efb8.wav";


	// Step2 create KWS module instance
	KWS kws;
	kws.ModelInitialize(model_path, params_, threshold);

	std::string keyword = "";
	float score = 0.0;
	bool is_wake = false;
	kws.ProcessWavFile(wav_file, keyword, score, is_wake);
	

	//while (1) { 
	//	kws.IsAwakenedWithFile(wav_file, keyword, score, is_wake); 
	//}

	// test wav file list
	std::string wav_dir = "E:/github/ASR/tensorflow-lite-audio/tensorflow-lite/audios_data";
	std::string outfile = "./wav_list_result.txt";
	std::vector<std::vector<std::string>> results;

	kws.ProcessWavFileList(wav_dir, results, outfile);



	return 0;
}