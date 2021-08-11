/*
 * @Author: your name
 * @Date: 2021-08-09 14:13:00
 * @LastEditTime: 2021-08-09 14:13:38
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\kws_unit_test.cc
 */
// Unit test of KWS module and MFCC feature extract module


#include "stdafx.h"
#include "kws.h"


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
	

	// Step3 FeatureExtract Class unit test
	// Step3-1, FeatureExtract::Initialize
	// Step3-2, FeatureExtract::ReadWav
	// Step3-3, FeatureExtract::AudioDataNorm
	// Step3-4, FeatureExtract::ExtractFeatures
	// Step3-5, FeatureExtract::ProcessSingleWav
	// Step3-6, FeatureExtract::MfccFlat

	// Step4 KWS class unit test
	// Step4-1, KWS::ModelInitialize
	// Step4-1, KWS::GetFeatures
	// Step4-1, KWS::Inference
	// Step4-1, KWS::IsAwakenedWithAudio
	// Step4-1, KWS::IsAwakenedWithFeature
	// Step4-1, KWS::ProcessWavFile
	// Step4-1, KWS::ProcessWavFileList

	
	return 0;
}