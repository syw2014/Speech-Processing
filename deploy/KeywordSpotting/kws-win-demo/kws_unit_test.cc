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
//#pragma comment( lib, "E:/github/ASR/Speech-Processing/deploy/KeywordSpotting/kws-win-demo/libs/tensorflowlite_c.dll.if.lib" )

// *NOTE* If you want to run this project please convert _main to main
// *You should define MDEBUG to open debug information*
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


	//-------------------------TEST2 Prediction with PCM File-------------------------------//
	std::cout << "//-------------------------TEST2 Prediction with PCM File-------------------------------//\n";
	// Test read data from PCM FILE just read int16_t data from pcm file
	std::string pcm_file = "E:/github/ASR/Speech-Processing/deploy/data/xiaoshunxiaoshun.PCM";
	std::cout << "Load PCM data from " << pcm_file << std::endl;
	std::ifstream inFile(pcm_file, std::ifstream::in | std::ifstream::binary);
	uint16_t bufferLength = 16000;
	int16_t *buffer = new int16_t[bufferLength];
	int bufferBPS = (sizeof buffer[0]);

	// Read all data into float_values
	// Sample data convert to -1.0~1.0
	inFile.read((char *)buffer, bufferLength * bufferBPS);
	inFile.close();

	//std::cout << "Load PCM data completed!\n";
	// Copy pointer data to vector
	std::vector<int16_t> data{buffer, buffer+bufferLength};
	bool wakeup = false;
	wakeup = kws.IsAwakenedWithAudio(data, keyword, label_id, score);
	std::cout << "Bot wakeup was:  " << wakeup << " keyword: " << keyword << " label_id: " << label_id << " score: " << score ;
	delete[] buffer;
	buffer = nullptr;


	//-----------------------TEST3 Predicton with PCM Char string format ---------------------//
	std::cout << "//-----------------------TEST3 Predicton with PCM Char string format ---------------------//\n";
	// Test PCM `IsAwakenedWithPCM` interface, which was load PCM format data as char string, then convert it to int16_t
	std::cout << "Load PCM data from " << pcm_file << std::endl;
	inFile.open(pcm_file, std::ifstream::in | std::ifstream::binary);
	if (!inFile.is_open()) {
		std::cout << "Can not open the WAV file !!" << std::endl;
	}

	std::ostringstream ostr;
	ostr << inFile.rdbuf();
	std::string pcm_char_str(ostr.str());
	int buf_size = pcm_char_str.size();
#ifdef MDEBUG
	std::cout << "size: " << buf_size << std::endl;
#endif
	inFile.close();
	wakeup = kws.IsAwakenedWithPCM(pcm_char_str.c_str(), buf_size, keyword, label_id, score);
	std::cout << "Bot wakeup was:  " << wakeup << " keyword: " << keyword << " label_id: " << label_id << " score: " << score;
	while (1) { 
		//kws.IsAwakenedWithFile(wav_file, keyword, score, is_wake); 
	}

	//------------------------TEST4 prediction from wav file list----------------------------//
	std::cout << "//------------------------TEST4 prediction from wav file list----------------------------//\n";
	//std::string wav_dir = "E:/github/ASR/tensorflow-lite-audio/tensorflow-lite/audios_data";
	//std::string outfile = "./wav_list_result.txt";
	//std::vector<std::vector<std::string>> results;

	//kws.ProcessWavFileList(wav_dir, results, outfile);



	return 0;
}