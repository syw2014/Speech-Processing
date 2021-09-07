/*
* @Author: your name
* @Date: 2021-08-09 14:13:00
* @LastEditTime: 2021-08-09 14:13:38
* @LastEditors: Please set LastEditors
* @Description: In User Settings Edit
* @FilePath: \deploy\cc\kws_unit_test.cc
*/
// Unit test of KWS module and MFCC feature extract module
// how to link static library in visual studio 
// ref:https://www.huaweicloud.com/articles/5cfd52b6cfed50a213315d020e3e25ea.html

#include "kws.h"
#include "CoreAudioCap.h"
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>
#include <time.h>

// link tflite static lib
//#pragma comment( lib, "E:/github/ASR/Speech-Processing/deploy/KeywordSpotting/kws-win-demo/libs/tensorflowlite_c.dll.if.lib" )
typedef void(CALLBACK *pCallBackAudioData_Out)(unsigned char* data, int length, void* user);
static mutex  g_Mutex_Capture;
static AudioList  g_DataList;         //数据列表
static int audio_pkg_cnt = 0;
									  // Define callback
void CALLBACK AudioCallBackProcess(unsigned char* data, int length, float fVolumeLevel, unsigned long dwtime, void* kws) {

	// Get number data block
	//std::cout << "Start to get audio from device...\n";
	//int audio_pkg_cnt = 100;
	{
		Audio_Data * pItem = NULL;
		if (!data)
			return;
		pItem = new Audio_Data();
		if (!pItem)
			return;

		pItem->iDataLen = length;
		pItem->fVolumelevel = fVolumeLevel;
		pItem->dwTime = dwtime;
		pItem->pData = new (std::nothrow) BYTE[length];


		if (pItem->pData)
		{
			memcpy_s(pItem->pData, length, data, length);
			{
				std::lock_guard<std::mutex> lock(g_Mutex_Capture);
				g_DataList.push_back(pItem);
			}
		}
		audio_pkg_cnt += 1;
	} while (0);
	//std::cout << "Finished get audio from device " << g_DataList.size() << std::endl;

	//--------------------------------------------------
	if(audio_pkg_cnt == 100)
	{
		// extract data from audo data list
		std::lock_guard<std::mutex> lock(g_Mutex_Capture);
		int buf_size = 0;
		// Extract 100 audio package as 1000ms audio for wakeup prediction
		std::string audio_pkg_str = "";
		for (auto index = g_DataList.begin(); index != g_DataList.end(); ++index) {
			std::string str((char*)(*index)->pData, (*index)->iDataLen);
			audio_pkg_str.append(str);
		}


		//std::cout << "Get audio data: " << buf_size << std::endl;
		// write final result to file
		std::ofstream outfs("./kws_test.txt", std::ios::app);
		if (!outfs.is_open()) {
			std::cout << "Open outfile report.txt error!\n";
			//return 1;
		}
		//outfs << std::fixed << std::setprecision(8);
		// Predict
		std::string keyword = "";
		float score = 0.0;
		bool wakeup = false;
		// Define variables for modules
		int label_id = -1;
		KWS* kws_ptr = (KWS*)kws;
		buf_size = audio_pkg_str.size();
		//std::cout << "ss=>" << buf_size << std::endl;
		// get date
		struct tm stime;
		time_t now = time(0);
		localtime_s(&stime, &now);
		char tmp[32] = { NULL };
		strftime(tmp, sizeof(tmp), "%Y-%m-%d_%H:%M:%S", &stime);
		std::string date(tmp);
		if (audio_pkg_str.size() != 0) {
			wakeup = kws_ptr->IsAwakenedWithPCM(audio_pkg_str.c_str(), buf_size, keyword, label_id, score);
			std::cout << "[INFO]Bot wakeup was: " << wakeup << "\tkeyword: " << keyword << "\tlabel_id: "
				<< label_id << "\tscore: " << score << std::endl;
			outfs << date << " [INFO]Bot wakeup was: " << wakeup << "\tkeyword: " << keyword << "\tlabel_id: "
				<< label_id << "\tscore: " << score << std::endl;
		}
		else {
			std::cout << date << " [WARNING]Get invalid audio data from device!\n";
			outfs << date << " [WARNING]Get invalid audio data from device!\n";
		}
		for (auto i = g_DataList.begin(); i != g_DataList.end(); i++)
		{
			delete[](*i)->pData;
			delete  (*i);
		}

		g_DataList.swap(std::list<Audio_Data*>());

		g_DataList.clear();
		//delete p;
		kws_ptr = NULL;
		outfs.close();
		audio_pkg_cnt = 0; // reset
	} // end wakeup block
}

// *NOTE* If you want to run this project please convert _main to main
// *You should define MDEBUG to open debug information*
int main()
{

	// Step1 define all parameters
	std::string model_path = "./kwsh5.tflite";
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

	//------------------------TEST5 prediction with audio get from device----------------------------//
	std::cout << "//----------------------TEST5 prediction with audio get from device--------------------------//\n";
	// Define audio capture
	CAudioCapT *capture_ptr;
	capture_ptr = NULL;

	// initialize
	// use do while format
	do {
		capture_ptr = new CAudioCapT();
		if (capture_ptr == NULL) {
			std::cout << "Initialize audio capture pointer error!\n";
			break;
		}

		// Set device
		capture_ptr->SetDeiveType(CAudioCapT::MICPHONE);

		// init
		if (!capture_ptr->Init()) {
			std::cout << "Capture init error!\n";
			break;
		}

	} while (0);

	// start 
	capture_ptr->Start();

	// register callback
	int n = 100;
	void* p = NULL;
	capture_ptr->RegistDataCallBack(AudioCallBackProcess, &kws);


	while (1)
	{

	}
	//capture_ptr->Stop();
	delete capture_ptr;
	capture_ptr = NULL;


	for (auto i = g_DataList.begin(); i != g_DataList.end(); i++)
	{
		delete[](*i)->pData;
		delete  (*i);
	}

	g_DataList.swap(std::list<Audio_Data*>());
	return 0;
}