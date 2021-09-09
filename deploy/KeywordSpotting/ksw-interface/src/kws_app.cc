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
#include <atlstr.h>
#include <stdlib.h>


// link tflite static lib
//#pragma comment( lib, "E:/github/ASR/Speech-Processing/deploy/KeywordSpotting/kws-win-demo/libs/tensorflowlite_c.dll.if.lib" )
typedef void(CALLBACK *pCallBackAudioData_Out)(unsigned char* data, int length, void* user);
static mutex  g_Mutex_Capture;
static AudioList  g_DataList;         //数据列表
static int audio_pkg_cnt = 0;

// calculate audio DB
int CalculatePcmDB(const char* pcm_data, std::size_t length) {
	int db = 0;
	short  value = 0;
	double sum = 0;
	short  maxvalue = 0;
	for (int i = 0; i < length; i += 2)
	{
		memcpy(&value, pcm_data + i, 2); //获取2个字节的大小（值）
		sum += abs(value); //绝对值求和

		if (abs(value) > maxvalue)
			maxvalue = abs(value);

	}
	sum = sum / (length / 2);    //求平均值（2个字节表示一个振幅，所以振幅个数为：size/2个）
	if (sum > 0)
	{
		db = (int)(20.0*log10(sum)) + 10;
	}

	return db;
}

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


		// Check voice volume
		int db = CalculatePcmDB(audio_pkg_str.c_str(), audio_pkg_str.length());
		
		// write final result to file
		std::ofstream outfs("./kws_test.txt", std::ios::app);
		if (!outfs.is_open()) {
			std::cout << "Open outfile report.txt error!\n";
			//return 1;
		}
		//outfs << std::fixed << std::setprecision(8);
		DWORD ttime = GetTickCount();
		// get date
		struct tm stime;
		time_t now = time(0);
		localtime_s(&stime, &now);
		char tmp[32] = { NULL };
		strftime(tmp, sizeof(tmp), "%Y-%m-%d_%H:%M:%S", &stime);
		std::string date(tmp);
		if (abs(db) >= 40) { // volume was > 40db then predict with KWS model

			//std::cout << "Get audio data: " << buf_size << std::endl;
			// Predict
			std::string keyword = "";
			float score = 0.0;
			bool wakeup = false;
			// Define variables for modules
			int label_id = -1;
			KWS* kws_ptr = (KWS*)kws;
			buf_size = audio_pkg_str.size();

			if (audio_pkg_str.size() != 0) {
				wakeup = kws_ptr->IsAwakenedWithPCM(audio_pkg_str.c_str(), buf_size, keyword, label_id, score);
				std::cout << "[INFO]Bot wakeup was: " << wakeup << "\tkeyword: " << keyword << "\tlabel_id: "
					<< label_id << "\tscore: " << score << "\tdb:" << db << std::endl;
				outfs << ttime << "-"<<date << " [INFO]Bot wakeup was: " << wakeup << "\tkeyword: " << 
					keyword << "\tlabel_id: "
					<< label_id << "\tscore: " << score << "\tdb:" <<db << std::endl;
			}
			else {
				std::cout << date << " [ERROR]Get invalid audio data from device!\n";
				outfs << ttime << "-" << date 
					<< " [ERROR]Get invalid audio data from device: 0\tkeyword: None" <<
					"\tlabel_id: -1\tscore: 0\tdb: 0.0" << std::endl;
			}

			// write auido to pcm file
			FILE * fDst = NULL;
			unsigned char  *tempdata = (unsigned char *)audio_pkg_str.c_str();
			int iLength = audio_pkg_str.length();

			CString szfilename;
			szfilename.Format(_T("./data/test_%d.pcm"), ttime);
			_tfopen_s(&fDst, szfilename, _T("wb+"));

			int nTotal = 0;

			while (nTotal < iLength)
			{
				int nTmp = fwrite(tempdata + nTotal, 1, iLength - nTotal, fDst);
				nTotal += nTmp;
			}
			fclose(fDst);
			kws_ptr = NULL;
		}			// end voice check
		else { // print log
			std::cout << "[WARNING]Voice volume was: " << db << ",Wake up not triggered" << std::endl;
			outfs << ttime << "-" << date << " [WARNING]volume of voice was < 40\tkeyword: None" <<
				"\tlabel_id: -1\tscore: 0\tdb: "<<db << std::endl;
		}

		

		// clean data 
		for (auto i = g_DataList.begin(); i != g_DataList.end(); i++)
		{
			delete[](*i)->pData;
			delete  (*i);
		}

		g_DataList.swap(std::list<Audio_Data*>());

		g_DataList.clear();
		//delete p;
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