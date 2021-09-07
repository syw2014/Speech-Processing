#include "CoreAudioCap.h"
#include <sstream>
#include <iostream>
#include <string>
#include <tchar.h>
using namespace std;

#include <wrl.h>
#include <propvarutil.h>
#include <mmdeviceapi.h> 
#include <endpointvolume.h>
#include <Audiopolicy.h>
#include <Mmdeviceapi.h>     // MMDevice
#include <mediaobj.h>        // IMediaObject
#include <functiondiscoverykeys_devpkey.h>
#include <mmsystem.h>
#include <strsafe.h>
#include <assert.h>
//#include "log.h"


#pragma comment(lib, "Avrt.lib")


#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  10000
#define AUDIO_SAMPLE_RATE 16000

#define SAFE_RELEASE(punk)  \
              if ((punk) != NULL)  \
		 { (punk)->Release(); (punk) = NULL; }




CAudioCapT::CAudioCapT()
{
	m_dwDataSize = 0;
	m_bInit = false;
	m_hThreadCapture = NULL;
	_OnDataCallBack = NULL;
	m_bStop = false;
	::InitializeCriticalSection(&m_cs);
}



CAudioCapT::~CAudioCapT()
{
	Audio_Data * pTmp = NULL;

	for (std::list<Audio_Data *>::iterator it = m_AudioListData.begin(); it != m_AudioListData.end(); it++)
	{
		pTmp = *it;
		if (pTmp)
		{
			if (pTmp->pData)
				delete[] pTmp->pData;
			delete pTmp;
		}
	}

	m_AudioListData.clear();
	::DeleteCriticalSection(&m_cs);
}


void CAudioCapT::_TraceCOMError(HRESULT hr, char* description = NULL) const
{
	TCHAR buf[MAXERRORLENGTH];
	TCHAR errorText[MAXERRORLENGTH];

	const DWORD dwFlags = FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS;
	const DWORD dwLangID = MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US);

	// Gets the system's human readable message string for this HRESULT.
	// All error message in English by default.
	DWORD messageLength = ::FormatMessageW(dwFlags,
		0,
		hr,
		dwLangID,
		errorText,
		MAXERRORLENGTH,
		NULL);

	assert(messageLength <= MAXERRORLENGTH);

	// Trims tailing white space (FormatMessage() leaves a trailing cr-lf.).
	for (; messageLength && ::isspace(errorText[messageLength - 1]);
		--messageLength)
	{
		errorText[messageLength - 1] = '\0';
	}

	StringCchPrintf(buf, MAXERRORLENGTH, TEXT("Error details: "));
	StringCchCat(buf, MAXERRORLENGTH, errorText);
	//ERRORLOG({0}, buf);
}

void CAudioCapT::RegistDataCallBack(pCallBackAudioData OnDataCallBack, void* userParameter)
{
	_OnDataCallBack = OnDataCallBack;
	_userPara = userParameter;

}




void CAudioCapT::SetDeiveType(int nType)
{
	m_nDeviceType = nType;
}


int CAudioCapT::GetDeviceType()
{
	return m_nDeviceType;
}

void CAudioCapT::OnCaptureData(LPBYTE pData, INT iDataLen, FLOAT fVolumelevel)
{
	Audio_Data * pItem = NULL;

	if (!pData)
		return;

	pItem = new Audio_Data();
	if (!pItem)
		return;

	pItem->iDataLen = iDataLen;
	pItem->pData = new (std::nothrow) BYTE[iDataLen];
	pItem->fVolumelevel = fVolumelevel;

	if (pItem->pData)
	{
		memcpy_s(pItem->pData, iDataLen, pData, iDataLen);
		::EnterCriticalSection(&m_cs);
		m_AudioListData.push_back(pItem);
		m_dwDataSize += iDataLen;
		::LeaveCriticalSection(&m_cs);
	}

	return;
}


Audio_Data * CAudioCapT::GetAudio()
{
	Audio_Data * pAudio = NULL;
	::EnterCriticalSection(&m_cs);
	if (m_AudioListData.empty() == false)
	{
		pAudio = m_AudioListData.front();
		m_AudioListData.pop_front();
	}
	::LeaveCriticalSection(&m_cs);
	return pAudio;
}

IMMDevice * CAudioCapT::GetDefaultDevice(int nType)
{
	IMMDevice *pDevice = nullptr;


	IMMDeviceEnumerator *pMMDeviceEnumerator = nullptr;
	HRESULT hr = CoCreateInstance(
		__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
		__uuidof(IMMDeviceEnumerator),
		(void**)&pMMDeviceEnumerator);
	if (FAILED(hr))
		return nullptr;

	IPropertyStore *pProps = NULL;
	LPWSTR pwszID = NULL;
	DWORD pdwState = 0;
	IMMDeviceCollection *pCollection = NULL;
	pMMDeviceEnumerator->EnumAudioEndpoints(EDataFlow::eCapture, DEVICE_STATEMASK_ALL, &pCollection);// eRender, eCapture, eAll
	std::cout << "EAAA!\n";
	if (nType == CAudioCapT::MICPHONE) {
		hr = pMMDeviceEnumerator->GetDefaultAudioEndpoint((EDataFlow)eCapture, eConsole, &pDevice);
		if (pDevice == nullptr) {
			std::cout << "Error!\n";
		}
	}
	else if (nType == CAudioCapT::SPEAKER)
		hr = pMMDeviceEnumerator->GetDefaultAudioEndpoint((EDataFlow)eRender, eConsole, &pDevice);
	else
		pDevice = nullptr;


	SAFE_RELEASE(pCollection);
	SAFE_RELEASE(pMMDeviceEnumerator);

	return pDevice;
}

bool CAudioCapT::Init()
{
	CoInitialize(NULL);

	if (m_bInit)
		return true;
	std::cout << "TEST-1\n";
	ClearAudioList();
	m_pDevice = GetDefaultDevice(m_nDeviceType);

	if (!m_pDevice)
	{
		return false;
	}

	m_hEventStarted = CreateEvent(nullptr, true, false, nullptr);

	if (m_hEventStarted == NULL)
	{
		return false;
	}
	m_bStop = false;
	m_dwDataSize = 0;
	m_bInit = true;

	return true;
}

bool CAudioCapT::SetMicPhoneVolume(int ivalue)
{
	HRESULT hr = 0;
	IMMDevice * pDevice = GetDefaultDevice(CAudioCapT::MICPHONE);
	if (!pDevice)
		return false;
	IAudioEndpointVolume *pAudioEndpointVolume = nullptr;
	hr = pDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, (void**)&pAudioEndpointVolume);
	if (FAILED(hr))
		return false;

	float fval = (float)ivalue / 100;

	pAudioEndpointVolume->SetMasterVolumeLevelScalar(fval, NULL);
	SAFE_RELEASE(pAudioEndpointVolume);
	if (pDevice)
	{
		pDevice->Release();
		pDevice = NULL;
	}
	return true;
}


bool CAudioCapT::GetMicPhoneVolume(int &ivalue)
{
	HRESULT hr = 0;
	IMMDevice * pDevice = GetDefaultDevice(CAudioCapT::MICPHONE);
	if (!pDevice)
		return false;
	IAudioEndpointVolume *pAudioEndpointVolume = nullptr;
	hr = pDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, (void**)&pAudioEndpointVolume);
	if (FAILED(hr))
		return false;

	float level = 0.0;
	pAudioEndpointVolume->GetMasterVolumeLevelScalar(&level);
	ivalue = (level * 100);
	SAFE_RELEASE(pAudioEndpointVolume);
	if (pDevice)
	{
		pDevice->Release();
		pDevice = NULL;
	}
	return true;

}


bool CAudioCapT::SetSpeekVolume(int ivalue)
{
	HRESULT hr = 0;
	IMMDevice * pDevice = GetDefaultDevice(CAudioCapT::SPEAKER);
	if (!pDevice)
		return false;
	IAudioEndpointVolume *pAudioEndpointVolume = nullptr;
	hr = pDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, (void**)&pAudioEndpointVolume);
	if (FAILED(hr))
		return false;

	float fval = (float)ivalue / 100;

	pAudioEndpointVolume->SetMasterVolumeLevelScalar(fval, NULL);
	SAFE_RELEASE(pAudioEndpointVolume);
	if (pDevice)
	{
		pDevice->Release();
		pDevice = NULL;
	}

	return true;
}


bool CAudioCapT::GetSpeekVolume(int &ivalue)
{
	HRESULT hr = 0;
	IMMDevice * pDevice = GetDefaultDevice(CAudioCapT::SPEAKER);
	if (!pDevice)
		return false;
	IAudioEndpointVolume *pAudioEndpointVolume = nullptr;
	hr = pDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, (void**)&pAudioEndpointVolume);
	if (FAILED(hr))
		return false;

	float level = 0.0;
	pAudioEndpointVolume->GetMasterVolumeLevelScalar(&level);
	ivalue = (level * 100);
	SAFE_RELEASE(pAudioEndpointVolume);
	if (pDevice)
	{
		pDevice->Release();
		pDevice = NULL;
	}

	return true;

}



DWORD CAudioCapT::GetDataSize()
{
	::EnterCriticalSection(&m_cs);
	DWORD dwSize = m_dwDataSize;
	::LeaveCriticalSection(&m_cs);
	return dwSize;
}

IMMDevice * CAudioCapT::GetDevice()
{
	return m_pDevice;
}

HANDLE CAudioCapT::GetStartEventHandle()
{
	return m_hEventStarted;
}

HANDLE CAudioCapT::GetStopEventHandle()
{
	return m_hEventStop;
}

AudioList * CAudioCapT::GetAudioList()
{
	return &m_AudioListData;
}

void CAudioCapT::ClearAudioList()
{
	::EnterCriticalSection(&m_cs);
	for (AudioList::iterator it = m_AudioListData.begin(); it != m_AudioListData.end(); it++)
	{
		byte * pData = (*it)->pData;
		if (pData)
		{
			delete[] pData;
		}
	}

	m_AudioListData.clear();
	::LeaveCriticalSection(&m_cs);
}


UINT __stdcall CAudioCapT::_CaptureThreadProc(LPVOID param)
{
	CAudioCapT * pObject = (CAudioCapT *)param;
	if (!pObject)
		return 0;

	HRESULT hr = 0;
	IAudioClient *pAudioClient = nullptr;
	WAVEFORMATEX *pWfx = nullptr;
	IAudioCaptureClient *pAudioCaptureClient = nullptr;
	IAudioEndpointVolume *pAudioEndpointVolume = nullptr;

	DWORD nTaskIndex = 0;
	HANDLE hTask = nullptr;
	bool bStarted(false);
	int nDeviceType = 0;
	IMMDevice * pDevice = pObject->GetDevice();
	HANDLE hEventStarted = pObject->GetStartEventHandle();

	REFERENCE_TIME hnsRequestedDuration = REFTIMES_PER_SEC;
	REFERENCE_TIME hnsActualDuration;
	UINT bufferFrameCount = 0;
	float Volumelevel = 0.0f;

	unsigned char* out_buf = new unsigned char[1024 * 10 *10];
	memset(out_buf, 0, 1024 * 10*10);

	if (!pDevice || !hEventStarted)
		return 0;

	CoInitialize(nullptr);

	do
	{
		hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
		if (FAILED(hr))
			break;

		hr = pDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, (void**)&pAudioEndpointVolume);
		if (FAILED(hr))
			break;

		pAudioEndpointVolume->SetMasterVolumeLevelScalar(0.822, NULL);  //设置最大音量
		hr = pAudioClient->GetMixFormat(&pWfx);
		if (FAILED(hr))
			break;


		SetEvent(hEventStarted);
		pObject->SetFormat(pWfx);
		pObject->m_pwfx = pWfx;

		pObject->m_Channels = pObject->m_pwfx->nChannels;
		pObject->m_SampleRate = pObject->m_pwfx->nSamplesPerSec;

		nDeviceType = pObject->GetDeviceType();
		DWORD err;
		if (nDeviceType == CAudioCapT::MICPHONE)
		{
			hr = pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 0, REFTIMES_PER_MILLISEC * 10, pWfx, 0);  //AUDCLNT_SHAREMODE_SHARED  共享方式   AUDCLNT_SHAREMODE_EXCLUSIVE 独占方式

		}
		else
		{
			break;
		}


		if (FAILED(hr))
		{
			//GenericLog(Trace, _T("_beginthreadex Initialize fail"));
			break;
		}

		hr = pAudioClient->GetBufferSize(&bufferFrameCount);

		if (FAILED(hr))
		{
			//GenericLog(Trace, _T("_beginthreadex GetBufferSize fail"));
			break;
		}


		hr = pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pAudioCaptureClient);

		if (FAILED(hr))
		{
			//GenericLog(Trace, _T("_beginthreadex GetService fail"));
			break;
		}

		hnsActualDuration = (double)REFTIMES_PER_SEC * bufferFrameCount / pWfx->nSamplesPerSec;

		if (nDeviceType == CAudioCapT::MICPHONE)
			hTask = AvSetMmThreadCharacteristics(_T("Audio"), &nTaskIndex);
		else
			hTask = AvSetMmThreadCharacteristics(_T("Capture"), &nTaskIndex);

		if (!hTask)
			break;

		hr = pAudioClient->Start();
		if (FAILED(hr))
		{
			//GenericLog(Trace, _T("_beginthreadex pAudioClient->Start() fail"));
			break;
		}

		bStarted = true;
		DWORD  dwWaitResult;
		UINT32 uiNextPacketSize(0);
		BYTE *pData = nullptr;
		UINT32 uiNumFramesToRead;
		DWORD dwFlags;

		while (pObject->m_bStop == false)
		{
			Sleep(hnsActualDuration / REFTIMES_PER_MILLISEC / 2);
			hr = pAudioCaptureClient->GetNextPacketSize(&uiNextPacketSize);
			if (FAILED(hr))
				break;

			while (uiNextPacketSize != 0)
			{
				hr = pAudioCaptureClient->GetBuffer(
					&pData,
					&uiNumFramesToRead,
					&dwFlags,
					nullptr,
					nullptr);

				if (FAILED(hr))
				{
					//GenericLog(Trace, _T("_beginthreadex pAudioCaptureClient fail"));
					pAudioCaptureClient->ReleaseBuffer(uiNumFramesToRead);
					break;
				}

				if (dwFlags & AUDCLNT_BUFFERFLAGS_SILENT)
				{
					pData = NULL;
				}

				if (pData&&uiNumFramesToRead>0)
				{					
						std::vector<BYTE> buffer;
						buffer.insert(buffer.end(), pData, pData + uiNumFramesToRead * pWfx->nBlockAlign);
						int icount = uiNumFramesToRead * pWfx->nBlockAlign;
						Volumelevel = pObject->GetPcmDB(pData, uiNumFramesToRead * pWfx->nBlockAlign);
						pObject->DataResample(buffer, 16000, pWfx->nSamplesPerSec);
						pObject->DataSingleChannel(buffer);
						if (NULL != pObject->_OnDataCallBack)
						{
							pObject->_OnDataCallBack(&buffer[0], buffer.size(), Volumelevel, GetTickCount(), pObject->_userPara);
						}
			
				}
				pAudioCaptureClient->ReleaseBuffer(uiNumFramesToRead);
				hr = pAudioCaptureClient->GetNextPacketSize(&uiNextPacketSize);

				if (FAILED(hr))
					break;
			}
		}

	} while (0);

	if (hTask)
	{
		AvRevertMmThreadCharacteristics(hTask);
		hTask = nullptr;
	}

	if (pWfx)
	{
		CoTaskMemFree(pWfx);
		pWfx = nullptr;
	}

	if (pAudioClient)
	{
		if (bStarted)
		{
			pAudioClient->Stop();
			pAudioClient->Reset();

		}
	}

	delete []out_buf;
	out_buf = NULL;

	SAFE_RELEASE(pAudioClient);
	SAFE_RELEASE(pAudioCaptureClient);

	CoUninitialize();

	return 0;
}



void CAudioCapT::SetFormat(WAVEFORMATEX * pwfx)
{
	if (!pwfx)
		return;

	if (pwfx->wFormatTag == WAVE_FORMAT_IEEE_FLOAT)
	{
		pwfx->wFormatTag = WAVE_FORMAT_PCM;
		pwfx->wBitsPerSample = 16;
		pwfx->nBlockAlign = pwfx->nChannels * pwfx->wBitsPerSample / 8;
		pwfx->nAvgBytesPerSec = pwfx->nBlockAlign * pwfx->nSamplesPerSec;
		//GenericLog(Trace, _T("****************** pwfx->nSamplesPerSec = %d\n"), pwfx->nSamplesPerSec);

	}
	else if (pwfx->wFormatTag == WAVE_FORMAT_EXTENSIBLE)
	{
		PWAVEFORMATEXTENSIBLE pEx = reinterpret_cast<PWAVEFORMATEXTENSIBLE>(pwfx);
		if (IsEqualGUID(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, pEx->SubFormat))
		{
			pEx->SubFormat = KSDATAFORMAT_SUBTYPE_PCM;
			pEx->Samples.wValidBitsPerSample = 16;
			pwfx->wBitsPerSample = 16;
			pwfx->nBlockAlign = pwfx->nChannels * pwfx->wBitsPerSample / 8;
			pwfx->nAvgBytesPerSec = pwfx->nBlockAlign * pwfx->nSamplesPerSec;
			//GenericLog(Trace, _T("############# pwfx->nSamplesPerSec = %d\n"), pwfx->nSamplesPerSec);
		}
	}

	memcpy(&m_WaveFormat, pwfx, sizeof(WAVEFORMATEX));

	return;
}

WAVEFORMATEX * CAudioCapT::GetWaveFormat()
{
	return &m_WaveFormat;
}



bool CAudioCapT::Start()
{
	if (!m_bInit)
		Init();

	if (m_hThreadCapture)
		return true;

	m_hThreadCapture = (HANDLE)_beginthreadex(nullptr, 0, _CaptureThreadProc, this, 0, nullptr);

	if (!m_hThreadCapture)
		return false;

	HANDLE ahWaits[2] = { m_hEventStarted, m_hThreadCapture };
	DWORD dwWaitResult = WaitForMultipleObjects(sizeof(ahWaits) / sizeof(ahWaits[0]), ahWaits, false, INFINITE);
	if (WAIT_OBJECT_0 != dwWaitResult)
	{
		if (m_hThreadCapture)
		{
			CloseHandle(m_hThreadCapture);
			m_hThreadCapture = NULL;
		}

		return false;
	}

	return true;
}

bool CAudioCapT::Stop()
{
	::CoUninitialize();
	if (m_bInit == false)
		return false;

	m_bStop = true;

	WaitForSingleObject(m_hThreadCapture, INFINITE);

	if (m_pDevice)
	{
		m_pDevice->Release();
		m_pDevice = NULL;
	}
	if (m_hEventStarted)
	{
		CloseHandle(m_hEventStarted);
		m_hEventStarted = NULL;
	}
	if (m_hThreadCapture)
	{
		CloseHandle(m_hThreadCapture);
		m_hThreadCapture = NULL;
	}

	m_bInit = false;

	return true;
}




int CAudioCapT::DataResample(std::vector<BYTE>& buffer, int rate, int src_amplesPerSec)
{
	if (m_SampleRate == rate)return buffer.size();
	if (m_pwfx == nullptr)return -1;
	std::vector<BYTE> resultBuffer;
	int bytes = m_pwfx->wBitsPerSample / 8;
	int sampleCount = buffer.size() / bytes;
	int srcRate = src_amplesPerSec;
	if (src_amplesPerSec == 44100)
		srcRate = 48000;

	int dstRate = rate;
	int rateLen = srcRate / dstRate;

	if (rateLen == 1) return buffer.size();

	if (rateLen > 0) {
		short tempRead = 0;
		int tempSum = 0;
		int flag = 0;

		for (int i = 0; i < sampleCount; i++) {
			memcpy(&tempRead, buffer.data() + i*bytes, bytes);
			tempSum = tempSum + tempRead;
			flag++;
			if (flag == rateLen)
			{
				flag = 0;
				tempSum = tempSum / rateLen;
				short tempSumex = (short)tempSum;
				resultBuffer.insert(resultBuffer.end(), ((BYTE*)&tempSumex), ((BYTE*)&tempSumex) + bytes);
				tempSum = 0;
			}
		}
	}
	else {
		rateLen = dstRate / srcRate;
		int tempRead1;
		int tempRead2;
		int tempSum;
		int tempAvgDiff;
		int tempWrite;
		int flag;

		for (int i = 0; i < (sampleCount - 1); i++) {
			memcpy(&tempRead1, buffer.data() + i * bytes, bytes);
			memcpy(&tempRead2, buffer.data() + i * bytes + bytes, bytes);
			tempSum = tempRead2 - tempRead1;
			tempAvgDiff = tempSum / rateLen;
			tempWrite = tempRead1;
			flag = rateLen;
			do
			{
				tempWrite += tempAvgDiff;
				resultBuffer.insert(resultBuffer.end(), ((BYTE*)&tempWrite), ((BYTE*)&tempWrite) + bytes);
			} while (--flag);
		}
	}
	buffer.swap(resultBuffer);
	return buffer.size();
}




int CAudioCapT::DataSingleChannel(std::vector<BYTE>& buffer)
{
	if (m_Channels == 1) return buffer.size();

	size_t len = buffer.size() / 2;
	int bytes = m_pwfx->wBitsPerSample / 8;
	BYTE *singleBuffer = new BYTE[len];
	for (int i = 0; i < len / bytes; i++)
	{
		memcpy(singleBuffer + i*bytes, buffer.data() + i*(2 * bytes), bytes);
	}

	buffer.assign(singleBuffer, singleBuffer + len);
	delete[] singleBuffer;
	return buffer.size();
}




int CAudioCapT::GetPcmDB(const unsigned char *pcmdata, size_t length)    //音量   
{

	int db = 0;
	short  value = 0;
	double sum = 0;
	short  maxvalue = 0;
	for (int i = 0; i < length; i += 2)
	{
		memcpy(&value, pcmdata + i, 2); //获取2个字节的大小（值）
		sum += abs(value); //绝对值求和

		if (abs(value) > maxvalue)
			maxvalue = abs(value);

	}
	sum = sum / (length / 2);    //求平均值（2个字节表示一个振幅，所以振幅个数为：size/2个）
	if (sum > 0)
	{
		db = (int)(20.0*log10(sum)) + 10;
	}
	return  db;

}



