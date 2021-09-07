#pragma once
#include <windows.h>
#include <windowsx.h>
#include <mmdeviceapi.h>
#include <Audioclient.h>
#include <process.h>
#include <avrt.h>
#include <list>
#include "vector"
#include "kws.h"

using namespace  std; 

typedef void(CALLBACK *pCallBackAudioData)(unsigned char* data, int length, float fVolumeLevel, unsigned long dwtime,void* user);

typedef struct TADUIO_DATA
{
	INT		iDataLen;
	LPBYTE	pData;
	FLOAT   fVolumelevel;
	DWORD   dwTime;
}Audio_Data, * PAudio_Data;

typedef std::list<PAudio_Data> AudioList;

class CAudioCapT
{
public:

	enum {SPEAKER = 1,MICPHONE};

public:
	CAudioCapT();
	~CAudioCapT();

	bool Init();
	bool Start(); 
	bool Stop(); 
	void RegistDataCallBack(pCallBackAudioData OnDataCallBack, void* userParameter);
	void SetDeiveType(int nType);

	bool SetMicPhoneVolume(int ivalue);
	bool GetMicPhoneVolume(int &ivalue);

	bool SetSpeekVolume(int ivalue);
	bool GetSpeekVolume(int &ivalue);




private:
	static UINT __stdcall _CaptureThreadProc(LPVOID param);
	void            OnCaptureData(LPBYTE pData, INT iDataLen, FLOAT fVolumelevel);
	IMMDevice*      GetDefaultDevice(int nType);  
	Audio_Data*     GetAudio();
	IMMDevice*      GetDevice();
	HANDLE          GetStartEventHandle();
	HANDLE          GetStopEventHandle();
	WAVEFORMATEX*   GetWaveFormat();
	AudioList*      GetAudioList();


	void  SetFormat(WAVEFORMATEX * wf);
	int   DataResample(std::vector<BYTE> &buffer, int rate,int src_amplesPerSec);
	int   DataSingleChannel(std::vector<BYTE> &buffer);
	int   GetPcmDB(const unsigned char *pcmdata, size_t size);
	void  _TraceCOMError(HRESULT hr, char* description /*= NULL*/) const;
	int   GetDeviceType();
	void  ClearAudioList();
	DWORD GetDataSize();

private:
	bool           m_bInit;
	HANDLE         m_hThreadCapture;
	WAVEFORMATEX   m_WaveFormat;
	int            m_nDeviceType = 0;
	AudioList      m_AudioListData;
	CRITICAL_SECTION m_cs;
	IMMDevice*     m_pDevice = NULL;
	HANDLE         m_hEventStarted = NULL;
	HANDLE         m_hEventStop = NULL;
	DWORD          m_dwDataSize;
	bool           m_bStop;
	WAVEFORMATEX * m_pwfx;
	pCallBackAudioData       _OnDataCallBack;
	void*                    _userPara;

	int m_SampleRate;
	int m_Channels;
};

