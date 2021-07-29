/*
 * @Author: jerryshi
 * @Date: 2021-07-28 13:56:20
 * @LastEditTime: 2021-07-28 15:18:55
 * @LastEditors: Please set LastEditors
 * @Description: Audio MFCC extract
 * @FilePath: \deploy\ksw\mfcc.h
 */
/* Mel-frequency cepstral coefficients extract class.
TODO: add the process logic
*/
#ifndef MFCC_H
#define MFCC_H

#include <iostream>
#include <string>
#include <vector>

class MFCC {
  public:
    // Define the window types ,default use Hamming
    enum WindowMethods { Hamming, Hann, Blackman, None };

    // Construction
    MFCC();

    /**
     * @description: MFCC class object
     * @param SampleRate: 采样频率，默认：16000
     * @param bitSize: bit数，默认：16bit
     * @param windowShift: 窗口移动数，默认，10ms
     * @param window: 加窗类型，默认：hanmming
     * @param filterNum: filter bank 数, 默认：40
     * @param mfccDim:每一帧抽取后mfcc的维度，默认是40，与tensorflow 保持一致
     * @return None
     */   
    MFCC(int SampleRate, int bitSize, int windowShift, WindowMethods window,
         int filterNum, int mfccDim);

    // De-constructor
    virtual ~MFCC();

    /**
     * @description: MFCC feature extraction entrance
     * @param audioData: input audio data int array
     * @param dataSize: input the size of audio data array
     * @return frame number in this audio
     */
    size_t FeatureExtarct(const short int audioData[], size_t dataSize);

    /**
     * @description: Save mfcc features in the given file
     * @param fileName: output file name
     * @return True if write succeed otherwise false
     */    
    bool Save(const std::string &fileName);

    /**
     * @description: Get the final mfcc features
     * @param None
     * @return std::vector<std::vector<double>>
     */    
    const std::vector<std::vector<double>> &GetMFCCData();

    /**
     * @description: Get the total frame number
     * @param {*}
     * @return {*}
     */    
    size_t GetFrameCount();

  private:
    /**
     * @description: 快速傅里叶变换
     * @param {vector<double>} &data：input audio data(以frame为单位)
     * @param {int} nn：frame length
     * @param {int} isign: where is sign or not
     * @return None
     */    
    void fft(std::vector<double> &data, int nn, int isign);

    /**
     * @description: 这是窗类型
     * @param {WindowMethods} method: 窗类型，默认是hamming
     * @return None
     */    
    void setWindowMethod(WindowMethods method);

    /**
     * @description: 设置filter bank
     * @param {*}
     * @return {*}
     */    
    void setFilterBank();

    /**
     * @description: 设置离散余弦变换系数
     * @param {*}
     * @return {*}
     */    
    void setDCTCoeff();

    /**
     * @description: 
     * @param {*}
     * @return {*}
     */    
    void setLiftCoeff();

    /**
     * @description: 
     * @param {*}
     * @return {*}
     */    
    void internalAnalyse(std::vector<std::vector<double>> &postData,
                         size_t frameCount, size_t currrentFrame);

    /**
     * @description: 
     * @param {double} freq：采样频率
     * @return {*}
     */    
    double freq2mel(double freq);

    /**
     * @description: 
     * @param {double} mel：mel值
     * @return {*}
     */    
    double mel2freq(double mel);

    // Settings
    int m_Frequence;    // sample_rate
    int m_FrameSize;    // frame number
    int m_FrameShift;   // frame shift samples
    int m_FilterNumber; // filter numbers
    int m_MFCCDim;      // final mfcc dimension

    // Internal
    size_t m_FrameCount;    // how many frame in audio
    std::vector<double> m_WindowCoefs;  // window coefficient
    std::vector<std::vector<double>> m_FilterBank;  // filter bank matrix
    std::vector<std::vector<double>> m_DCTCoeff;    // dct coefficient matrix
    std::vector<double> m_CepLifter;                // 
    std::vector<std::vector<double>> m_MFCCData;    // final mfcc feature matrix

    size_t m_CurrentFrame;      // the index of current frame
    std::vector<short int> m_RestData;

    // Constantes
    static const double PI;
    static const double PI2;
    static const double PI4;
};

#endif // end mfcc.h
