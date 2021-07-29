/*
 * @Author: jerryshi
 * @Date: 2021-07-28 14:25:30
 * @LastEditTime: 2021-07-29 13:39:20
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\ksw\MFCC.cc
 */
#include "mfcc.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip> //For setprecision
#include <math.h>
#include<complex>

// Define pi as constant variable
const double MFCC::PI = 3.14159265358979323846;
const double MFCC::PI2 = 6.28318530717958647692;
const double MFCC::PI4 = 12.56637061435917295384;

MFCC::MFCC() : MFCC(16000, 16, 10, WindowMethods::Hamming, 40, 40) {
    // m_FrameSize must be a power of 2 !!
    // or modify MFCC::fft()
}

MFCC::MFCC(int SampleRate, int bitSize, int windowShift, WindowMethods window,
         int filterNum, int mfccDim) {
    m_Frequence = SampleRate;   // default 16000
    m_FrameSize = SampleRate * bitSize / 1000;  // default=16000 * 16 /1000=256
    m_FrameShift = SampleRate * windowShift / 1000; // default=16000 * 10/1000=160
    m_FilterNumber = filterNum; // default=40
    m_MFCCDim = mfccDim;        // default=40

    m_FrameCount = 0;
    m_CurrentFrame = 0;

    setWindowMethod(window);
    setFilterBank();
    setDCTCoeff();
    setLiftCoeff();
}

MFCC::~MFCC() {
    m_WindowCoefs.clear();
    m_FilterBank.clear();
    m_DCTCoeff.clear();
    m_CepLifter.clear();
    m_MFCCData.clear();
    m_RestData.clear();
}

size_t MFCC::FeatureExtarct(const short int data[], size_t sizeData) {
    std::vector<std::vector<double>> postData;

    ///*** Initialisation
    m_MFCCData.clear();
    m_CurrentFrame = 0;
    m_FrameCount = (sizeData - m_FrameSize + m_FrameShift) / m_FrameShift;
    m_MFCCData.resize(m_FrameCount);
    postData.resize(m_FrameCount);

    ///*** Apply the window coefficients
    for (size_t i = 0; i < m_FrameCount; i++) {
        for (int j = 0; j < m_FrameSize; j++) {
            postData[i].push_back(data[i * m_FrameShift + j] *
                                  m_WindowCoefs[j]);
            postData[i].push_back(0);
        }
    }

    ///*** Analyse
    internalAnalyse(postData, m_FrameCount, 0);
    postData.clear();

    return m_FrameCount;
}

void MFCC::internalAnalyse(std::vector<std::vector<double>> &postData, size_t frameCount,
                           size_t currrentFrame) {
    std::vector<std::vector<double>> spectraP;
    std::vector<std::vector<double>> melSpectraP;

    ///*** FFT matrix
    for (size_t i = 0; i < frameCount; i++)
        fft(postData[i], m_FrameSize, 1);

    ///*** Energy matrix
    spectraP.resize(frameCount);

    for (size_t i = 0; i < frameCount; i++)
        for (int j = 0; j < m_FrameSize / 2 + 1; j++)
            spectraP[i].push_back(postData[i][j << 1] * postData[i][j << 1] +
                                  postData[i][(j << 1) + 1] *
                                      postData[i][(j << 1) + 1]);

    melSpectraP.resize(m_FilterNumber);

    ///*** Apply filter bank
    for (int i = 0; i < m_FilterNumber; i++) {
        for (size_t k = 0; k < frameCount; k++) {
            melSpectraP[i].push_back(0);
            for (int j = 0; j < m_FrameSize / 2 + 1; j++)
                melSpectraP[i][k] += m_FilterBank[i][j] * spectraP[k][j];

            melSpectraP[i][k] = log(melSpectraP[i][k]);
        }
    }

    spectraP.clear();

    ///*** MFCC matrix
    for (size_t k = 0; k < frameCount; k++) {
        for (int i = 0; i < m_MFCCDim; i++) {
            m_MFCCData[currrentFrame + k].push_back(0);
            for (int j = 0; j < m_FilterNumber; j++)
                m_MFCCData[currrentFrame + k][i] +=
                    m_DCTCoeff[i][j] * melSpectraP[j][k];
        }
    }
    melSpectraP.clear();

    ///*** Ceplift
    for (size_t i = 0; i < frameCount; i++)
        for (int j = 0; j < m_MFCCDim; j++)
            m_MFCCData[currrentFrame + i][j] *= m_CepLifter[j];

    return;
}

bool MFCC::Save(const std::string &filePath) {
    size_t frameCount;
    std::ofstream outFile(filePath);
    if (!outFile.is_open())
        return false;

    outFile << std::fixed << std::setprecision(6);

    if (m_CurrentFrame > 0)
        frameCount = m_CurrentFrame;
    else
        frameCount = m_FrameCount;

    for (size_t i = 0; i < frameCount; i++) {
        for (int j = 0; j < m_MFCCDim; j++)
            outFile << m_MFCCData[i][j] << " ";
        outFile << std::endl;
    }

    outFile.close();
    return true;
}

const std::vector<std::vector<double>> &MFCC::GetMFCCData() {
    return m_MFCCData;
}


size_t MFCC::GetFrameCount() { return m_CurrentFrame; }

void MFCC::fft(std::vector<double> &data, int nn, int isign) {
    int i, j, m, n, mmax, istep;
    double wtemp, wr, wpr, wpi, wi, theta;
    double tempr, tempi;

    n = nn << 1;
    j = 1;
    for (i = 1; i < n; i += 2) {
        if (j > i) {
            std::swap(data[j - 1], data[i - 1]);
            std::swap(data[j], data[i]);
        }
        m = nn;

        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = isign * (PI2 / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= n; i += istep) {
                if (i >= n)
                    continue; // Buffer overflow if nn is not a power of 2
                j = i + mmax;
                if (j >= n)
                    continue; // Buffer overflow if nn is not a power of 2
                tempr = wr * data[j - 1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j - 1];
                data[j - 1] = data[i - 1] - tempr;
                data[j] = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }
            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

void MFCC::setWindowMethod(WindowMethods method) {
    m_WindowCoefs.clear();

    switch (method) {
    case WindowMethods::Hamming:
        for (int i = 0; i < m_FrameSize; i++)
            m_WindowCoefs.push_back(
                0.54 - 0.46 * (cos(PI2 * (double)i / (m_FrameSize))));
        break;

    case WindowMethods::Hann:
        for (int i = 0; i < m_FrameSize; i++)
            m_WindowCoefs.push_back(
                0.5 - 0.5 * (cos(PI2 * (double)i / (m_FrameSize - 1))));
        break;

    case WindowMethods::Blackman:
        for (int i = 0; i < m_FrameSize; i++)
            m_WindowCoefs.push_back(
                0.42 - 0.5 * (cos(PI2 * (double(i) / (m_FrameSize - 1)))) +
                0.08 * (cos(PI4 * (double(i) / (m_FrameSize - 1)))));
        break;

    case WindowMethods::None:
        for (int i = 0; i < m_FrameSize; i++)
            m_WindowCoefs.push_back(1);
        break;
    }
}

double MFCC::freq2mel(double freq) { return 1125 * log10(1 + freq / 700); }

double MFCC::mel2freq(double mel) { return (pow(10, mel / 1125) - 1) * 700; }

void MFCC::setFilterBank() {
    // http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-MFCCs/
    double maxMelf, deltaMelf;
    // lowFreq: 20, highFreq: 4000 the same as it in tensorflow
    double lowFreq, mediumFreq, highFreq, currentFreq;
    int filterSize = m_FrameSize / 2 + 1;

    maxMelf = freq2mel(m_Frequence / 4);   // mel(4000)
    deltaMelf = maxMelf / (m_FilterNumber + 1);

    m_FilterBank.resize(m_FilterNumber);
    lowFreq = mel2freq(0);
    mediumFreq = mel2freq(deltaMelf);
    for (int i = 0; i < m_FilterNumber; i++) {
        highFreq = mel2freq(deltaMelf * (i + 2));

        for (int j = 0; j < filterSize; j++) {
            currentFreq = (j * 1.0 / (filterSize - 1) * (m_Frequence / 4));

            if ((currentFreq >= lowFreq) && (currentFreq <= mediumFreq))
                m_FilterBank[i].push_back(2 * (currentFreq - lowFreq) /
                                          (mediumFreq - lowFreq));
            else if ((currentFreq >= mediumFreq) && (currentFreq <= highFreq))
                m_FilterBank[i].push_back(2 * (highFreq - currentFreq) /
                                          (highFreq - mediumFreq));
            else
                m_FilterBank[i].push_back(0);
        }

        lowFreq = mediumFreq;
        mediumFreq = highFreq;
    }
    // Algorithm2
    // double lowFreqMel = freq2mel(20);
    // double highFreqMel = freq2mel(4000);
    // m_FilterBank.resize(m_FilterNumber+2);
    // for (int i = 0; i < m_FilterNumber+2; ++i){
    //     for(int j = 0; j < filterSize; j++) {
    //         m_FilterBank[i].push_back(mel2freq(lowFreqMel+(highFreqMel-lowFreqMel)/(m_FilterNumber+1)*i));
    //     }
    // }

}

void MFCC::setDCTCoeff() {
    m_DCTCoeff.resize(m_MFCCDim);
    // TODO, implementaion1
    for (int i = 0; i < m_MFCCDim; i++)
        for (int j = 0; j < m_FilterNumber; j++)
            m_DCTCoeff[i].push_back(
                2 * cos((PI * (i + 1) * (2 * j + 1)) / (2 * m_FilterNumber)));
    
    // TODO, algorithm2
    // double c = sqrt(2.0/m_FilterNumber);
    // std::vector<double> v1(m_MFCCDim+1, 0), v2(m_FilterNumber, 0);
    // for(int i = 0; i < m_MFCCDim; ++i) v1[i] = i;
    // for(int j = 0; j < m_FilterNumber; ++j) v2[j] = j +0.5;

    // for(int i = 0; i < m_MFCCDim; ++i) {
    //     for(int j = 0; j < m_FilterNumber; ++j) {
    //         m_DCTCoeff[i].push_back(c * cos(PI / m_FilterNumber * v1[i] * v2[j]));
    //     }
    // }

}

void MFCC::setLiftCoeff() {
    for (int i = 0; i < m_MFCCDim; i++)
        m_CepLifter.push_back(
            (1.0 + 0.5 * m_MFCCDim * sin(PI * (i + 1) / (m_MFCCDim))) /
            ((double)1.0 + 0.5 * m_MFCCDim));
}