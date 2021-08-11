/*
 * @Author: your name
 * @Date: 2021-07-21 14:37:21
 * @LastEditTime: 2021-08-03 09:55:22
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\compute-mfcc-master\wavHeader.h
 * ref:https://zhuanlan.zhihu.com/p/44483056
 */
#ifndef WAV_HEADER_H
#define WAV_HEADER_H

#include <iostream>
#include <stdint.h>

// -----------------------------------------------------------------------------
//  Header file for wave file header
// -----------------------------------------------------------------------------

struct WAVHeader {
	/* RIFF Chunk Descriptor */
	uint8_t         RIFF[4];        // RIFF Header Magic header,4字节大端序。文件从此处开始，对于WAV或AVI文件，其值总为“RIFF”
	uint32_t        ChunkSize;      // RIFF Chunk Size,4字节小端序。表示文件总字节数减8，减去的8字节表示,ChunkID与ChunkSize本身所占字节数
	uint8_t         WAVE[4];        // WAVE Header,4字节大端序。对于WAV文件，其值总为“WAVE”
									/* "fmt" sub-chunk */
	uint8_t         fmt[4];         // FMT header, 4字节大端序。其值总为“fmt ”，表示Format Chunk从此处开始
	uint32_t        Subchunk1Size;  // Size of the fmt chunk,4字节小端序。表示Format Chunk的总字节数减8
	uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM,2字节小端序
	uint16_t        NumOfChannels;      // Number of channels 1=Mono 2=Stereo,2字节小端序
	uint32_t        SamplesPerSec;  // Sampling Frequency in Hz,4字节小端序,表示在每个通道上每秒包含多少帧
	uint32_t        bytesPerSec;    // bytes per second,4字节小端序。大小等于SampleRate * BlockAlign，表示每秒共包含多少字节
	uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo,2字节小端序。大小等于NumChannels * BitsPerSample / 8， 表示每帧的多通道总字节数
	uint16_t        bitsPerSample;  // Number of bits per sample,2字节小端序。表示每帧包含多少比特
									/* "data" sub-chunk */
	uint8_t         Subchunk2ID[4]; // "data"  string,4字节大端序。其值总为“data”，表示Data Chunk从此处开始
	uint32_t        Subchunk2Size;  // Sampled data length, 4字节小端序。表示data的总字节数


};


#endif // wavHeader.h